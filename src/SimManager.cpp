#include "SimManager.h"
#include "System/PatchManager.h"
#include <cstdio>
#include <fstream>
#include <random>

namespace ARBD {

//================================================================================
// Constructor
//================================================================================

SimManager::SimManager(SimSystem& sys) : sys_(sys), sys_state_(sys) {
	timer0_.timer = wkf_timer_create();
	timerS_.timer = wkf_timer_create();
	timerE_.timer = wkf_timer_create();
	timerP_.timer = wkf_timer_create();
}

SimManager::SimManager(SimSystem& sys, const ConfigParser& parser) : sys_(sys), sys_state_(sys) {
	load_config(parser);
	timer0_.timer = wkf_timer_create();
	timerS_.timer = wkf_timer_create();
	timerE_.timer = wkf_timer_create();
	timerP_.timer = wkf_timer_create();
}

//================================================================================
// Initialization
//================================================================================

void SimManager::load_config(const ConfigParser& parser) {
	sys_.set_temperature(parser.get_sim_system().get_temperature());
	sys_.set_cutoff(parser.get_sim_system().get_cutoff());
	sys_.set_timestep(parser.get_sim_system().get_timestep());
	sys_.set_num_steps(parser.get_sim_system().get_num_steps());
	sys_.set_neighbor_list_rebuild_period(
		parser.get_sim_system().get_neighbor_list_rebuild_period());
	sys_.set_output_period(parser.get_sim_system().get_output_period());
	sys_.set_energy_output_period(parser.get_sim_system().get_energy_output_period());
	sys_.set_output_name(parser.get_sim_system().get_output_name());
	sys_.set_output_format(parser.get_sim_system().get_output_format());
	sys_.set_decomposer_type(parser.get_sim_system().get_decomposer_type());
	sys_.set_long_range_method(parser.get_sim_system().get_long_range_method());
	sys_.set_particle_integrator_type(parser.get_sim_system().get_particle_algorithm());
	sys_.set_rigid_body_integrator_type(parser.get_sim_system().get_rigid_body_algorithm());
	sys_.set_particle_types(parser.get_sim_system().get_particle_types());
	sys_.set_rigid_body_types(parser.get_sim_system().get_rigid_body_types());
	sys_.set_base_seed(parser.get_sim_system().get_base_seed());

	// Cache particles here; they are converted (type_name -> type_id) in init(),
	// once particle type IDs have been assigned - SystemState::set_init_particle_data()
	// would otherwise look up type names in an empty/unbuilt map.
	pending_initial_particles_ = parser.get_init_particles();
	sys_state_.update_bonded_interactions(parser.get_init_bonded_interactions());
}

void SimManager::init() {
	LOGINFO("SimManager: Initializing simulation");

	// Initialize output writers based on configuration
	initialize_output_writers();
	// Populate configuration.

	// Initialize decomposer if not already set
	if (!sys_.get_decomposer()) {
		LOGINFO("SimManager: Setting up default spatial decomposer");
		sys_.set_decomposer_type(sys_.get_decomposer_type());
	}

	// Particle type IDs and name->id maps must exist before any particle data
	// referencing type names (by name) can be converted, so this must run
	// before set_init_particle_data() below and before decomposition.
	sys_.assign_particle_type_ids();
	LOGINFO("SimManager: Particle type IDs assigned");
	sys_.build_name_to_id_maps();
	LOGINFO("SimManager: Name to ID maps built");

	// Load cached initial particle data (from load_config or set_initial_particles)
	if (!pending_initial_particles_.empty()) {
		sys_state_.set_init_particle_data(pending_initial_particles_);
		LOGINFO("SimManager: Loaded {} initial particles into system state",
				pending_initial_particles_.size());
	}

	// Load cached bonded interactions (from load_config or set_bonded_interactions)
	if (pending_bonded_interactions_.get_num_bonds() > 0 ||
		pending_bonded_interactions_.get_num_angles() > 0 ||
		pending_bonded_interactions_.get_num_dihedrals() > 0) {
		sys_state_.update_bonded_interactions(pending_bonded_interactions_);
		LOGINFO("SimManager: Loaded {} bonds, {} angles, {} dihedrals, {} exclusions into system "
				"state",
				pending_bonded_interactions_.get_num_bonds(),
				pending_bonded_interactions_.get_num_angles(),
				pending_bonded_interactions_.get_num_dihedrals(),
				pending_bonded_interactions_.get_exclusions().size());
	}

	// Perform domain decomposition (creates PatchManager in SimSystem)
	LOGINFO("SimManager: Performing domain decomposition");
	sys_.decompose_system(sys_state_);

	// Verify PatchManager was created
	if (!sys_.has_patch_manager()) {
		throw Exception(ExceptionType::RuntimeError,
						SourceLocation(),
						"Domain decomposition failed to create PatchManager");
	}
	LOGINFO("SimManager: Domain decomposition complete");

	// Distribute initial particles from system state into patches (device storage)
	sys_.get_patch_manager()->distribute_particles_from_state(sys_state_);
	LOGINFO("SimManager: Initial particles distributed to patches");

	// Transfer grids to all GPU resources
	sys_.get_grid_manager().build_device_arrays();
	LOGINFO("SimManager: Grids transferred to all resources");
	sys_.get_tables_registry().build_device_arrays();
	LOGINFO("SimManager: Tables transferred to all resources");
	sys_.get_nonbonded_interactions().prepare_device_data();
	LOGINFO("SimManager: Nonbonded interactions transferred to all resources");

	LOGINFO("SimManager: Initialization completed");
}

//================================================================================
// Main Simulation Loop
//================================================================================

void SimManager::run() {
	LOGINFO("SimManager: Starting simulation loop");

	// Get simulation parameters
	const size_t num_steps = sys_.get_num_steps();
	const size_t output_period = static_cast<size_t>(sys_.get_output_period());
	const size_t energy_output_period = static_cast<size_t>(sys_.get_energy_output_period());
	const auto& resources = sys_.get_resources();

	LOGINFO("SimManager: Running {} steps with {} resources", num_steps, resources.size());

	const size_t progress_period =
		energy_output_period > 0 ? energy_output_period : 1000;
	const int num_replicas = 1; // TODO: expose replicas from SimSystem config

	std::printf("Configuration: %zu particles | %d replicas\n",
				sys_state_.get_num_particles(),
				num_replicas);
	std::fflush(stdout);

	wkf_timer_start(timer0_.timer);
	wkf_timer_start(timerP_.timer);

	for (size_t step = 1; step <= num_steps; ++step) {
		// ===== FORCE CALCULATION PHASE =====
		execute_force_calculation(step);

		// ===== INTEGRATION PHASE =====
		execute_integration(step);

		// ===== MULTI-RESOURCE SYNCHRONIZATION =====
		if (resources.size() > 1) {
			synchronize_multi_resource();
		}

		// ===== OUTPUT PHASE =====
		handle_output(step);

		// ===== IMD HANDLING =====
		if (imd_on_ && clientsock_) {
			handle_imd_commands();
		}

		// ===== PROGRESS REPORTING =====
		if (step % progress_period == 0) {
			report_progress(step, num_steps, progress_period);
		}
	}

	// ===== FINALIZATION =====
	std::printf("\n");
	std::fflush(stdout);

	wkf_timer_stop(timer0_.timer);
	const float elapsed = wkf_timer_time(timer0_.timer);

	report_performance(elapsed, num_steps);
	write_final_restart();

	// Cleanup IMD
	if (imd_on_ && clientsock_) {
		// TODO: imd_disconnect(clientsock_);
	}

	LOGINFO("SimManager: Simulation completed successfully");
}

//================================================================================
// Output Writers Initialization
//================================================================================

void SimManager::initialize_output_writers() {
	// TODO: Get output format and name from SimSystem when accessors are added
	// For now, default to DCD format
	std::string output_name = sys_.get_output_name();

	dcd_writer_ = std::make_unique<DcdWriter>(output_name + ".dcd");
	LOGINFO("SimManager: Initialized DCD writer for '{}.dcd'", output_name);

	// TODO: Add support for other output formats (PDB, HDF5) when needed

	// Momentum trajectory/restart only make sense for Langevin dynamics
	// (matches legacy ARBD's particle_dynamic == "Langevin" gating).
	has_momentum_output_ = (sys_.get_particle_algorithm() == IntegratorType::Langevin);
	if (has_momentum_output_) {
		momentum_dcd_writer_ = std::make_unique<DcdWriter>(output_name + ".0.momentum.dcd");
		LOGINFO("SimManager: Initialized momentum DCD writer for '{}.0.momentum.dcd'", output_name);
	}

	has_rigid_bodies_ = !sys_.get_rigid_body_types().empty();
}

void SimManager::initialize_imd(int port) {
	LOGINFO("SimManager: IMD initialization (port {}) not yet implemented", port);
	// TODO: Implement IMD when needed
}

//================================================================================
// Force Calculation Phase
//================================================================================

void SimManager::execute_force_calculation(size_t step) {
	PatchManager* patch_mgr = sys_.get_patch_manager();
	if (!patch_mgr) {
		throw Exception(ExceptionType::RuntimeError,
						SourceLocation(),
						"PatchManager not available for force calculation");
	}

	const auto& resources = sys_.get_resources();
	const auto& grid_manager = sys_.get_grid_manager();

	if (device_particle_types_cache_.size() < resources.size()) {
		device_particle_types_cache_.resize(resources.size());
	}

	for (auto& patch : patch_mgr->get_patches()) {
		size_t resource_idx = 0;
		for (size_t i = 0; i < resources.size(); ++i) {
			if (resources[i] == patch->get_resource()) {
				resource_idx = i;
				break;
			}
		}

		// Particle types are static for the whole run - build the device
		// table once per resource instead of re-allocating (13 device
		// buffers) and re-uploading it on every step.
		if (!device_particle_types_cache_[resource_idx]) {
			device_particle_types_cache_[resource_idx] =
				std::make_unique<DeviceParticleTypes>(sys_.get_particle_types(),
													  patch->get_resource());
		}
		DeviceParticleTypes& particle_types = *device_particle_types_cache_[resource_idx];

		// calculate_nonbonded_forces() clears the force buffer and writes the
		// PMF/grid + pairwise nonbonded terms; calculate_bonded_forces() must
		// run after it and accumulate on top (it only ever atomic-adds, never
		// clears).
		//
		// Energy accumulation adds an extra atomic write (ForceEnergy.t)
		// alongside the 3 force-component atomics on every pairwise-neighbor
		// and bonded-term evaluation, so it's only enabled on the same steps
		// handle_output() will actually report energy for - matching legacy
		// ARBD, which likewise only computes energy every outputEnergyPeriod
		// steps rather than every step.
		const size_t energy_output_period = static_cast<size_t>(sys_.get_energy_output_period());
		const bool compute_energy =
			energy_output_period > 0 && step % energy_output_period == 0;
		Event evt = patch->calculate_nonbonded_forces(
			sys_.get_nonbonded_interactions(),
			sys_state_.get_bonded_interactions(),
			particle_types,
			grid_manager.get_device_grid_views(resource_idx),
			sys_.get_tables_registry(),
			resource_idx,
			static_cast<float>(sys_.get_pairlist_cutoff()),
			step,
			static_cast<size_t>(sys_.get_neighbor_list_rebuild_period()),
			0.0f,
			1,
			compute_energy);

		Event bonded_evt = patch->calculate_bonded_forces(sys_state_.get_bonded_interactions(),
														  particle_types,
														  sys_.get_tables_registry(),
														  resource_idx,
														  compute_energy);

		// Both events were previously discarded, relying on the (removed)
		// per-step DeviceParticleTypes malloc/free to accidentally serialize
		// the pipeline. Without that, forces from this patch must be waited
		// on explicitly before integrate_motion() reads ForceEnergy, and
		// before the next step's clear_forces()/PMF kernel overwrites it.
		bonded_evt.wait();
		evt.wait();
	}

	if (step == 1) {
		LOGINFO("SimManager: PMF/grid and pairwise nonbonded force kernels launched");
	}
}

//================================================================================
// Integration Phase
//================================================================================

void SimManager::execute_integration(size_t step) {
	const IntegratorType particle_algorithm = sys_.get_particle_algorithm();
	const IntegratorType rigidbody_algorithm = sys_.get_rigid_body_algorithm();
	const float timestep = sys_.get_timestep();
	const Temperature& temperature = sys_.get_temperature_struct();

	// Access PatchManager through SimSystem
	PatchManager* patch_mgr = sys_.get_patch_manager();
	if (!patch_mgr) {
		throw Exception(ExceptionType::RuntimeError,
						SourceLocation(),
						"PatchManager not available for integration");
	}

	// Execute integration for each patch
	// Each patch runs independently on its assigned resource
	for (auto& patch : patch_mgr->get_patches()) {
		Event evt = patch->integrate_motion(timestep, temperature, particle_algorithm, step);
		// Must complete before the next step's clear_forces()/PMF kernel
		// touches positions again - see the matching wait in
		// execute_force_calculation for why this can no longer be implicit.
		evt.wait();
	}

	current_step_ = step; // Update step counter for RNG state tracking
}

//================================================================================
// Multi-Resource Synchronization
//================================================================================

void SimManager::synchronize_multi_resource() {
	PatchManager* patch_mgr = sys_.get_patch_manager();
	if (!patch_mgr) {
		LOGWARN("SimManager: Cannot synchronize - PatchManager not available");
		return;
	}

	// TODO: Implement halo exchange through PatchManager
	// patch_mgr->exchange_halos();

	// Placeholder
	static bool logged_once = false;
	if (!logged_once) {
		LOGINFO("SimManager: Multi-resource synchronization not yet implemented");
		logged_once = true;
	}
}

//================================================================================
// Output Handling
//================================================================================

void SimManager::handle_output(size_t step) {
	const size_t output_period = static_cast<size_t>(sys_.get_output_period());
	const size_t energy_output_period = static_cast<size_t>(sys_.get_energy_output_period());

	// Trajectory output
	if (output_period > 0 && step % output_period == 0) {
		wkf_timer_start(timerS_.timer);

		if (dcd_writer_) {
			write_dcd_frame(step);
			// Reuses the SystemState write_dcd_frame() just gathered/synced.
			if (momentum_dcd_writer_) {
				write_momentum_dcd_frame(step);
			}
		} else if (traj_writer_) {
			// Write with generic trajectory writer
			// traj_writer_->write_frame(step);
		}

		wkf_timer_stop(timerS_.timer);
	}

	// Energy calculation and output (also refreshes the restart files, mirroring
	// legacy ARBD's cadence of writing restarts alongside the energy report).
	if (energy_output_period > 0 && step % energy_output_period == 0) {
		wkf_timer_start(timerE_.timer);
		write_energy_output(step);
		wkf_timer_stop(timerE_.timer);
	}
}

void SimManager::gather_particle_data_from_patches() {
	PatchManager* patch_mgr = sys_.get_patch_manager();
	sys_state_.clear_global_arrays();
	if (!patch_mgr) {
		throw Exception(ExceptionType::RuntimeError,
						SourceLocation(),
						"PatchManager not available for gathering particle data");
	}
	patch_mgr->gather_particles_to_state(sys_state_);

	sys_state_.mark_synced();
}

void SimManager::write_dcd_frame(size_t step) {
	// Gather global state from patches
	gather_particle_data_from_patches();
	if (sys_state_.prepare_for_dcd_output()) {
		// 3. Get positions for DCD writing
		const auto& positions = sys_state_.get_global_positions();

		const auto& periodicity = sys_.get_boundary_conditions().get_periodicity();
		const bool with_unitcell = periodicity[0] || periodicity[1] || periodicity[2];

		if (!dcd_header_written_) {
			const int nsavc = std::max(1, static_cast<int>(sys_.get_output_period()));
			dcd_writer_->writeHeader(static_cast<int>(positions.size()),
									 1,
									 nsavc,
									 nsavc,
									 0,
									 sys_.get_timestep(),
									 with_unitcell);
			dcd_header_written_ = true;
		}

		// The header's with_unitcell flag promises an extra block on every
		// frame, so the cell must be supplied here or readers misparse the
		// trajectory. CHARMM on-disk order is [A, cos(g), B, cos(b), cos(a), C];
		// the box is orthorhombic, so all three cosines are 0.
		if (with_unitcell) {
			const Vector3 box = sys_.get_boundary_conditions().get_box_size();
			const std::vector<double> unitcell{box.x, 0.0, box.y, 0.0, 0.0, box.z};
			dcd_writer_->writeStep(positions, unitcell);
		} else {
			dcd_writer_->writeStep(positions);
		}
	}
}

void SimManager::write_momentum_dcd_frame(size_t step) {
	(void)step;
	// SystemState was already gathered and synced by write_dcd_frame() this
	// tick (handle_output only calls this right after it) - no need to
	// re-gather from patches.
	if (!sys_state_.is_state_synced()) {
		return;
	}
	const auto& momentum = sys_state_.get_global_momentum();

	const auto& periodicity = sys_.get_boundary_conditions().get_periodicity();
	const bool with_unitcell = periodicity[0] || periodicity[1] || periodicity[2];

	if (!momentum_dcd_header_written_) {
		const int nsavc = std::max(1, static_cast<int>(sys_.get_output_period()));
		momentum_dcd_writer_->writeHeader(static_cast<int>(momentum.size()),
										  1,
										  nsavc,
										  nsavc,
										  0,
										  sys_.get_timestep(),
										  with_unitcell);
		momentum_dcd_header_written_ = true;
	}

	if (with_unitcell) {
		const Vector3 box = sys_.get_boundary_conditions().get_box_size();
		const std::vector<double> unitcell{box.x, 0.0, box.y, 0.0, 0.0, box.z};
		momentum_dcd_writer_->writeStep(momentum, unitcell);
	} else {
		momentum_dcd_writer_->writeStep(momentum);
	}
}

//================================================================================
// Progress and Performance Reporting
//================================================================================

void SimManager::report_progress(size_t current_step,
								 size_t total_steps,
								 size_t report_period) {
	wkf_timer_stop(timerP_.timer);
	const float interval_elapsed = static_cast<float>(wkf_timer_time(timerP_.timer));
	wkf_timer_start(timerP_.timer);

	const float percent =
		(100.0f * static_cast<float>(current_step)) / static_cast<float>(total_steps);
	const float ms_per_step = interval_elapsed * 1000.0f / static_cast<float>(report_period);
	const int num_replicas = 1; // TODO: expose replicas from SimSystem config
	const float ns_per_day =
		static_cast<float>(num_replicas) * sys_.get_timestep() / ms_per_step * 86400000.0f;

	std::printf("\rStep %zu [%.2f%% complete | %.3f ms/step | %.3f ns/day]",
				current_step,
				percent,
				ms_per_step,
				ns_per_day);
	std::fflush(stdout);
}

void SimManager::report_performance(float elapsed_time, size_t total_steps) {
	const float steps_per_second = static_cast<float>(total_steps) / elapsed_time;
	const float io_time = wkf_timer_time(timerS_.timer);
	const float energy_time = wkf_timer_time(timerE_.timer);
	const float compute_time = elapsed_time - io_time - energy_time;

	LOGINFO("=========================================");
	LOGINFO("SimManager: Performance Summary");
	LOGINFO("=========================================");
	LOGINFO("  Total time:        {:.2f} s", elapsed_time);
	LOGINFO("  Compute time:      {:.2f} s ({:.1f}%)",
			compute_time,
			compute_time / elapsed_time * 100);
	LOGINFO("  I/O time:          {:.2f} s ({:.1f}%)", io_time, io_time / elapsed_time * 100);
	LOGINFO("  Energy time:       {:.2f} s ({:.1f}%)",
			energy_time,
			energy_time / elapsed_time * 100);
	LOGINFO("  Steps/second:      {:.2f}", steps_per_second);
	LOGINFO("  ns/day (est):      {:.2f}",
			(steps_per_second * sys_.get_timestep() * 86400.0f) / 1e6f);
	LOGINFO("=========================================");
}

//================================================================================
// IMD Handling
//================================================================================

void SimManager::handle_imd_commands() {
	// TODO: Implement IMD command handling when needed
	// Check for incoming IMD commands
	// Update forces/positions based on user interaction
}

//================================================================================
// Initial Conditions
//================================================================================

void SimManager::generate_initial_particles(std::vector<Vector3>& positions,
											std::vector<int>& types) {
	const Vector3 box_size = sys_.get_box_size();

	// TODO: Get number of particles from configuration
	const size_t num_particles = sys_state_.get_num_particles();

	positions.reserve(num_particles);
	types.reserve(num_particles);

	// Simple random placement for testing
	// TODO: Replace with proper initial condition generation
	for (size_t i = 0; i < num_particles; ++i) {
		positions.emplace_back(box_size.x * (float)rand() / float(RAND_MAX),
							   box_size.y * (float)rand() / float(RAND_MAX),
							   box_size.z * (float)rand() / float(RAND_MAX));
		types.push_back(0); // All type 0 for now
	}

	LOGINFO("SimManager: Generated {} particles", num_particles);
}

/**
 * @brief Generate initial particle momentum and types according to Boltzmann distribution
 * @todo Make sure this is correct
 * @param v_com
 * @note V1 Configuration::Boltzmann(const Vector3& v_com, int N)
 */
void SimManager::generate_initial_momentum(const Vector3& v_com) {
	const Temperature& temperature = sys_.get_temperature_struct();
	float kT = 1.0f;
	if (temperature.format == Temperature::Format::Grid) {
		throw Exception(ExceptionType::RuntimeError,
						SourceLocation(),
						"Grid temperature not supported for initial momentum generation");
	} else {
		kT = temperature.kT; // Fix: don't redeclare, just assign
	}

	const size_t num_particles = sys_state_.get_num_particles();
	const auto& particle_types = sys_.get_particle_types();
	std::vector<Vector3> momentum(num_particles);

	// Constants for unit conversion
	// SQRT_CAL_TO_JOULE = 2.046167337e4 (from Constants.h)

	// Initialize random number generator for host-side generation
	static std::random_device rd;
	static std::mt19937 gen(rd());
	std::normal_distribution<double> gaussian(0.0, 1.0);

	// Generate momenta from Maxwell-Boltzmann distribution
	// p = sqrt(kT * m) * random_gaussian
	Vector3 total_momentum(0.0, 0.0, 0.0);

	for (size_t i = 0; i < num_particles; ++i) {
		int typ = particle_types[i].id;
		double M = particle_types[typ].mass;
		double sigma = sqrt(kT * M) * constants::SQRT_CAL_TO_JOULE;

		// Generate 3D Gaussian random vector
		Vector3 tmp(gaussian(gen) * sigma, gaussian(gen) * sigma, gaussian(gen) * sigma);

		momentum[i] = tmp;
		total_momentum += tmp;
	}

	// Remove center of mass momentum to ensure zero net momentum
	if (num_particles > 1) {
		Vector3 p_com = total_momentum / static_cast<double>(num_particles);
		for (size_t i = 0; i < num_particles; ++i) {
			int typ = particle_types[i].id;
			double M = particle_types[typ].mass;
			momentum[i] = momentum[i] - p_com + M * v_com;
		}
	}

	LOGINFO("SimManager: Generated initial momenta for {} particles at kT={}", num_particles, kT);
}
void SimManager::load_restart_data(const std::string& filename) {
	// TODO: Implement restart file loading
	LOGINFO("SimManager: Restart file loading not yet implemented");
	throw Exception(ExceptionType::NotImplementedError,
					SourceLocation(),
					"Restart file loading not yet implemented");
}

//================================================================================
// Energy Output
//================================================================================

void SimManager::write_energy_output(size_t step) {
	(void)step;
	// Gather the SystemState that execute_force_calculation() this step
	// accumulated potential energy into (ForceEnergy.t, unpacked into
	// HostParticleData::energy by DeviceParticle::copy_to_host).
	gather_particle_data_from_patches();

	const auto& particles = sys_state_.get_global_particles();
	const size_t n = particles.size();

	// Potential energy: sum of per-particle energy from bonded + pairwise
	// nonbonded kernels (only accumulated when execute_force_calculation()
	// enabled compute_energy for this step - see the energy_output_period
	// gating there). Grid/PMF contributions are not tracked yet.
	double potential_energy = 0.0;
	for (float e : particles.energy) {
		potential_energy += e;
	}

	// Kinetic energy: momenta are generated/integrated in units scaled by
	// SQRT_CAL_TO_JOULE (see generate_initial_momentum), so 0.5*p^2/m must be
	// divided by SQRT_CAL_TO_JOULE^2 to recover kcal/mol before expressing it
	// as a multiple of kT - matching legacy ARBD's energy.dat convention.
	const auto& particle_types = sys_.get_particle_types();
	const Temperature& temperature = sys_.get_temperature_struct();
	const float kT = temperature.kT;
	double kinetic_energy_kcal = 0.0;
	for (size_t i = 0; i < n; ++i) {
		const int typ = particles.type_id[i];
		const double mass = particle_types[typ].mass;
		kinetic_energy_kcal += 0.5 * particles.mom[i].length2() / mass;
	}
	kinetic_energy_kcal /= (constants::SQRT_CAL_TO_JOULE * constants::SQRT_CAL_TO_JOULE);
	const double kinetic_energy_kT = kT > 0.0f ? kinetic_energy_kcal / kT : 0.0;

	const std::string energy_filename = sys_.get_output_name() + ".energy.dat";
	std::ofstream energy_file(energy_filename, std::ios::out | std::ios::app);
	if (energy_file.is_open()) {
		energy_file << "Kinetic Energy: " << kinetic_energy_kT << " (kT) " << std::endl;
		energy_file << "Potential Energy: " << potential_energy << " (kcal/mol) " << std::endl;
	} else {
		LOGWARN("SimManager: Failed to open '{}' for energy output", energy_filename);
	}

	if (has_rigid_bodies_) {
		// TODO: rigid-body kinetic/potential energy is not yet computed by
		// SimManager (no rigid-body dynamics pipeline exists yet); write
		// zeros so downstream tooling that expects this file still gets it,
		// matching legacy ARBD's rb_energy.dat format.
		const std::string rb_energy_filename = sys_.get_output_name() + ".rb_energy.dat";
		std::ofstream rb_energy_file(rb_energy_filename, std::ios::out | std::ios::app);
		if (rb_energy_file.is_open()) {
			rb_energy_file << "Kinetic Energy 0 (kT)" << std::endl;
			rb_energy_file << "Potential Energy 0 (kcal/mol)" << std::endl;
		} else {
			LOGWARN("SimManager: Failed to open '{}' for rigid body energy output",
					rb_energy_filename);
		}
	}

	write_restart_files();
}

//================================================================================
// Restart Files
//================================================================================

void SimManager::write_restart_files() {
	// Assumes sys_state_ already holds the state to persist (gathered by the
	// caller - write_energy_output() during the run, write_final_restart()
	// at the end).
	const auto& particles = sys_state_.get_global_particles();

	const std::string restart_filename = sys_.get_output_name() + ".restart";
	FILE* out = std::fopen(restart_filename.c_str(), "w");
	if (out) {
		for (size_t i = 0; i < particles.size(); ++i) {
			const Vector3& p = particles.pos[i];
			std::fprintf(out, "%d %.10g %.10g %.10g\n", particles.type_id[i], p.x, p.y, p.z);
		}
		std::fclose(out);
	} else {
		LOGWARN("SimManager: Failed to open '{}' for restart output", restart_filename);
	}

	// Momentum restart only applies to Langevin dynamics, matching the
	// momentum trajectory gating in initialize_output_writers(). The "0" in
	// the filename mirrors legacy ARBD's on-disk naming for this file.
	if (has_momentum_output_) {
		const std::string momentum_restart_filename = sys_.get_output_name() + ".0.momentum.restart";
		FILE* mout = std::fopen(momentum_restart_filename.c_str(), "w");
		if (mout) {
			for (size_t i = 0; i < particles.size(); ++i) {
				const Vector3& p = particles.mom[i];
				std::fprintf(mout, "%d %.10g %.10g %.10g\n", particles.type_id[i], p.x, p.y, p.z);
			}
			std::fclose(mout);
		} else {
			LOGWARN("SimManager: Failed to open '{}' for momentum restart output",
					momentum_restart_filename);
		}
	}
}

void SimManager::write_final_restart() {
	// Gather particle data before writing restart
	gather_particle_data_from_patches();
	write_restart_files();
	LOGINFO("SimManager: Wrote final restart files for '{}'", sys_.get_output_name());
}

//================================================================================
// Particle Reactions
//================================================================================
/*
void SimManager::perform_reactions() {
	Patch& patch = sys_.get_patch_manager()->get_local_patch();

	// 1. Run Reaction Kernel
	//    Sets FLAG_DEAD on some particles.
	//    Creates new particles in a temporary "Birth Buffer".

	// 2. Remove Dead Particles
	//    Compacts DeviceParticle array.
	//    Generates 'permutation_map'.
	auto perm_map = patch.compact_particles();

	// 3. Update Interactions (Fix Indices)
	if (patch.has_topology_changes()) {
		for (auto& interaction : interactions_) {
			interaction->update_topology(perm_map);
		}
	}

	// 4. Add New Particles
	//    Appends from "Birth Buffer" to end of DeviceParticle.
	//    (No index shifting for existing particles, so safe).
	patch.append_from_buffer(birth_buffer);

	// 5. Rebuild Neighbor Lists
	//    Mandatory after moving particles.
	neighbor_list_.force_rebuild();
}
*/
} // namespace ARBD
