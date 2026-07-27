#include "SimManager.h"
#include "System/PatchManager.h"
#include <random>

namespace ARBD {

//================================================================================
// Constructor
//================================================================================

SimManager::SimManager(SimSystem& sys) : sys_(sys), sys_state_(sys) {
	timer0_.timer = wkf_timer_create();
	timerS_.timer = wkf_timer_create();
	timerE_.timer = wkf_timer_create();
}

SimManager::SimManager(SimSystem& sys, const ConfigParser& parser) : sys_(sys), sys_state_(sys) {
	load_config(parser);
	timer0_.timer = wkf_timer_create();
	timerS_.timer = wkf_timer_create();
	timerE_.timer = wkf_timer_create();
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
	wkf_timer_start(&timer0_);
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

	for (size_t step = 1; step <= num_steps; ++step) {
		LOGINFO("DEBUG step {} dcd_writer_={}", step, static_cast<void*>(dcd_writer_.get()));
		// ===== FORCE CALCULATION PHASE =====
		execute_force_calculation(step);
		LOGINFO("DEBUG step {} after force dcd_writer_={}", step, static_cast<void*>(dcd_writer_.get()));

		// ===== INTEGRATION PHASE =====
		execute_integration(step);
		LOGINFO("DEBUG step {} after integrate dcd_writer_={}", step, static_cast<void*>(dcd_writer_.get()));

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
		if (step % 1000 == 0) {
			report_progress(step, num_steps);
		}
	}

	// ===== FINALIZATION =====
	wkf_timer_stop(&timer0_);
	const float elapsed = wkf_timer_time(&timer0_);

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

	for (auto& patch : patch_mgr->get_patches()) {
		DeviceParticleTypes particle_types(sys_.get_particle_types(), patch->get_resource());
		particle_types.copy_from_host(sys_.get_particle_types());

		size_t resource_idx = 0;
		for (size_t i = 0; i < resources.size(); ++i) {
			if (resources[i] == patch->get_resource()) {
				resource_idx = i;
				break;
			}
		}

		Event evt = patch->calculate_nonbonded_forces(sys_.get_nonbonded_interactions(),
													  particle_types,
													  grid_manager.get_device_grid_views(resource_idx));
		(void)evt;
	}

	if (step == 1) {
		LOGINFO("SimManager: PMF/grid force kernel launched (pairwise forces still TODO)");
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

	// Energy calculation and output
	if (energy_output_period > 0 && step % energy_output_period == 0) {
		wkf_timer_start(&timerE_);
		// TODO: Implement energy calculation
		// calculate_energy(step);
		wkf_timer_stop(&timerE_);
	}

	// Trajectory output
	if (output_period > 0 && step % output_period == 0) {
		wkf_timer_start(&timerS_);

		if (dcd_writer_) {
			write_dcd_frame(step);
		} else if (traj_writer_) {
			// Write with generic trajectory writer
			// traj_writer_->write_frame(step);
		}

		wkf_timer_stop(&timerS_);
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
		dcd_writer_->writeStep(positions);
	}
}

//================================================================================
// Progress and Performance Reporting
//================================================================================

void SimManager::report_progress(size_t current_step, size_t total_steps) {
	const float progress =
		static_cast<float>(current_step) / static_cast<float>(total_steps) * 100.0f;
	const float elapsed = wkf_timer_time(&timer0_);
	const float steps_per_sec = static_cast<float>(current_step) / elapsed;

	LOGINFO("SimManager: Step {}/{} ({:.1f}%) | Elapsed: {:.2f}s | Rate: {:.1f} steps/s",
			current_step,
			total_steps,
			progress,
			elapsed,
			steps_per_sec);
}

void SimManager::report_performance(float elapsed_time, size_t total_steps) {
	const float steps_per_second = static_cast<float>(total_steps) / elapsed_time;
	const float io_time = wkf_timer_time(&timerS_);
	const float energy_time = wkf_timer_time(&timerE_);
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
// Final Restart File
//================================================================================

void SimManager::write_final_restart() {
	// Gather particle data before writing restart
	gather_particle_data_from_patches();

	const std::string restart_filename = "out_final.restart"; // TODO: Get from SimSystem

	// Get current particle positions
	const auto& positions = sys_state_.get_global_positions();

	LOGINFO("SimManager: Writing final restart to '{}'", restart_filename);
	LOGINFO("SimManager: {} particles at final state", positions.size());

	// TODO: Implement actual restart file writing
	// write_restart_file(restart_filename, positions, velocities, types);
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
