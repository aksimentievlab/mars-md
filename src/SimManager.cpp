#include "SimManager.h"
#include "System/PatchManager.h"
#include <charconv>
#include <cstdio>
#include <fstream>
#include <random>
#include <unordered_map>

namespace ARBD {

namespace {
void append_restart_line(std::string& buf, int type_id, const Vector3& v) {
	char tmp[64];
	auto append = [&](auto value) {
		auto res = std::to_chars(tmp, tmp + sizeof(tmp), value);
		buf.append(tmp, res.ptr);
	};
	append(type_id);
	buf.push_back(' ');
	append(v.x);
	buf.push_back(' ');
	append(v.y);
	buf.push_back(' ');
	append(v.z);
	buf.push_back('\n');
}

/// Segname for a rigid body's cosmetic atoms: the type name up to its first
/// '.', capped at the PDB segID column width. See dev_notes.md.
std::string rigid_body_segname(const std::string& type_name) {
	std::string s = type_name.substr(0, type_name.find('.'));
	if (s.size() > 4) {
		s.resize(4);
	}
	return s.empty() ? std::string("RB") : s;
}
} // namespace

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
	sys_.set_rb_update_period(parser.get_sim_system().get_rb_update_period());
	sys_.set_base_seed(parser.get_sim_system().get_base_seed());

	// Cache particles here; they are converted (type_name -> type_id) in init(),
	// once particle type IDs have been assigned - SystemState::set_init_particle_data()
	// would otherwise look up type names in an empty/unbuilt map.
	pending_initial_particles_ = parser.get_init_particles();
	// Unlike particles, ConfigParser already resolves RigidBodyIO::type_id at
	// parse time (see ConfigParser.cpp's rigidBody block), so this can be
	// cached as-is with no further conversion needed in init().
	pending_initial_rigid_bodies_ = parser.get_init_rigid_bodies();
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
	sys_.assign_rigid_body_type_ids();
	LOGINFO("SimManager: Rigid body type IDs assigned");
	sys_.build_name_to_id_maps();
	LOGINFO("SimManager: Name to ID maps built");

	// Load cached initial particle data (from load_config or set_initial_particles)
	if (!pending_initial_particles_.empty()) {
		// The staged list order *is* the global particle indexing, so assign
		// ids from it (ConfigParser already does this for its own path) and,
		// while the mapping is known, rewrite any bonded terms that named
		// their particles by ParticleIO::uid handle rather than by index -
		// see ParticleUids in Interactions/BondedInteraction.h.
		std::unordered_map<int, int> uid_to_index;
		for (size_t i = 0; i < pending_initial_particles_.size(); ++i) {
			auto& particle = pending_initial_particles_[i];
			particle.id = static_cast<int>(i);
			if (particle.uid >= 0) {
				uid_to_index[particle.uid] = static_cast<int>(i);
			}
		}
		if (!uid_to_index.empty()) {
			pending_bonded_interactions_.resolve_particle_uids([&](int uid) {
				auto it = uid_to_index.find(uid);
				if (it == uid_to_index.end()) {
					throw_value_error("SimManager: bonded interaction references a particle that "
									  "was not staged (uid %d)",
									  uid);
				}
				return it->second;
			});
		}

		sys_state_.set_init_particle_data(pending_initial_particles_);
		LOGINFO("SimManager: Loaded {} initial particles into system state",
				pending_initial_particles_.size());
	}

	// Load cached initial rigid-body data (from load_config or set_initial_rigid_bodies)
	if (!pending_initial_rigid_bodies_.empty()) {
		sys_state_.set_init_rigid_body_data(pending_initial_rigid_bodies_);
		LOGINFO("SimManager: Loaded {} initial rigid bodies into system state",
				pending_initial_rigid_bodies_.size());
	}

	// Load cached bonded interactions (from load_config or set_bonded_interactions).
	// Every term that rides in the struct has to be in the guard, or a config
	// carrying only that term loses it silently - see dev_notes.md.
	if (pending_bonded_interactions_.get_num_bonds() > 0 ||
		pending_bonded_interactions_.get_num_angles() > 0 ||
		pending_bonded_interactions_.get_num_dihedrals() > 0 ||
		!pending_bonded_interactions_.get_exclusions().empty() ||
		!pending_bonded_interactions_.get_restraints().empty()) {
		sys_state_.update_bonded_interactions(pending_bonded_interactions_);
		LOGINFO("SimManager: Loaded {} bonds, {} angles, {} dihedrals, {} exclusions, {} "
				"restraints into system state",
				pending_bonded_interactions_.get_num_bonds(),
				pending_bonded_interactions_.get_num_angles(),
				pending_bonded_interactions_.get_num_dihedrals(),
				pending_bonded_interactions_.get_exclusions().size(),
				pending_bonded_interactions_.get_restraints().size());
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

	// Rigid-body manager construction needs grids already on-device
	// (prepare_grid_grid_dispatch/prepare_particle_grid_dispatch read grid
	// sizes/views), so this must run after build_device_arrays() above.
	if (!sys_.get_rigid_body_types().empty()) {
		rigid_body_manager_ = std::make_unique<RigidBodyManager>(sys_.get_resources());
		rigid_body_manager_->initialize(
			sys_.get_rigid_body_types(),
			sys_state_.get_global_rigid_bodies(),
			[this](int id) { return sys_.get_grid_manager().get_grid_format(id); },
			sys_.get_rb_update_period());
		// AoS -> SoA here, at the IO boundary: RigidBodyIO is a parse-time
		// structure and must not reach the device-side managers. Mirrors what
		// SystemState does for the fields HostRigidBodyData does carry.
		{
			const size_t rb_count = pending_initial_rigid_bodies_.size();
			std::vector<Vector3> constant_force(rb_count);
			std::vector<Vector3> constant_torque(rb_count);
			for (size_t i = 0; i < rb_count; ++i) {
				constant_force[i] = pending_initial_rigid_bodies_[i].external_force;
				constant_torque[i] = pending_initial_rigid_bodies_[i].external_torque;
			}
			rigid_body_manager_->set_external_loads(constant_force, constant_torque);
		}
		rigid_body_manager_->prepare_grid_grid_dispatch(sys_.get_grid_manager(), 0);
		rigid_body_manager_->prepare_particle_grid_dispatch(sys_.get_rigid_body_types(),
															sys_state_.get_num_particles());
		// Uses the AoS instance list rather than SystemState's SoA copy: the
		// attached_start/attached_count ranges ConfigParser assigned live on
		// RigidBodyIO and have no HostRigidBodyData counterpart. Index and
		// RigidBodyIO::id agree (ConfigParser numbers instances sequentially),
		// which is what lets rb_id double as the device-array index.
		rigid_body_manager_->prepare_attached_particles(
			sys_.get_rigid_body_types(),
			pending_initial_rigid_bodies_,
			sys_.get_patch_manager() ? sys_.get_patch_manager()->get_patches().size() : 1);
		rigid_body_manager_->prepare_cosmetic_atoms(sys_.get_rigid_body_types(),
													pending_initial_rigid_bodies_);
		LOGINFO("SimManager: Rigid body manager initialized with {} bodies",
				sys_state_.get_num_rigid_bodies());

		// Langevin/DLM and Brownian are implemented; VelocityVerlet is not, so
		// warn rather than silently running something else.
		const IntegratorType rb_algorithm = sys_.get_rigid_body_algorithm();
		if (rb_algorithm == IntegratorType::VelocityVerlet) {
			LOGWARN("SimManager: VelocityVerlet rigid-body dynamics is not implemented; "
					"falling back to Langevin/DLM");
		}
		for (const RigidBodyType& rb_type : sys_.get_rigid_body_types()) {
			rb_type.check_damping(rb_algorithm);
		}
	}

	// Structure files describing the trajectory this run is about to write.
	// Emitted here, before any frame exists, so the DCD always has a matching
	// PSF/PDB on disk even if the run is interrupted. Non-fatal: a failure to
	// write a visualization file must not abort an otherwise valid simulation.
	try {
		write_psf();
		write_pdb();
	} catch (const std::exception& e) {
		LOGWARN("SimManager: could not write structure files ({}) - the simulation will still run, "
				"but the trajectory will have no matching PSF/PDB",
				e.what());
	}

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

	const size_t progress_period = energy_output_period > 0 ? energy_output_period : 1000;
	const int num_replicas = 1; // TODO: expose replicas from SimSystem config

	std::printf("Configuration: %zu particles | %d replicas\n",
				sys_state_.get_num_particles(),
				num_replicas);
	std::fflush(stdout);

	wkf_timer_start(timer0_.timer);
	wkf_timer_start(timerP_.timer);

	// DLM splits its two half-kicks around the force phase. See dev_notes.md.
	const bool split_dlm = rigid_body_manager_ && rigid_body_manager_->size() > 0 &&
						   sys_.get_rigid_body_algorithm() != IntegratorType::Brownian;
	const PeriodicBox& sim_box = sys_.get_boundary_conditions();
	const float dt = sys_.get_timestep();

	if (split_dlm) {
		execute_force_calculation(0); // prime step 1's opening half-kick
	}

	for (size_t step = 1; step <= num_steps; ++step) {
		// ===== RB DRIFT PHASE (DLM substeps 0-1) =====
		if (split_dlm) {
			rigid_body_manager_->integrate_drift(dt, sim_box).wait();
			sys_state_.invalidate_rigid_bodies();
		}

		// ===== FORCE CALCULATION PHASE =====
		execute_force_calculation(step);

		// ===== RB KICK PHASE (DLM substep 2) =====
		if (split_dlm) {
			rigid_body_manager_->integrate_kick(dt, sim_box).wait();
			sys_state_.invalidate_rigid_bodies();
		}

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
	// The final restart carries momenta, and the last step's closing half-kick
	// is still outstanding - without this the file would resume the run from a
	// half-step momentum treated as a whole-step one.
	settle_momenta_for_output(num_steps);
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

	// Legacy step order: clear RB forces -> particle nonbonded/bonded -> RB
	// grid-grid/particle-RB -> RB Langevin (see the RB force block below).
	if (rigid_body_manager_) {
		rigid_body_manager_->bodies().clear_forces();

		// Attached particles are slaved to their bodies, so their positions
		// must be refreshed from the state the previous step's RB integration
		// left behind - before the pairlist and force kernels below read them.
		if (rigid_body_manager_->has_attached_particles()) {
			auto& patches = patch_mgr->get_patches();
			if (!patches.empty()) {
				rigid_body_manager_
					->sync_attached_particle_positions(patches.front()->get_particles().view())
					.wait();
			}
		}
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

		// Particle types are static for the whole run
		if (!device_particle_types_cache_[resource_idx]) {
			device_particle_types_cache_[resource_idx] =
				std::make_unique<DeviceParticleTypes>(sys_.get_particle_types(),
													  patch->get_resource());
		}
		DeviceParticleTypes& particle_types = *device_particle_types_cache_[resource_idx];

		// Nonbonded clears and writes; bonded must follow and accumulate.
		// Energy is gated to output steps - it costs an extra atomic per term.
		// See dev_notes.md.
		const size_t energy_output_period = static_cast<size_t>(sys_.get_energy_output_period());
		const bool compute_energy = energy_output_period > 0 && step % energy_output_period == 0;
		Event evt = patch->calculate_nonbonded_forces(
			sys_.get_nonbonded_interactions(),
			sys_state_.get_bonded_interactions(),
			particle_types,
			grid_manager.get_device_grid_views(resource_idx),
			sys_.get_tables_registry(),
			resource_idx,
			static_cast<float>(sys_.get_pairlist_cutoff()),
			static_cast<float>(sys_.get_cutoff()),
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

	// RB grid-grid, then particle-RB (single-patch assumption, matching
	// Patch::calculate_bonded_forces's own documented limitation), then RB
	// Langevin - legacy step order.
	if (rigid_body_manager_) {
		// Attached particles have now accumulated their full nonbonded+bonded
		// force. That force never moves them (the integrators skip them);
		// it acts on the parent body instead, so fold it into the body's net
		// force/torque before the Langevin term and the integration below.
		if (rigid_body_manager_->has_attached_particles()) {
			auto& patches = patch_mgr->get_patches();
			if (!patches.empty()) {
				rigid_body_manager_
					->reduce_attached_particle_forces(
						std::as_const(patches.front()->get_particles()).view())
					.wait();
			}
		}

		Event grid_evt =
			rigid_body_manager_->compute_grid_grid_forces(grid_manager,
														  0,
														  step,
														  static_cast<float>(sys_.get_cutoff()));

		auto& patches = patch_mgr->get_patches();
		if (!patches.empty()) {
			Event particle_rb_evt = rigid_body_manager_->compute_particle_rb_forces(
				grid_manager,
				0,
				patches.front()->get_particles().view());
			particle_rb_evt.wait();
		}
		grid_evt.wait();

		if (sys_.get_rigid_body_algorithm() != IntegratorType::Brownian) {
			const Temperature& temperature = sys_.get_temperature_struct();
			rigid_body_manager_
				->add_langevin_forces(sys_.get_timestep(),
									  temperature.kT,
									  sys_.get_base_seed(),
									  step)
				.wait();
		}
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

	if (rigid_body_manager_) {
		if (sys_.get_rigid_body_algorithm() == IntegratorType::Brownian) {
			const Temperature& temperature = sys_.get_temperature_struct();
			rigid_body_manager_
				->integrate_brownian(timestep,
									 temperature.kT,
									 sys_.get_base_seed(),
									 step,
									 sys_.get_boundary_conditions())
				.wait();
		}
		// DLM is integrated around the force phase in run(), not here.
		sys_state_.invalidate_rigid_bodies();
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

void SimManager::settle_momenta_for_output(size_t step) {
	// BAOAB's closing half-kick is deferred, so momentum is half a kick behind
	// position. Pay it before reading, which needs a fresh force. See dev_notes.md.
	PatchManager* patch_mgr = sys_.get_patch_manager();
	if (!patch_mgr) {
		return;
	}

	bool any_pending = false;
	for (auto& patch : patch_mgr->get_patches()) {
		any_pending = any_pending || patch->has_deferred_kick();
	}
	if (!any_pending) {
		return;
	}

	execute_force_calculation(step);
	const float timestep = sys_.get_timestep();
	for (auto& patch : patch_mgr->get_patches()) {
		patch->finish_deferred_kick(timestep).wait();
	}
}

void SimManager::handle_output(size_t step) {
	const size_t output_period = static_cast<size_t>(sys_.get_output_period());
	const size_t energy_output_period = static_cast<size_t>(sys_.get_energy_output_period());

	const bool trajectory_due = output_period > 0 && step % output_period == 0;
	const bool energy_due = energy_output_period > 0 && step % energy_output_period == 0;

	// Only needed when something about to be written reads momentum: the
	// momentum trajectory, or the energy report and the restart files it
	// refreshes. A positions-only DCD frame does not.
	if ((trajectory_due && momentum_dcd_writer_) || energy_due) {
		settle_momenta_for_output(step);
	}

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

		// Plaintext rigid-body trajectory, for direct comparison against v1's
		// "<name>.0.rb-traj". Independent of the DCD writers above.
		write_rb_traj_frame(step);

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

void SimManager::gather_particle_data_from_patches(bool need_energy) {
	PatchManager* patch_mgr = sys_.get_patch_manager();
	sys_state_.clear_global_arrays();
	if (!patch_mgr) {
		throw Exception(ExceptionType::RuntimeError,
						SourceLocation(),
						"PatchManager not available for gathering particle data");
	}
	patch_mgr->gather_particles_to_state(sys_state_, need_energy);

	sys_state_.mark_synced();
}

void SimManager::gather_rigid_body_data() {
	if (!rigid_body_manager_ || rigid_body_manager_->size() == 0) {
		return;
	}
	rigid_body_manager_->gather_to_host(sys_state_.mutable_rigid_bodies());
	sys_state_.mark_rigid_bodies_synced();
}

void SimManager::write_dcd_frame(size_t step) {
	// Gather global state from patches
	gather_particle_data_from_patches();
	if (sys_state_.prepare_for_dcd_output()) {
		// Frame layout is [regular particles][attached particles][cosmetic
		// atoms]. The first two blocks are exactly the global particle array -
		// ConfigParser appends attached particles after all regular ones - so
		// this is "every real particle, then the visualization-only remainder".
		// A PSF describing this trajectory must use the same order.
		const auto& positions = sys_state_.get_global_positions();
		std::vector<Vector3> frame(positions.begin(), positions.end());
		if (rigid_body_manager_ && rigid_body_manager_->num_cosmetic_atoms() > 0) {
			// Placed on the device from the bodies' current transforms; this is
			// the only work in the step that touches these atoms.
			rigid_body_manager_->compute_cosmetic_positions().wait();
			rigid_body_manager_->copy_cosmetic_positions_to_host(frame);
		}

		const auto& periodicity = sys_.get_boundary_conditions().get_periodicity();
		const bool with_unitcell = periodicity[0] || periodicity[1] || periodicity[2];

		if (!dcd_header_written_) {
			const int nsavc = std::max(1, static_cast<int>(sys_.get_output_period()));
			dcd_writer_->writeHeader(static_cast<int>(frame.size()),
									 1,
									 nsavc,
									 nsavc,
									 0,
									 sys_.get_timestep(),
									 with_unitcell);
			dcd_header_written_ = true;
			LOGINFO("SimManager: DCD has {} atom(s) = {} particle(s) + {} cosmetic",
					frame.size(),
					positions.size(),
					frame.size() - positions.size());
		}

		// The header's with_unitcell flag promises an extra block on every
		// frame, so the cell must be supplied here or readers misparse the
		// trajectory. CHARMM on-disk order is [A, cos(g), B, cos(b), cos(a), C];
		// the box is orthorhombic, so all three cosines are 0.
		if (with_unitcell) {
			const Vector3 box = sys_.get_boundary_conditions().get_box_size();
			const std::vector<double> unitcell{box.x, 0.0, box.y, 0.0, 0.0, box.z};
			dcd_writer_->writeStep(frame, unitcell);
		} else {
			dcd_writer_->writeStep(frame);
		}
	}
}

void SimManager::build_structure_view() {
	if (structure_view_) {
		return; // atom set is fixed for the run
	}
	structure_view_ = std::make_unique<PsfPdbStructure>();
	PsfPdbStructure& s = *structure_view_;

	const Vector3 box = sys_.get_boundary_conditions().get_box_size();
	s.box_dimensions = box;
	s.has_cryst1 = true;

	// --- Block 1+2: the global particle array (regular, then attached) -------
	const HostParticleData& particles = sys_state_.get_global_particles();
	const auto& ptypes = sys_.get_particle_types();
	const size_t num_particles = particles.size();

	s.atoms.reserve(num_particles);
	for (size_t i = 0; i < num_particles; ++i) {
		const int tid = particles.type_id[i];
		const bool known = tid >= 0 && static_cast<size_t>(tid) < ptypes.size();

		PdbAtomRecord a;
		a.serial = static_cast<int>(i) + 1;
		a.name = known ? ptypes[tid].name : "X";
		a.resname = a.name;
		a.type_name = a.name;
		a.mass = known ? ptypes[tid].mass : 1.0f;
		a.charge = known ? ptypes[tid].charge : 0.0f;
		a.resid = static_cast<int>(i) + 1;
		a.chain = "A";
		// An attached particle is tagged so it can be selected in VMD and told
		// apart from free particles of the same type.
		const bool is_attached =
			i < particles.attached_rigid_body_id.size() && particles.attached_rigid_body_id[i] >= 0;
		a.segname = is_attached ? "ATT" : "SYS";
		if (is_attached) {
			a.beta = static_cast<float>(particles.attached_rigid_body_id[i]);
		}
		a.position = i < particles.pos.size() ? particles.pos[i] : Vector3(0.0f);
		s.atoms.push_back(std::move(a));
	}

	// Real bonded topology: indices are already into the global particle array.
	for (const Bond& b : sys_state_.get_bonded_interactions().get_bonds()) {
		s.bonds.emplace_back(b.ind1, b.ind2);
	}

	// --- Block 3: cosmetic template atoms -----------------------------------
	// Also build, per instance, template-atom index -> global index, so the
	// template's own bonds can be emitted in this file's numbering. Attached
	// atoms map back into block 2; cosmetic ones into the tail being appended.
	const auto& rtypes = sys_.get_rigid_body_types();
	for (const RigidBodyIO& rb : pending_initial_rigid_bodies_) {
		if (rb.type_id < 0 || static_cast<size_t>(rb.type_id) >= rtypes.size()) {
			continue;
		}
		const RigidBodyType& type = rtypes[rb.type_id];
		std::vector<int> template_to_global(type.template_particles.size(), -1);

		for (size_t t = 0; t < type.template_particles.size(); ++t) {
			const CosmeticParticle& c = type.template_particles[t];
			if (c.attached_particle_index >= 0) {
				template_to_global[t] = rb.attached_start + c.attached_particle_index;
				continue;
			}
			template_to_global[t] = static_cast<int>(s.atoms.size());

			PdbAtomRecord a;
			a.serial = static_cast<int>(s.atoms.size()) + 1;
			a.name = c.name.empty() ? c.resname : c.name;
			a.resname = c.resname;
			a.type_name = c.type_name.empty() ? c.resname : c.type_name;
			a.resid = c.resid;
			a.chain = "R";
			// Type in segname, instance in beta - see dev_notes.md.
			a.segname = rigid_body_segname(type.name);
			a.beta = static_cast<float>(rb.id);
			a.mass = 0.0f;	 // no physics
			a.charge = 0.0f; // no physics
			// Placed at t=0 by this instance's transform; write_pdb refreshes it.
			a.position = rb.orientation * c.body_frame_position + rb.position;
			s.atoms.push_back(std::move(a));
		}

		for (const int2& b : type.template_bonds) {
			if (b.x < 0 || b.y < 0 || static_cast<size_t>(b.x) >= template_to_global.size() ||
				static_cast<size_t>(b.y) >= template_to_global.size()) {
				continue;
			}
			const int g1 = template_to_global[b.x];
			const int g2 = template_to_global[b.y];
			if (g1 >= 0 && g2 >= 0) {
				s.bonds.emplace_back(g1, g2);
			}
		}
	}

	// Nothing at all to describe. Throwing rather than warning-and-returning:
	// these writers are called explicitly (including from Python), so a silent
	// no-op just looks like a broken writer, and leaving structure_view_ set but
	// empty would cache that emptiness for the rest of the run.
	if (s.atoms.empty()) {
		structure_view_.reset();
		throw Exception(ExceptionType::RuntimeError,
						SourceLocation(),
						"SimManager: no particles or rigid-body template atoms to write - was "
						"init() called, and does the system actually contain anything?");
	}
}

void SimManager::refresh_structure_positions() {
	if (!structure_view_) {
		return;
	}
	PsfPdbStructure& s = *structure_view_;

	gather_particle_data_from_patches();
	std::vector<Vector3> frame;
	if (sys_state_.prepare_for_dcd_output()) {
		const auto& positions = sys_state_.get_global_positions();
		frame.assign(positions.begin(), positions.end());
	}
	if (rigid_body_manager_ && rigid_body_manager_->num_cosmetic_atoms() > 0) {
		rigid_body_manager_->compute_cosmetic_positions().wait();
		rigid_body_manager_->copy_cosmetic_positions_to_host(frame);
	}

	const size_t n = std::min(frame.size(), s.atoms.size());
	for (size_t i = 0; i < n; ++i) {
		s.atoms[i].position = frame[i];
	}
	if (frame.size() != s.atoms.size()) {
		LOGWARN("SimManager: structure has {} atom(s) but the current frame has {} - "
				"only the overlap was refreshed",
				s.atoms.size(),
				frame.size());
	}
}

void SimManager::write_psf(const std::string& path) {
	build_structure_view();
	const std::string out = path.empty() ? sys_.get_output_name() + ".psf" : path;
	structure_view_->write_psf(out);
	LOGINFO("SimManager: wrote PSF '{}' ({} atoms, {} bonds)",
			out,
			structure_view_->atoms.size(),
			structure_view_->bonds.size());
}

void SimManager::write_pdb(const std::string& path) {
	build_structure_view();
	refresh_structure_positions();
	const std::string out = path.empty() ? sys_.get_output_name() + ".pdb" : path;
	structure_view_->write_pdb(out, sys_.get_boundary_conditions().get_box_size());
	LOGINFO("SimManager: wrote PDB '{}' ({} atoms)", out, structure_view_->atoms.size());
}

void SimManager::write_rb_traj_frame(size_t step) {
	if (!rigid_body_manager_ || rigid_body_manager_->size() == 0) {
		return;
	}

	const idx_t count = rigid_body_manager_->size();
	gather_rigid_body_data();
	const HostRigidBodyData& rb = sys_state_.get_global_rigid_bodies();

	if (!rb_traj_file_.is_open()) {
		// "<name>.0.rb-traj" - the "0" is v1's replica-index artifact, kept for
		// output compatibility (same reasoning as the momentum DCD's name).
		const std::string path = sys_.get_output_name() + ".0.rb-traj";
		rb_traj_file_.open(path);
		if (!rb_traj_file_) {
			throw Exception(ExceptionType::FileIoError,
							SourceLocation(),
							"SimManager: could not open rigid-body trajectory file '%s'",
							path.c_str());
		}
		rb_traj_file_ << "# RigidBody trajectory file\n";
		rb_traj_file_ << "#$LABELS step RigidBodyKey"
						 " posX  posY  posZ"
						 " rotXX rotXY rotXZ"
						 " rotYX rotYY rotYZ"
						 " rotZX rotZY rotZZ"
						 " velX  velY  velZ"
						 " angVelX angVelY angVelZ\n";
		LOGINFO("SimManager: writing rigid-body trajectory to '{}'", path);
	}

	// v1 keys a body as "<typeName>#<index within that type>", numbering each
	// type's instances from 0 independently.
	const auto& types = sys_.get_rigid_body_types();
	std::unordered_map<int, int> seen_per_type;

	for (idx_t i = 0; i < count; ++i) {
		const int type_id = rb.type_id[i];
		const int instance = seen_per_type[type_id]++;
		const std::string type_name = (type_id >= 0 && static_cast<size_t>(type_id) < types.size())
										  ? types[type_id].name
										  : std::string("RB");

		const Vector3& p = rb.position[i];
		const Matrix3& o = rb.orientation[i];
		const Vector3& mom = rb.momentum[i];
		const Vector3& ang = rb.angular_momentum[i];

		// Position at the stream's default precision, everything after it at 10
		// significant digits - exactly v1's printData() formatting.
		rb_traj_file_ << std::setprecision(6) << step << " " << type_name << "#" << instance << " "
					  << p.x << " " << p.y << " " << p.z;
		// v1's legend is row-major (rotXX rotXY rotXZ | rotYX ...), while Matrix3
		// stores columns (ex/ey/ez) - so row r of the output reads component r
		// across all three column vectors, not one column vector.
		rb_traj_file_ << std::setprecision(10) << " " << o.ex().x << " " << o.ey().x << " "
					  << o.ez().x << " " << o.ex().y << " " << o.ey().y << " " << o.ez().y << " "
					  << o.ex().z << " " << o.ey().z << " " << o.ez().z << " " << mom.x << " "
					  << mom.y << " " << mom.z << " " << ang.x << " " << ang.y << " " << ang.z
					  << "\n";
	}
	rb_traj_file_.flush();
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

void SimManager::report_progress(size_t current_step, size_t total_steps, size_t report_period) {
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
	const float ms_per_step = elapsed_time * 1000.0f / static_cast<float>(total_steps);
	const float io_time = wkf_timer_time(timerS_.timer);
	const float energy_time = wkf_timer_time(timerE_.timer);
	const float compute_time = elapsed_time - io_time - energy_time;
	std::cout << "=========================================" << std::endl;
	std::cout << "SimManager: Performance Summary: " << std::endl;
	std::cout << "  Total time:        " << elapsed_time << " s" << std::endl;
	std::cout << "  Compute time:      " << compute_time << " s ("
			  << compute_time / elapsed_time * 100 << "%)" << std::endl;
	std::cout << "  I/O time:          " << io_time << " s (" << io_time / elapsed_time * 100
			  << "%)" << std::endl;
	std::cout << "  Energy time:       " << energy_time << " s ("
			  << energy_time / elapsed_time * 100 << "%)" << std::endl;
	std::cout << "  ms/Step:      " << steps_per_second << std::endl;
	std::cout << "  ns/day (est):      "
			  << (steps_per_second * sys_.get_timestep() * 86400.0f) / 1e6f << std::endl;
	std::cout << "=========================================" << std::endl;
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
/**
 * @brief
 *
 * @param positions
 * @param types
 */
void SimManager::generate_initial_particles(std::vector<Vector3>& positions,
											std::vector<int>& types) {
	const Vector3 box_size = sys_.get_box_size();

	const size_t num_particles = sys_state_.get_num_particles();

	positions.reserve(num_particles);
	types.reserve(num_particles);

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
	gather_particle_data_from_patches(/*need_energy=*/true);

	const auto& particles = sys_state_.get_global_particles();
	const size_t n = particles.size();

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

	if (!energy_file_.is_open()) {
		energy_file_.open(sys_.get_output_name() + ".energy.dat");
	}
	if (energy_file_.is_open()) {
		energy_file_ << "Kinetic Energy: " << kinetic_energy_kT << " (kT) " << std::endl;
		energy_file_ << "Potential Energy: " << potential_energy << " (kcal/mol) " << std::endl;
	} else {
		LOGWARN("SimManager: Failed to open '{}' for energy output",
				sys_.get_output_name() + ".energy.dat");
	}

	if (has_rigid_bodies_) {
		// TODO: rigid-body kinetic/potential energy is not yet computed by
		// SimManager; write zeros so downstream tooling still gets the file,
		// matching legacy ARBD's rb_energy.dat format.
		if (!rb_energy_file_.is_open()) {
			rb_energy_file_.open(sys_.get_output_name() + ".rb_energy.dat");
		}
		if (rb_energy_file_.is_open()) {
			rb_energy_file_ << "Kinetic Energy 0 (kT)" << std::endl;
			rb_energy_file_ << "Potential Energy 0 (kcal/mol)" << std::endl;
		} else {
			LOGWARN("SimManager: Failed to open '{}' for rigid body energy output",
					sys_.get_output_name() + ".rb_energy.dat");
		}
	}

	write_restart_files();
}

//================================================================================
// Restart Files
//================================================================================

void SimManager::wait_for_pending_restart_write() {
	if (pending_restart_write_.valid()) {
		pending_restart_write_.get();
	}
}

void SimManager::write_restart_files() {
	wait_for_pending_restart_write();

	const auto& particles = sys_state_.get_global_particles();
	std::vector<int> type_id = particles.type_id;
	std::vector<Vector3> pos = particles.pos;
	std::vector<Vector3> mom = has_momentum_output_ ? particles.mom : std::vector<Vector3>{};
	const std::string output_name = sys_.get_output_name();
	const bool write_momentum = has_momentum_output_;

	pending_restart_write_ = std::async(
		std::launch::async,
		[type_id = std::move(type_id),
		 pos = std::move(pos),
		 mom = std::move(mom),
		 output_name,
		 write_momentum]() {
			const std::string restart_filename = output_name + ".restart";
			FILE* out = std::fopen(restart_filename.c_str(), "w");
			if (out) {
				std::string buf;
				buf.reserve(pos.size() * 32);
				for (size_t i = 0; i < pos.size(); ++i) {
					append_restart_line(buf, type_id[i], pos[i]);
				}
				std::fwrite(buf.data(), 1, buf.size(), out);
				std::fclose(out);
			} else {
				LOGWARN("SimManager: Failed to open '{}' for restart output", restart_filename);
			}

			// Momentum restart only applies to Langevin dynamics
			// The "0" in the filename mirrors legacy ARBD's on-disk naming.
			if (write_momentum) {
				const std::string momentum_restart_filename = output_name + ".0.momentum.restart";
				FILE* mout = std::fopen(momentum_restart_filename.c_str(), "w");
				if (mout) {
					std::string buf;
					buf.reserve(mom.size() * 32);
					for (size_t i = 0; i < mom.size(); ++i) {
						append_restart_line(buf, type_id[i], mom[i]);
					}
					std::fwrite(buf.data(), 1, buf.size(), mout);
					std::fclose(mout);
				} else {
					LOGWARN("SimManager: Failed to open '{}' for momentum restart output",
							momentum_restart_filename);
				}
			}
		});
}

void SimManager::write_final_restart() {
	gather_particle_data_from_patches();
	write_restart_files();
	wait_for_pending_restart_write();
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
