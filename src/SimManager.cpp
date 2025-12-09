#include "SimManager.h"
#include "System/PatchManager.h"

namespace ARBD {

//================================================================================
// Constructor
//================================================================================

SimManager::SimManager(SimSystem& sys) : sys_(sys), sys_state_(sys) {
	timer0_.timer = wkf_timer_create();
	timerS_.timer = wkf_timer_create();
	timerE_.timer = wkf_timer_create();
}

//================================================================================
// Initialization
//================================================================================

void SimManager::init() {
	wkf_timer_start(&timer0_);
	LOGINFO("SimManager: Initializing simulation");

	// Initialize output writers based on configuration
	initialize_output_writers();

	// Initialize decomposer if not already set
	if (!sys_.get_decomposer()) {
		LOGINFO("SimManager: Setting up default spatial decomposer");
		sys_.set_decomposer_type(sys_.get_decomposer_type());
	}

	// Perform domain decomposition (creates PatchManager in SimSystem)
	LOGINFO("SimManager: Performing domain decomposition");
	sys_.decompose_system();

	// Verify PatchManager was created
	if (!sys_.has_patch_manager()) {
		throw Exception(ExceptionType::RuntimeError,
						SourceLocation(),
						"Domain decomposition failed to create PatchManager");
	}
	LOGINFO("SimManager: Domain decomposition complete");

	// Transfer grids to all GPU resources
	sys_.get_grid_manager().build_device_arrays();
	LOGINFO("SimManager: Grids transferred to all resources");
	sys_.get_tables_registry().build_device_arrays();
	LOGINFO("SimManager: Tables transferred to all resources");

	// Load initial conditions (particles, bonds, etc.)
	load_initial_conditions();

	// Initialize IMD if requested
	// TODO: Add IMD support when needed
	// if (sys_.get_config().imd_enabled) {
	//     initialize_imd(sys_.get_config().imd_port);
	// }

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
	const std::string output_name = "out"; // TODO: Get from SimSystem

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
	// Access PatchManager through SimSystem
	PatchManager* patch_mgr = sys_.get_patch_manager();
	if (!patch_mgr) {
		throw Exception(ExceptionType::RuntimeError,
						SourceLocation(),
						"PatchManager not available for force calculation");
	}

	// TODO: Implement force calculation
	// For each patch:
	//   1. Update neighbor list if needed
	//   2. Clear forces
	//   3. Compute pairwise forces
	//   4. Compute bonded forces if present
	//   5. Apply external forces (electric field, grids)

	// Placeholder logging
	if (step == 1) {
		LOGINFO("SimManager: Force calculation not yet implemented");
	}
}

//================================================================================
// Integration Phase
//================================================================================

void SimManager::execute_integration(size_t step) {
	const DynamicType particle_algorithm = sys_.get_particle_algorithm();
	const DynamicType rigidbody_algorithm = sys_.get_rigid_body_algorithm();
	const float timestep = sys_.get_timestep();
	const float temperature = sys_.get_temperature();

	// Access PatchManager through SimSystem
	PatchManager* patch_mgr = sys_.get_patch_manager();
	if (!patch_mgr) {
		throw Exception(ExceptionType::RuntimeError,
						SourceLocation(),
						"PatchManager not available for integration");
	}

	// TODO: Implement integration for each patch
	// Select integrator based on algorithm type
	// Execute integration kernel on each resource

	// Placeholder logging
	if (step == 1) {
		LOGINFO("SimManager: Integration (algorithm: {}, dt: {}, T: {}) not yet implemented",
				static_cast<int>(particle_algorithm),
				timestep,
				temperature);
	}
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

		// Periodic restart files
		// TODO: Add restart file writing
		// const size_t restart_period = 10000; // from config
		// if (step % restart_period == 0) {
		//     write_restart_file(step);
		// }

		wkf_timer_stop(&timerS_);
	}
}

void SimManager::gather_particle_data_from_patches() {
	PatchManager* patch_mgr = sys_.get_patch_manager();
	if (!patch_mgr) {
		throw Exception(ExceptionType::RuntimeError,
						SourceLocation(),
						"PatchManager not available for gathering particle data");
	}

	SystemState& state = sys_state_;
	state.clear_global_arrays();

#ifdef USE_MPI
	// In MPI mode: gather from local patch + exchange with other ranks
	const Patch& local_patch = patch_mgr->get_local_patch();
	const HostParticleData& particle_data = local_patch.get_particle_data();
	idx_t local_num = local_patch.get_num();

	// Collect from local patch
	std::vector<Vector3> positions;
	std::vector<Vector3> momenta;
	std::vector<int> ids;
	positions.reserve(local_num);
	momenta.reserve(local_num);
	ids.reserve(local_num);

	for (idx_t i = 0; i < local_num; ++i) {
		positions.push_back(particle_data.pos[i]);
		momenta.push_back(particle_data.mom[i]);
		ids.push_back(particle_data.id[i]);
	}

	state.add_particle_data(positions, momenta, ids);

	// TODO: MPI_Gatherv to collect from all ranks and assemble in global order
	// This requires coordination with PatchManager's MPI communication

#else
	// Non-MPI: collect from all local patches
	const auto& patches = patch_mgr->get_all_patches();
	for (const auto& patch : patches) {
		const HostParticleData& particle_data = patch.get_particle_data();
		idx_t num = patch.get_num();

		std::vector<Vector3> positions;
		std::vector<Vector3> momenta;
		std::vector<int> ids;
		positions.reserve(num);
		momenta.reserve(num);
		ids.reserve(num);

		for (idx_t i = 0; i < num; ++i) {
			positions.push_back(particle_data.pos[i]);
			momenta.push_back(particle_data.mom[i]);
			ids.push_back(particle_data.id[i]);
		}

		state.add_particle_data(positions, momenta, ids);
	}
#endif

	state.mark_synced();
}

void SimManager::write_dcd_frame(size_t step) {
	// Gather global state from patches
	gather_particle_data_from_patches();

	// Get global positions for DCD writing
	const auto& positions = sys_state_.get_global_positions();

	if (positions.empty()) {
		LOGWARN("SimManager: No particles to write at step {}", step);
		return;
	}

	// Write DCD frame
	if (dcd_writer_) {
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

void SimManager::load_initial_conditions() {
	// SystemState is already initialized in constructor
	// TODO: Load actual particle data from files or generate initial positions
	LOGINFO("SimManager: Initial conditions loaded");
}

void SimManager::generate_initial_particles(std::vector<Vector3>& positions,
											std::vector<int>& types) {
	const Vector3 box_size = sys_.get_box_size();

	// TODO: Get number of particles from configuration
	const size_t num_particles = 1000; // Placeholder

	positions.reserve(num_particles);
	types.reserve(num_particles);

	// Simple random placement for testing
	// TODO: Replace with proper initial condition generation
	for (size_t i = 0; i < num_particles; ++i) {
		positions.emplace_back(box_size.x * (float)rand() / RAND_MAX,
							   box_size.y * (float)rand() / RAND_MAX,
							   box_size.z * (float)rand() / RAND_MAX);
		types.push_back(0); // All type 0 for now
	}

	LOGINFO("SimManager: Generated {} particles in box {}", num_particles, box_size);
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

} // namespace ARBD
