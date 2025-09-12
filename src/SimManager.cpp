#include "SimManager.h"

namespace ARBD {
void SimManager::init() {
	wkf_timer_start(&timer0_);

	// Initialize domain decomposition
	sys_.decompose_system();

	// Initialize output writers
	if (sys_.conf.outputFormat == OutputFormat::DCD) {
		dcd_writer_ = std::make_unique<DcdWriter>(sys_.conf.outputFile);
	}

	// Initialize IMD if requested
	if (sys_.conf.imd_on) {
		initialize_imd(sys_.conf.imd_port);
	}

	// Load restart or initialize
	if (!sys_.conf.restartFile.empty()) {
		sys_.load_restart(sys_.conf.restartFile);
	} else {
		sys_.initialize_particles();
	}
}

void SimManager::run() {
	// ===== INITIALIZATION PHASE =====

	// ===== MAIN SIMULATION LOOP =====
	const size_t numSteps = sys_.conf.numSteps;
	const size_t outputPeriod = sys_.conf.outputPeriod;
	const size_t outputEnergyPeriod = sys_.conf.outputEnergyPeriod;

	for (size_t step = 1; step <= numSteps; ++step) {

		// Build kernel pipeline for this timestep
		for (auto& [resource, rng] : rngs_) {

			// Create pipeline for this resource
			KernelPipeline pipeline(resource);

			// Get buffers for this resource's patch
			auto& buffers = patch_manager_.get_patch(resource)->get_buffers();

			// ===== FORCE CALCULATION PHASE =====

			// 1. Update neighbor list if needed
			if (step % sys_.conf.neighborListPeriod == 0) {
				pipeline.then(update_neighbor_list_kernel,
							  buffers.positions,
							  patch_manager_.get_neighbor_list(resource));
			}

			// 2. Clear forces
			pipeline.then(clear_forces_kernel, buffers.forces);

			// 3. Compute pairwise forces
			pipeline.then(compute_pair_forces_kernel,
						  buffers.positions,
						  buffers.forces,
						  buffers.types,
						  patch_manager_.get_neighbor_list(resource));

			// 4. Compute bonded forces if present
			if (sys_.has_bonds()) {
				pipeline.then(compute_bond_forces_kernel,
							  buffers.positions,
							  buffers.forces,
							  sys_.get_bond_list());
			}

			// 5. Apply external forces (electric field, grids)
			if (sys_.has_external_forces()) {
				pipeline.then(compute_external_forces_kernel,
							  buffers.positions,
							  buffers.forces,
							  buffers.types,
							  sys_.get_electric_field(),
							  sys_.get_force_grid());
			}

			// ===== INTEGRATION PHASE =====

			// Select integrator based on algorithm
			switch (sys_.conf.algorithm) {
			case SimSystem::Conf::Algorithm::Langevin:
				// BAOAB integrator
				pipeline.then(baoab_integrate_kernel,
							  buffers.positions,
							  buffers.momenta,
							  buffers.forces,
							  buffers.types,
							  sys_.conf.timestep,
							  sys_.conf.temperature,
							  rng->get_state(),
							  step == 1 // first step flag
				);
				break;

			case SimSystem::Conf::Algorithm::NoseHooverLangevin:
				// Nose-Hoover-Langevin integrator
				pipeline.then(nose_hoover_integrate_kernel,
							  buffers.positions,
							  buffers.momenta,
							  buffers.random,
							  buffers.forces,
							  buffers.types,
							  sys_.conf.timestep,
							  sys_.conf.temperature,
							  rng->get_state(),
							  step == 1);
				break;

			case SimSystem::Conf::Algorithm::BD:
				// Brownian dynamics integrator
				pipeline.then(brownian_integrate_kernel,
							  buffers.positions,
							  buffers.forces,
							  buffers.types,
							  sys_.conf.timestep,
							  sys_.conf.temperature,
							  rng->get_state());
				break;
			}

			// Synchronize this resource's pipeline
			pipeline.synchronize();
		}

		// ===== MULTI-RESOURCE SYNCHRONIZATION =====
		if (resources_.size() > 1) {
			// Exchange halos between patches
			patch_manager_.exchange_halos();
		}

		// ===== OUTPUT PHASE =====

		// Energy calculation
		if (step % outputEnergyPeriod == 0) {
			wkf_timer_start(timerE_);

			EventList energy_events;
			for (auto& [resource, rng] : rngs_) {
				auto& buffers = patch_manager_.get_patch(resource)->get_buffers();

				// Launch energy reduction kernel
				Event e = launch_kernel(resource,
										KernelConfig{},
										compute_total_energy_kernel,
										buffers.positions,
										buffers.momenta,
										buffers.energy);
				energy_events.add(e);
			}

			energy_events.wait_all();
			output_energy(step);

			wkf_timer_stop(timerE_);
		}

		// Trajectory output
		if (step % outputPeriod == 0) {
			wkf_timer_start(timerS_);

			if (dcd_writer_) {
				write_dcd_frame(step);
			}

			// Write restart files
			if (step % sys_.conf.restartPeriod == 0) {
				for (int repID = 0; repID < sys_.conf.numReplicas; ++repID) {
					sys_.write_restart(repID);
				}
			}

			wkf_timer_stop(timerS_);
		}

		// IMD handling
		if (imd_on_ && clientsock_) {
			handle_imd_commands();
		}

		// Progress reporting
		if (step % 1000 == 0) {
			report_progress(step, numSteps);
		}
	}

	// ===== FINALIZATION =====
	wkf_timer_stop(timer0_);

	const float elapsed = wkf_timer_time(timer0_);
	report_performance(elapsed, numSteps);

	// Final restart
	for (int repID = 0; repID < sys_.conf.numReplicas; ++repID) {
		sys_.write_restart(repID);
	}

	// Cleanup IMD
	if (imd_on_ && clientsock_) {
		imd_disconnect(clientsock_);
	}
} // namespace ARBD::SimManager::run()
} // namespace ARBD
