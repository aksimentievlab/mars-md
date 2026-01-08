#pragma once

/**
 * @file SimManager.h
 * @author Pin-Yi Li <pinyili2@illinois.edu>
 * @brief Simulation manager class. Manages the simulation loop and the parallelization.
 * @version 0.1
 * @date 2025-09-09
 *
 * @copyright Copyright (c) 2025
 */

#include <iostream>

#include "ARBDException.h"
#include "ARBDLogger.h"
#include "Backend/Buffer.h"
#include "Backend/Events.h"
#include "Backend/Kernels.h"
#include "Backend/Profiler.h"
#include "Backend/Resource.h"
#include "IO/ConfigParser.h"
#include "IO/DcdWriter.h"
#include "IO/TrajectoryWriter.h"
#include "IO/WKFUtils.h"
#include "System/SimSystem.h"
#include "System/SystemState.h"

// Q: what is our parallel heirarchy?
// A: depends!

// Serial/openMP, MPI-only, Single-GPU, or NVSHMEM

// 1 Patch per MPI rank or GPU
// Patches should work independently with syncronization mediated by SimManager
// Patch to Patch data exchange should not require explicit scheduling by SimManager

namespace ARBD {

/**
 * @brief Simulation manager - handles the main simulation loop and runtime operations
 *
 * SimManager is responsible for:
 * - Managing the simulation loop
 * - Coordinating kernel execution across resources
 * - Handling output
 * - Managing inter-patch communication and synchronization
 * - Performance monitoring and reporting
 */
class SimManager {
  public:
	/**
	 * @brief Construct simulation manager
	 * @param sys Simulation system containing configuration and global objects
	 */
	SimManager(SimSystem& sys);
	/**
	 * @brief Construct simulation manager
	 * @param sys Simulation system containing configuration and global objects
	 * @param parser Configuration parser
	 */
	SimManager(SimSystem& sys, const ConfigParser& parser);

	/**
	 * @brief Initialize simulation manager
	 * Sets up decomposition, output writers, IMD, and initial conditions
	 */
	void init();

	/**
	 * @brief Initialize simulation manager configuration
	 * @param parser Configuration parser
	 */
	void load_config(const ConfigParser& parser);
	/**
	 * @brief Run the main simulation loop
	 * Executes the complete simulation with force calculation, integration, and I/O
	 */
	void run();

	/**
	 * @brief Get timing information
	 */
	float get_total_time() const {
		return wkf_timer_time(timer0_.timer);
	}
	float get_io_time() const {
		return wkf_timer_time(timerS_.timer);
	}
	float get_energy_time() const {
		return wkf_timer_time(timerE_.timer);
	}

  private:
	//================================================================================
	// Core Components
	//================================================================================
	SimSystem& sys_; // System owns PatchManager, accessible via sys_.get_patch_manager()
	SystemState sys_state_;

	// Random number generation
	size_t current_step_{0}; ///< Current simulation step (used for RNG counter)
	//================================================================================
	// Timing and Performance
	//================================================================================
	wkfmsgtimer timer0_, timerS_, timerE_;

	//================================================================================
	// I/O and Output Management
	//================================================================================
	std::unique_ptr<TrajectoryWriter> traj_writer_;
	std::unique_ptr<DcdWriter> dcd_writer_;

	//================================================================================
	// IMD (Interactive Molecular Dynamics) Support
	//================================================================================
	void* clientsock_{nullptr};
	bool imd_on_{false};

	//================================================================================
	// Initialization Methods
	//================================================================================

	/**
	 * @brief Initialize output writers based on configuration
	 */
	void initialize_output_writers();

	/**
	 * @brief Initialize IMD if requested
	 * @param port IMD port number
	 */
	void initialize_imd(int port);

	//================================================================================
	// Simulation Loop Components
	//================================================================================
	/**
	 * @brief Execute force calculation phase for all resources
	 * @param step Current simulation step
	 */
	void execute_force_calculation(size_t step);

	/**
	 * @brief Execute integration phase for all resources
	 * @param step Current simulation step
	 */
	void execute_integration(size_t step);

	/**
	 * @brief Synchronize multi-resource simulations (halo exchange)
	 */
	void synchronize_multi_resource();

	/**
	 * @brief Handle output operations (trajectory, energy, restart)
	 * @param step Current simulation step
	 */
	void handle_output(size_t step);

	//================================================================================
	// Output Methods
	//================================================================================

	/**
	 * @brief Gather particle data from all patches into SystemState
	 * Collects positions, momenta, and IDs from all patches and assembles
	 * them into global arrays in SystemState for output.
	 */
	void gather_particle_data_from_patches();

	/**
	 * @brief Write a single DCD trajectory frame
	 * @param step Current simulation step
	 */
	void write_dcd_frame(size_t step);
	/**
	 * @brief Report simulation progress
	 * @param current_step Current step
	 * @param total_steps Total steps
	 */
	void report_progress(size_t current_step, size_t total_steps);

	/**
	 * @brief Report final performance statistics
	 * @param elapsed_time Total simulation time
	 * @param total_steps Total steps completed
	 */
	void report_performance(float elapsed_time, size_t total_steps);

	//================================================================================
	// IMD Methods
	//================================================================================

	/**
	 * @brief Handle IMD commands and communication
	 */
	void handle_imd_commands();

	//================================================================================
	// I/O Methods (SimManager handles file I/O, not SimSystem)
	//================================================================================

	/**
	 * @brief Load initial conditions from files or generate them
	 */
	void load_initial_conditions();

	/**
	 * @brief Generate initial particle positions and types
	 * @param positions Output vector for particle positions
	 * @param types Output vector for particle types
	 */
	void generate_initial_particles(std::vector<Vector3>& positions, std::vector<int>& types);

	/**
	 * @brief Write final restart file
	 */
	void write_final_restart();
	void load_restart_data(const std::string& filename);

	/**
	 * @brief Perform particle reactions
	 */
	void perform_reactions();
};

} // namespace ARBD
