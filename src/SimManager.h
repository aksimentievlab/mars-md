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
#include "IO/DcdWriter.h"
#include "IO/TrajectoryWriter.h"
#include "IO/WKFUtils.h"
#include "Random/Random.h"
#include "SimSystem.h"
#include "System/PatchManager.h"

// Q: what is our parallel heirarchy?
// A: depends!

// Serial/openMP, MPI-only, Single-GPU, or NVSHMEM

// 1 Patch per MPI rank or GPU
// Patches should work independently with syncronization mediated by SimManager
// Patch to Patch data exchange should not require explicit scheduling by SimManager

namespace ARBD {

class SimManager {
  public:
	SimManager(SimSystem& sys, const ResourceCollection& resources);
	void init();
	void run(); // Main simulation loop

  private:
	SimSystem& sys_;
	ResourceCollection resources_;
	PatchManager patch_manager_;
	CellDecomposer cell_decomp;
	// Timing
	wkfmsgtimer timer0_, timerS_, timerE_;

	// IMD support
	void* clientsock_{nullptr};
	bool imd_on_{false};

	// Output management
	std::unique_ptr<TrajectoryWriter> traj_writer_;
	std::unique_ptr<DcdWriter> dcd_writer_;
	void decompose_system();
	// Kernel functions (no inheritance)
	Event clear_forces();
	Event compute_pair_forces();
	Event compute_bonded_forces();
};

} // namespace ARBD
