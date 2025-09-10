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
#include "SimSystem.h"
#include "System/PatchManager.h"

// Q: what is our parallel heirarchy?
// A: depends!

// Serial/openMP, MPI-only, Single-GPU, or NVSHMEM

// 1 Patch per MPI rank or GPU
// Patches should work independently with syncronization mediated by SimManager
// Patch to Patch data exchange should not require explicit scheduling by SimManager

namespace ARBD {

struct LoadBalancer {
	void balance(SimSystem& sys, const ResourceCollection& resources);
};

class SimManager {

  public:
	SimManager(SimSystem& sys, const ResourceCollection& resources);

  private:
	LoadBalancer load_balancer;
	SimSystem sys; // make it a list for replicas
	CellDecomposer cell_decomp;
	ResourceCollection resources;
	PatchManager patch_manager;
	// std::vector<SymbolicOp> sym_ops;
	// std::vector<PatchOp>  ops;

  public:
	class CheckPairlist {
	  public:
		CheckPairlist(SimSystem& sys, const ResourceCollection& resources);
		void check_pairlist(SimSystem& sys, const ResourceCollection& resources);

	  private:
		SimSystem& sys_;
		ResourceCollection resources_;
	};
	void run();
};
} // namespace ARBD
