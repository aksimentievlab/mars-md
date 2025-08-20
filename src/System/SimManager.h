#pragma once
#include <iostream>


#include "System/SimSystem.h"
#include "ARBDLogger.h"
#include "ARBDException.h"

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
    SimSystem sys;	// make it a list for replicas
    CellDecomposer cell_decomp;
    //std::vector<SymbolicOp> sym_ops;
    //std::vector<PatchOp>  ops;
    
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
}