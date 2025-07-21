/*********************************************************************
 * @file  Patch.h
 * 
 * @brief Declaration of BasePatchOp class.
 * 
 * @details This file contains the declaration of the abstract base
 *          class BasePatchOp, which operates on Patch data. It also
 *          includes headers of derived classes for convenient access
 *          to factory methods. The virtual method
 *          `BasePatchOp::compute(Patch* patch)` is called by
 *          `patch->compute()` by any Patch to which the PatchOp has
 *          been added.
 *********************************************************************/

#pragma once
#include "PatchOp.h"


// template<typename Pos, typename Force>
// class Patch {
//     // size_t num_replicas;

// public:    
//     Patch(SimParameters* sim);
	
    
//     void compute();
    
// private:

//     SimParameters* sim;
    
//     Pos minimum;
//     Pos maximum;

//     ParticleTypes* types;
//     ParticleTypes* types_d;

//     // Particle data
//     size_t num_particles;

//     idx_t* global_idx;		// global index of particle
//     size_t* type_ids;
//     Pos* pos;
//     Force* force;
    
//     size_t* type_ids_d;
//     Pos* pos_d;
//     Force* force_d;

//     std::vector<Patch> neighbor_patches;
//     std::vector<BasePatchOp> ops;
// };
