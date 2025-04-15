#pragma once

#include "Types.h"
#include "ParticlePatch.h"
#include "PatchOp.h"
#include "Proxy.h"

// Class providing a description of a simulation system, including composition, coordinates, interactions, state parameters (temperature, boundary conditions, etc)
// Also includes RNG state

//   Although only one instance of this should be created per replica of the system, it should be possible to distribute (at least parts of) the description of the system

using Temperature = float;	// TODO: replace with units object
using Length = float;

class SimSystem;

class BoundaryConditions {
    friend class Decomposer;

public:
    BoundaryConditions() :  origin(0), basis{(5000,0,0),(0,5000,0),(0,0,5000)}, periodic{true,true,true} {
	// do things
	LOGINFO("BoundaryConditions()");
    }
    BoundaryConditions( Vector3 basis1, Vector3 basis2, Vector3 basis3, Vector3 origin = Vector3(0), bool periodic1 = true, bool periodic2 = true, bool periodic3 = true ) {
	// do things
	LOGINFO("BoundaryConditions(...)");
    }

    static constexpr size_t dim = 3;
    Vector3 origin;
    Vector3 basis[dim];
    bool periodic[dim];    
};

// class ProxyPatch : public Proxy<Patch>, public BasePatch {
// public:
//     ProxyPatch(Resource& r, Patch* obj) : location(&r), addr(obj) { };
//     Resource* location;
//     Patch* addr;
//     size_t num;
//     Vector3 min;
//     Vector3 max;    
// };

struct ResourceCollection {
    
    // // Not sure yet what all will go here

    // `send_to` functionality below already implemented with templated `send`
    // template<typename T>
    // Proxy<T> send_to(Resource& r, T* obj) {
    // 	// ...
    // 	Proxy<T> obj_proxy(r, nullptr);
    // 	if (r.type == Resource::CPU) {
    // 	    // obj_proxy.addr = 
    // 	    //...
    // 	} else if (r.type == Resource::GPU) {
    // 	    // obj_proxy.addr = 
    // 	    // ...
    // 	}
    // }

    std::vector<Resource> resources;
};

// Class that operates on sys and its data, creating Patch objects and moving data as needed
//   Q1: Should this normally only happen at initialization? Future decompositions should probably be expressed as a series of PatchOp objects
//   Q2: Should this also be the object that converts SymbolicPatchOps to concrete PatchOp? Probably because this object should be the only one aware of the details of the decomposition
class Decomposer {
    // Make virtual?
    inline void decompose(SimSystem& sys, ResourceCollection& resources);

    // Also update PatchOp objects...
    void concretize_symbolic_ops(SimSystem& sys) {}
};

class CellDecomposer : public Decomposer {
    inline void decompose(SimSystem& sys, ResourceCollection& resources);
    struct Worker {
	Length cutoff;
	std::vector<Proxy<Patch>> patches;
    };
};

class SimSystem {
    friend class Decomposer;
    friend class CellDecomposer;
    friend class SimManager;
public:
    struct Conf {
	enum Decomposer   { CellDecomp };
	enum Periodicity { AllPeriodic };
	// enum Algorithm { BD, MD };
	// enum Backend   { Default, CUDA, CPU };    

	Temperature temperature;
	Decomposer decomp;
	Periodicity periodicity;
	Length cell_lengths[3];
	Length cutoff;		// not sure this belongs here
	// Object object_type;
	// Algorithm algorithm;
	// Backend backend;

	// explicit operator int() const {return object_type*16 + algorithm*4 + backend;};
    };

    inline			// C++17 feature needed for compilation... unsure of why
    static constexpr Conf default_conf = Conf{291.0f, Conf::CellDecomp, Conf::AllPeriodic, {5000.0f,5000.0f,5000.0f}, 50.0f };

    SimSystem() : SimSystem(default_conf) {}
    // temperature(291.0f), decomp(), boundary_conditions() {}
    SimSystem(const Conf& conf) : temperature(conf.temperature), cutoff(conf.cutoff) {

	// Set up decomposition
	switch (conf.decomp) {
	case Conf::CellDecomp:
	    CellDecomposer _d;
	    decomp = static_cast<Decomposer>(_d);
	    break;
	default:
	    Exception( ValueError, "SimSystem::GetIntegrator: Unrecognized CellDecomp type; exiting" );
	    // std::cerr << "Error: SimSystem::GetIntegrator: "
	    // 	      << "Unrecognized CellDecomp type; exiting" << std::endl;
	    // assert(false);
	}

	// Set up boundary_conditions
	switch (conf.periodicity) {
	case Conf::AllPeriodic:
	    boundary_conditions = BoundaryConditions( Vector3{conf.cell_lengths[0],0,0}, Vector3{0,conf.cell_lengths[1],0}, Vector3{0,0,conf.cell_lengths[2]} );
	    break;
	default:
	    Exception( ValueError, "Integrator::GetIntegrator: Unrecognized algorithm type; exiting" );
	    // std::cerr << "Error: Integrator::GetIntegrator: "
	    // 	      << "Unrecognized algorithm type; exiting" << std::endl;
	    // assert(false);
	}
	
	// decomp = static_cast<Decomposer>( CellDecompser() );
	// decomp(decomp), boundary_conditions(boundary_conditions) {}
    }
    SimSystem(Temperature& temp, Decomposer& decomp, BoundaryConditions boundary_conditions) :
	temperature(temp), decomp(decomp), boundary_conditions(boundary_conditions) {}

    const Vector3 get_min() const {
	Vector3 min(Vector3::highest());
	for (auto& p: patches) {
	    if (min.x > p.metadata->min.x) min.x = p.metadata->min.x;
	    if (min.y > p.metadata->min.y) min.y = p.metadata->min.y;
	    if (min.z > p.metadata->min.z) min.z = p.metadata->min.z;
	    if (min.w > p.metadata->min.w) min.w = p.metadata->min.w;
	}
	return min;
    }

    const Vector3 get_max() const {
	Vector3 max(Vector3::lowest());
	for (auto& p: patches) {
	    if (max.x < p.metadata->max.x) max.x = p.metadata->max.x;
	    if (max.y < p.metadata->max.y) max.y = p.metadata->max.y;
	    if (max.z < p.metadata->max.z) max.z = p.metadata->max.z;
	    if (max.w < p.metadata->max.w) max.w = p.metadata->max.w;
	}
	return max;
    }
    
    void use_decomposer(Decomposer &d) {
	decomp = d;
    }
	       
    // void consolidate_patches(); // maybe not needed
    void distribute_patches() {
	LOGINFO("distribute_patches()");
	decomp;
    }

protected:
    Temperature temperature;
    // std::vector<Interactions> interactions; // not quite..
    
    std::vector<Proxy<Patch>> patches;
    std::vector<SymbolicOp> Interactions;
    std::vector<SymbolicOp> computations;
    
    // size_t particle_count;
    // Array<size_t>* particle_types;

    // size_t rigid_Body_count;
    // Array<size_t>* rigid_body_types;

    Length cutoff;
    Decomposer decomp;
    BoundaryConditions boundary_conditions;
};

// inline void Decomposer::decompose(SimSystem& sys, ResourceCollection& resources) {
//     // sys.patches
// };
