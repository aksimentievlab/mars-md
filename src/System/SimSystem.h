#pragma once

#include "ARBDException.h"
#include "ARBDLogger.h"
#include "Backend/Buffer.h"
#include "Math/IndexList.h"
#include "Objects/BasePatch.h"
//#include "Objects/Particles/ParticlePatch.h"
#include "Math/Types.h"

// Class providing a description of a simulation system, including composition, coordinates,
// interactions, state parameters (temperature, boundary conditions, etc) + RNG counter

//   Although only one instance of this should be created per replica of the system, it should be
//   possible to distribute (at least parts of) the description of the system

namespace ARBD {
struct Temperature {
	float value;
};
struct Length {
	float value;
};
struct ResourceCollection {
	std::vector<Resource> resources;
};
class Patch;

class BoundaryConditions {

  public:
	BoundaryConditions()
		: origin(0,0,0), basis{Vector3(5000, 0, 0), Vector3(0, 5000, 0), Vector3(0, 0, 5000)},
		  periodic{true, true, true} {
		// do things
		LOGINFO("BoundaryConditions()");
	}
	BoundaryConditions(Vector3 basis1,
					   Vector3 basis2,
					   Vector3 basis3,
					   Vector3 origin = Vector3(0.0f, 0.0f, 0.0f),
					   bool periodic1 = true,
					   bool periodic2 = true,
					   bool periodic3 = true) {
		// do things
		LOGINFO("BoundaryConditions(...)");
	}

	static constexpr size_t dim = 3;
	Vector3 origin;
	Vector3 basis[dim];
	bool periodic[dim];
};

/**
 * Class that operates on sys and its data, creating Patch objects and moving data as needed
 *
 * Q1: Should this normally only happen at initialization? Future decompositions should probably
 * be expressed as a series of PatchOp objects
 *
 * Q2: Should this also be the object that converts
 * SymbolicPatchOps to concrete PatchOp? Probably because this object should be the only one aware
 * of the details of the decomposition
 */


class SimSystem {
//friend class CellDecomposer;
  public:
	struct Conf {
		enum Decomposer { CellDecomp };
		enum Periodicity { AllPeriodic, Free, OneDimensional, TwoDimensional };
		enum Algorithm { BD, MD };

		Temperature temperature;
		Decomposer decomp;
		Periodicity periodicity;
		const float cell_lengths[3];
		const float cutoff;

		// explicit operator int() const {return object_type*16 + algorithm*4 + backend;};
	};

	inline static constexpr Conf default_conf =
		Conf{290.0f, Conf::CellDecomp, Conf::AllPeriodic, {5000.0f, 5000.0f, 5000.0f}, 50.0f};

	SimSystem() : SimSystem(default_conf) {}
	// temperature(291.0f), decomp(), boundary_conditions() {}
	SimSystem(const Conf& conf) : temperature(conf.temperature), cutoff(conf.cutoff) {

		switch (conf.decomp) {
		case Conf::CellDecomp:
			CellDecomposer _d;
			decomp = static_cast<Decomposer>(_d);
			break;
		default:
			throw_value_error("SimSystem::GetIntegrator: Unrecognized CellDecomp type; exiting");
		}

		// Set up boundary_conditions
		switch (conf.periodicity) {
		case Conf::AllPeriodic:
			boundary_conditions = BoundaryConditions(Vector3(conf.cell_lengths[0], 0, 0),
													 Vector3(0, conf.cell_lengths[1], 0),
													 Vector3(0, 0, conf.cell_lengths[2]));
			break;
		default:
			throw_value_error("Integrator::GetIntegrator: Unrecognized algorithm type; exiting");
		}

		// decomp = static_cast<Decomposer>( CellDecompser() );
		// decomp(decomp), boundary_conditions(boundary_conditions) {}
	}
	SimSystem(Temperature& temp, Decomposer& decomp, BoundaryConditions boundary_conditions)
		: temperature(temp), decomp(decomp), boundary_conditions(boundary_conditions) {}
	
    Patch patches; // TODO: should this be a vector?

	const Vector3 get_min() const {
		Vector3 min(Vector3::highest());
		for (auto& p : patches) {
			if (min.x > p.metadata->min.x)
				min.x = p.metadata->min.x;
			if (min.y > p.metadata->min.y)
				min.y = p.metadata->min.y;
			if (min.z > p.metadata->min.z)
				min.z = p.metadata->min.z;
			if (min.w > p.metadata->min.w)
				min.w = p.metadata->min.w;
		}
		return min;
	}

	const Vector3 get_max() const {
		Vector3 max(Vector3::lowest());
		for (auto& p : patches) {
			if (max.x < p.metadata->max.x)
				max.x = p.metadata->max.x;
			if (max.y < p.metadata->max.y)
				max.y = p.metadata->max.y;
			if (max.z < p.metadata->max.z)
				max.z = p.metadata->max.z;
			if (max.w < p.metadata->max.w)
				max.w = p.metadata->max.w;
		}
		return max;
	}

	void use_decomposer(Decomposer& d) {
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

	// DeviceBuffer<Particle> particles;
	//  std::vector<SymbolicOp> Interactions;
	//  std::vector<SymbolicOp> computations;

	// size_t particle_count;
	// Array<size_t>* particle_types;

	// size_t rigid_Body_count;
	// Array<size_t>* rigid_body_types;

	Length cutoff;
	Decomposer decomp;
	BoundaryConditions boundary_conditions;
};
} // namespace ARBD
// inline void Decomposer::decompose(SimSystem& sys, ResourceCollection& resources) {
//     // sys.patches
// };
