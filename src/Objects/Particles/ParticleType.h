#pragma once
#include "Types/BaseGrid.h"
#include "Types/Vector3.h"
#include <string>

namespace ARBD {
struct Particle {
	int id;
	int type_id;
	Vector3 position;
	Vector3 momentum;
	Vector3 force;
	bool is_dummy = false;
	bool has_orientation = false;
	Particle& operator=(const Particle& src) {
		id = src.id;
		type_id = src.type_id;
		position = src.position;
		momentum = src.momentum;
		force = src.force;
		is_dummy = src.is_dummy;
		has_orientation = src.has_orientation;
		return *this;
	}
};

class ParticleType {
  public:
	char name[32];
	struct Properties {
		int id;
		float mass;
		float charge;
		float radius;
		float eps;
		float diffusion;
		float mu; // for Nose-Hoover Langevin dynamics
		int numPartGridFiles;
		float* meanPmf;
		float* pmf_scale;
	};
	BaseGrid<float>** pmf;
	GridConfig<float>::BoundaryCondition* pmf_boundary_conditions;
	BaseGrid<float>* diffusionGrid;
	BaseGrid<float>* forceGrid;
	using props = Properties;
};
} // namespace ARBD
