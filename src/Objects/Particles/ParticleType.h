#pragma once
#include "Math/BaseGrid.h"
#include <string>

namespace ARBD {
struct Particle {
	int id;
	int type_index;
	Vector3 position;
	Vector3 momentum;
	bool is_dummy = false;
	bool has_orientation = false;
};

class ParticleType {
  public:
	std::string name;
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
	BaseGrid** pmf;
	BoundaryCondition* pmf_boundary_conditions;
	BaseGrid* diffusionGrid;
	BaseGrid* forceXGrid;
	BaseGrid* forceYGrid;
	BaseGrid* forceZGrid;
	using props = Properties;
};
} // namespace ARBD

