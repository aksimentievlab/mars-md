// RigidBodyType.h (2025)
// Author: Chris Maffeo <cmaffeo2@illinois.edu>
// Author: Pin-Yi Li <pin-yi.li@illinois.edu>
// Metal does not support inheritance, so create a new class.
#pragma once
#include "Header.h"
#include "Objects/ParticleProperties.h"
#include "Types/BaseGrid.h"
#include "Types/Types.h"
#include <vector>

namespace ARBD {
class RigidBodyController;
struct RigidBody {
	int id;
	int type_id;
	Vector3 position;
	Matrix3 orientation;
	Vector3 momentum;
	Vector3 angularMomentum;
	Vector3 force;
	Vector3 torque;

	bool is_dummy = false;
	bool has_orientation = false;
	RigidBody& operator=(const RigidBody& src) {
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

class RigidBodyType {
  public:
	std::string name;
	int id;
	float mass;
	Vector3 inertia;
	Vector3 transDamping;
	Vector3 rotDamping;
	Vector3 transForceCoeff;
	Vector3 rotTorqueCoeff;
	float rotational_diffusivity;
	float rotational_damping_coefficient;
	float charge;
	float radius;
	float eps;
	float diffusion;
	float mu; // for Nose-Hoover Langevin dynamics
	int numPartGridFiles;
	float* meanPmf;
	float* pmf_scale;

	std::vector<ParticleRead> attached_particle;
	size_t* potential_grid_idx;
	size_t* density_grid_idx;
	size_t* pmf_grid_idx;

	size_t* potential_grid_idx_d;
	size_t* density_grid_idx_d;
	size_t* pmf_grid_idx_d;

	RigidBodyController* RBC;
	BaseGrid<float>** pmfGrid;
	BaseGrid<float>* diffusionGrid;
	BaseGrid<float>* forceGrid;
};

} // namespace ARBD
