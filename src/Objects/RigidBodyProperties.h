// RigidBodyType.h (2025)
// Author: Chris Maffeo <cmaffeo2@illinois.edu>
// Author: Pin-Yi Li <pinyili2@illinois.edu>
// Metal does not support inheritance, so create a new class.
#pragma once
#include "Header.h"
#include "Objects/ParticleProperties.h"
#include "Types/BaseGrid.h"
#include "Types/Types.h"
#include <vector>

namespace ARBD {

struct RigidBodyIO {
	int id;
	int type_id;
	Vector3 position;
	Matrix3 orientation;
	Vector3 momentum;
	Vector3 angular_momentum;
	Vector3 force;
	Vector3 torque;

	bool is_dummy = false;
	bool has_orientation = false;
	RigidBodyIO& operator=(const RigidBodyIO& src) {
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
	Vector3 trans_damping;
	Vector3 rot_damping;
	Vector3 trans_force_coeff;
	Vector3 rot_torque_coeff;
	float rot_diffusivity;
	float rot_damping_coefficient;
	float charge;
	float radius;
	float eps;
	float diffusion;
	float mu; // for Nose-Hoover Langevin dynamics
	int num_grid_files;
	// float meanPmf;
	float pmf_scale;
	float pmf_scale_slope;
	uint32_t pmf_smd_freq;

	std::vector<ParticleIO> attached_particle;
	std::vector<uint32_t> potential_grid_ids;
	std::vector<uint32_t> density_grid_ids;
	std::vector<uint32_t> pmf_grid_ids;
};

} // namespace ARBD
