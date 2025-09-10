// v1 RBType.h
// Metal does not support inheritance, so create a new class.
#pragma once
#include "Header.h"
#include "Types/BaseGrid.h"
#include "Types/Types.h"

namespace ARBD {
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
	char name[32];
	struct Properties {
		int id;
		float mass;
		Vector3 inertia;
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
	};
	BaseGrid<float>** pmfGrid;
	BaseGrid<float>* diffusionGrid;
	BaseGrid<float>* forceGrid;
	using props = Properties;
};
} // namespace ARBD
