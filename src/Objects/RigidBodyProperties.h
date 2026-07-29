// RigidBodyType.h (2025)
// Author: Chris Maffeo <cmaffeo2@illinois.edu>
// Author: Pin-Yi Li <pinyili2@illinois.edu>
// Metal does not support inheritance, so create a new class.
#pragma once
#include "Header.h"
#include "Objects/ParticleProperties.h"
#include "Types/BaseGrid.h"
#include "Types/GridTerm.h"
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
		orientation = src.orientation;
		momentum = src.momentum;
		angular_momentum = src.angular_momentum;
		force = src.force;
		torque = src.torque;
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
	// One GridTerm (grid_id + scale + scale_slope + boundary_condition) per
	// referenced grid - mirrors ParticleType::pmf_grids (see Types/GridTerm.h)
	// instead of separate parallel id/scale arrays, since a grid-force kernel
	// reads one of these per grid rather than combining several arrays.
	// Parallel to (same length/order as) the key vectors below.
	std::vector<GridTerm> potential_grids;
	std::vector<GridTerm> density_grids;
	std::vector<GridTerm> pmf_grids;
	// Logical grid-key name per entry, parallel to the grid lists above -
	// legacy: potentialGridKeys/densityGridKeys/pmfKeys. Distinct from the
	// grid's filename/grid_id: two types loading different grid files pair up
	// for a grid-grid force whenever their key names match (e.g. both call a
	// grid "Elec"), which is how RigidBodyForcePairList (Phase 3) finds
	// force-pair candidates.
	std::vector<std::string> potential_grid_keys;
	std::vector<std::string> density_grid_keys;
	std::vector<std::string> pmf_keys;
};

// ============================================================================
// HOST STORAGE (SoA, for device upload/download) - mirrors HostParticleData
// ============================================================================
struct HostRigidBodyData {
	std::vector<int> global_id;
	std::vector<int> type_id;
	std::vector<Vector3> position;
	std::vector<Matrix3> orientation;
	std::vector<Vector3> momentum;
	std::vector<Vector3> angular_momentum;
	std::vector<Vector3> force;
	std::vector<Vector3> torque;

	size_t size() const {
		return global_id.size();
	}

	void resize(size_t n) {
		global_id.resize(n);
		type_id.resize(n);
		position.resize(n);
		orientation.resize(n);
		momentum.resize(n);
		angular_momentum.resize(n);
		force.resize(n);
		torque.resize(n);
	}

	void clear() {
		global_id.clear();
		type_id.clear();
		position.clear();
		orientation.clear();
		momentum.clear();
		angular_momentum.clear();
		force.clear();
		torque.clear();
	}

	void reserve(size_t n) {
		global_id.reserve(n);
		type_id.reserve(n);
		position.reserve(n);
		orientation.reserve(n);
		momentum.reserve(n);
		angular_momentum.reserve(n);
		force.reserve(n);
		torque.reserve(n);
	}

	void push_back(const RigidBodyIO& rb) {
		global_id.push_back(rb.id);
		type_id.push_back(rb.type_id);
		position.push_back(rb.position);
		orientation.push_back(rb.orientation);
		momentum.push_back(rb.momentum);
		angular_momentum.push_back(rb.angular_momentum);
		force.push_back(rb.force);
		torque.push_back(rb.torque);
	}

	void push_back(const HostRigidBodyData& rb, size_t idx) {
		global_id.push_back(rb.global_id[idx]);
		type_id.push_back(rb.type_id[idx]);
		position.push_back(rb.position[idx]);
		orientation.push_back(rb.orientation[idx]);
		momentum.push_back(rb.momentum[idx]);
		angular_momentum.push_back(rb.angular_momentum[idx]);
		force.push_back(rb.force[idx]);
		torque.push_back(rb.torque[idx]);
	}

	void remove_swap(size_t idx) {
		if (idx >= size())
			return;

		const size_t last = size() - 1;
		if (idx < last) {
			std::swap(global_id[idx], global_id[last]);
			std::swap(type_id[idx], type_id[last]);
			std::swap(position[idx], position[last]);
			std::swap(orientation[idx], orientation[last]);
			std::swap(momentum[idx], momentum[last]);
			std::swap(angular_momentum[idx], angular_momentum[last]);
			std::swap(force[idx], force[last]);
			std::swap(torque[idx], torque[last]);
		}

		global_id.pop_back();
		type_id.pop_back();
		position.pop_back();
		orientation.pop_back();
		momentum.pop_back();
		angular_momentum.pop_back();
		force.pop_back();
		torque.pop_back();
	}
};

} // namespace ARBD
