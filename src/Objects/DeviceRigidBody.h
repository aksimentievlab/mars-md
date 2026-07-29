#pragma once
#include "Header.h"
#include "Types/Types.h"
#include "Types/Vector3.h"
namespace ARBD {
struct RigidBodyView {
	DEVICE_PTR(int) __restrict__ id;
	DEVICE_PTR(int) __restrict__ type_id;
	DEVICE_PTR(Vector3) __restrict__ position;
	DEVICE_PTR(Matrix3) __restrict__ orientation;
	DEVICE_PTR(Vector3) __restrict__ momentum;
	DEVICE_PTR(Vector3) __restrict__ angular_momentum;
	DEVICE_PTR(Vector3) __restrict__ force;
	DEVICE_PTR(Vector3) __restrict__ torque;

	/// DLM body-frame rotation step (legacy RigidBody::applyRotation).
	HOST DEVICE void apply_body_frame_rotation(idx_t idx, const Matrix3& R) const {
		angular_momentum[idx] = R * angular_momentum[idx];
		orientation[idx] = orientation[idx] * R;
		orientation[idx] = normalize_orientation(orientation[idx]);
	}
};

struct ConstRigidBodyView {
	CONSTANT_PTR(int) __restrict__ id;
	CONSTANT_PTR(int) __restrict__ type_id;
	CONSTANT_PTR(Vector3) __restrict__ position;
	CONSTANT_PTR(Matrix3) __restrict__ orientation;
	CONSTANT_PTR(Vector3) __restrict__ momentum;
	CONSTANT_PTR(Vector3) __restrict__ angular_momentum;
	CONSTANT_PTR(Vector3) __restrict__ force;
	CONSTANT_PTR(Vector3) __restrict__ torque;
};

// A type's density/potential/pmf grid lists are variable length (unlike
// ParticleTypeView's fixed int3 force_grid_id), so each list is represented
// as offset+count into a flat int grid-id buffer owned by
// DeviceRigidBodyTypes rather than as a fixed-size field here.
struct alignas(16) RigidBodyTypeView {
	CONSTANT_PTR(float) __restrict__ mass;
	CONSTANT_PTR(Vector3) __restrict__ inertia;
	CONSTANT_PTR(Vector3) __restrict__ trans_damping;
	CONSTANT_PTR(Vector3) __restrict__ rot_damping;
	CONSTANT_PTR(Vector3) __restrict__ trans_force_coeff;
	CONSTANT_PTR(Vector3) __restrict__ rot_torque_coeff;
	CONSTANT_PTR(float) __restrict__ rot_diffusivity;
	CONSTANT_PTR(float) __restrict__ rot_damping_coefficient;
	CONSTANT_PTR(int) __restrict__ potential_grid_offset;
	CONSTANT_PTR(int) __restrict__ potential_grid_count;
	CONSTANT_PTR(int) __restrict__ density_grid_offset;
	CONSTANT_PTR(int) __restrict__ density_grid_count;
	CONSTANT_PTR(int) __restrict__ pmf_grid_offset;
	CONSTANT_PTR(int) __restrict__ pmf_grid_count;
	// Backing storage the offset/count pairs above index into - one shared
	// pointer (not per-type), laid out per type as [potential ids][density
	// ids][pmf ids] contiguously. grid_scales is the parallel per-grid scale
	// factor at the same index (mirrors ParticleType::pmf_scale, generalized
	// to one-per-grid; legacy: potentialGridScale/densityGridScale).
	CONSTANT_PTR(int) __restrict__ grid_ids;
	CONSTANT_PTR(float) __restrict__ grid_scales;
};
} // namespace ARBD

// SYCL device copyable trait
#ifdef USE_SYCL
#include <sycl/sycl.hpp>
template<>
struct sycl::is_device_copyable<ARBD::RigidBodyTypeView> : std::true_type {};
template<>
struct sycl::is_device_copyable<ARBD::RigidBodyView> : std::true_type {};
template<>
struct sycl::is_device_copyable<ARBD::ConstRigidBodyView> : std::true_type {};
#endif
