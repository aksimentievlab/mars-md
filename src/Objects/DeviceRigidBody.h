#pragma once
#include "Header.h"
#include "Types/GridTerm.h"
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
	DEVICE_PTR(Vector3) __restrict__ external_force;
	DEVICE_PTR(Vector3) __restrict__ external_torque;

	/// DLM body-frame rotation step. Caller owns re-orthonormalization.
	HOST DEVICE void apply_body_frame_rotation(idx_t idx, const Matrix3& R) const {
		angular_momentum[idx] = R * angular_momentum[idx];
		orientation[idx] = orientation[idx] * R;
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
	CONSTANT_PTR(Vector3) __restrict__ external_force;
	CONSTANT_PTR(Vector3) __restrict__ external_torque;
};

// Grid lists are variable length, so each is offset+count. See dev_notes.md.
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
	// pointer (not per-type), laid out per type as [potential terms][density
	// terms][pmf terms] contiguously. One GridTerm (grid_id + scale +
	// scale_slope + boundary_condition) per grid, matching
	// ParticleTypeView::pmf_grid_terms - a single load per grid instead of
	// combining parallel id/scale arrays.
	CONSTANT_PTR(GridTerm) __restrict__ grid_terms;
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
