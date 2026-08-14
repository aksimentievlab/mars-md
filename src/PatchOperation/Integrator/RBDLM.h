#pragma once

#include "../Random/philox.h"
#include "Constants.h"
#include "Header.h"
#include "Objects/DeviceRigidBody.h"
#include "System/PeriodicBox.h"
#include "Types/Matrix3.h"
#include "Types/Types.h"

namespace ARBD {

// Legacy RigidBody unit conversions (v1 RigidBody.cu).
/**
 * @brief Port of legacy RigidBody::addLangevin onto SoA RigidBodyView.
 */
template<typename TemperatureType = float>
struct RBLangevinForceKernel {
	RigidBodyView rb;
	RigidBodyTypeView types;
	float timestep;
	TemperatureType kT;
	idx_t num_rb;
	uint64_t base_seed;
	size_t current_step;

	RBLangevinForceKernel(RigidBodyView rb_view,
						  RigidBodyTypeView type_view,
						  float dt,
						  TemperatureType temp,
						  idx_t n,
						  uint64_t seed,
						  size_t step)
		: rb(rb_view), types(type_view), timestep(dt), kT(temp), num_rb(n), base_seed(seed),
		  current_step(step) {}

	KERNEL_FUNC void operator()(idx_t idx) const {
		if (idx >= num_rb)
			return;

		const int type = rb.type_id[idx];
		const float mass = types.mass[type];
		const Vector3 inertia = types.inertia[type];
		const Vector3 trans_damping = types.trans_damping[type];
		const Vector3 rot_damping = types.rot_damping[type];
		const Matrix3 orientation = rb.orientation[idx];

		//
		// draw_float4() returns values in [0,1); log(0) is -inf, which would
		// make the whole trajectory NaN, so the radius argument is clamped away
		// from zero. The bias is ~1 ulp and only on a ~2^-32 tail.
		openrand::Philox rng(base_seed + current_step, static_cast<uint32_t>(idx));
		auto gaussian_pair = [](float u_r, float u_theta, float& a, float& b) {
			const float r = sqrtf(-2.0f * logf(fmaxf(u_r, 1e-20f)));
			const float theta = 2.0f * 3.1415926535f * u_theta;
			a = r * cosf(theta);
			b = r * sinf(theta);
		};
		const openrand::float4 u0 = rng.draw_float4();
		const openrand::float4 u1 = rng.draw_float4();
		float g[8];
		gaussian_pair(u0.x, u0.y, g[0], g[1]);
		gaussian_pair(u0.z, u0.w, g[2], g[3]);
		gaussian_pair(u1.x, u1.y, g[4], g[5]);
		gaussian_pair(u1.z, u1.w, g[6], g[7]);
		const Vector3 w1(g[0], g[1], g[2]);
		const Vector3 w2(g[3], g[4], g[5]);

		// kT is legacy's `Temp` exactly - RigidBody.cu:28 computes it as
		// temperature * 0.0019872065, i.e. kT in kcal/mol, not Kelvin.
		const Vector3 trans_damp = trans_damping * constants::langevin_damping_unit;
		const Vector3 rot_damp = rot_damping * constants::langevin_damping_unit;

		const Vector3 trans_coeff = Vector3::element_sqrt(2.0f * kT * mass * trans_damp / timestep);
		const Vector3 rot_coeff =
			Vector3::element_sqrt(2.0f * kT * Vector3::element_mult(inertia, rot_damp) / timestep);

		Vector3 f = Vector3::element_mult(trans_coeff, w1) -
					Vector3::element_mult(trans_damp, orientation.transpose() * rb.momentum[idx]) *
						constants::langevin_damp_scale;
		Vector3 torq = Vector3::element_mult(rot_coeff, w2) -
					   Vector3::element_mult(rot_damp, rb.angular_momentum[idx]) *
						   constants::langevin_damp_scale;

		f = orientation * f;
		torq = orientation * torq;

		rb.force[idx] += f;
		rb.torque[idx] += torq;
	}
};

/**
 * @brief Port of legacy RigidBody::integrateDLM onto SoA RigidBodyView.
 * @param substep 0 or 2: half-step momenta; 1: full-step positions/orientations.
 */
struct RBIntegrateDLMKernel {
	RigidBodyView rb;
	RigidBodyTypeView types;
	PeriodicBox sim_box;
	float timestep;
	idx_t num_rb;
	int substep;

	RBIntegrateDLMKernel(RigidBodyView rb_view,
						 RigidBodyTypeView type_view,
						 const PeriodicBox& box,
						 float dt,
						 idx_t n,
						 int integration_substep)
		: rb(rb_view), types(type_view), sim_box(box), timestep(dt), num_rb(n),
		  substep(integration_substep) {}

	KERNEL_FUNC void operator()(idx_t idx) const {
		if (idx >= num_rb)
			return;

		const int type = rb.type_id[idx];
		const float mass = types.mass[type];
		const Vector3 inertia = types.inertia[type];

		if (substep == 0 || substep == 2) {
			const Vector3 force = rb.force[idx] + rb.external_force[idx];
			const Vector3 torque = rb.torque[idx] + rb.external_torque[idx];
			rb.momentum[idx] += 0.5f * timestep * force * constants::impulse_to_momentum;
			rb.angular_momentum[idx] += 0.5f * timestep *
										(rb.orientation[idx].transpose() * torque) *
										constants::impulse_to_momentum;
			return;
		}

		rb.position[idx] += timestep * rb.momentum[idx] / mass * constants::velocity_scale;
		rb.position[idx] = sim_box.wrap(rb.position[idx]);

		Matrix3 R = rotation_matrix_x(0.5f * timestep * rb.angular_momentum[idx].x / inertia.x *
									  constants::velocity_scale);
		rb.apply_body_frame_rotation(idx, R);

		R = rotation_matrix_y(0.5f * timestep * rb.angular_momentum[idx].y / inertia.y *
							  constants::velocity_scale);
		rb.apply_body_frame_rotation(idx, R);

		R = rotation_matrix_z(timestep * rb.angular_momentum[idx].z / inertia.z *
							  constants::velocity_scale);
		rb.apply_body_frame_rotation(idx, R);

		R = rotation_matrix_y(0.5f * timestep * rb.angular_momentum[idx].y / inertia.y *
							  constants::velocity_scale);
		rb.apply_body_frame_rotation(idx, R);

		R = rotation_matrix_x(0.5f * timestep * rb.angular_momentum[idx].x / inertia.x *
							  constants::velocity_scale);
		rb.apply_body_frame_rotation(idx, R);
	}
};

} // namespace ARBD

#ifdef USE_CUDA
namespace ARBD {
extern template struct RBLangevinForceKernel<float>;
extern template Event launch_cuda_kernel(const Resource& resource,
										 const KernelConfig& config,
										 RBLangevinForceKernel<float> kernel_func);
extern template Event launch_cuda_kernel(const Resource& resource,
										 const KernelConfig& config,
										 RBIntegrateDLMKernel kernel_func);
} // namespace ARBD
#endif

#ifdef USE_SYCL
#include <sycl/sycl.hpp>
template<>
struct sycl::is_device_copyable<ARBD::RBLangevinForceKernel<float>> : std::true_type {};
template<>
struct sycl::is_device_copyable<ARBD::RBIntegrateDLMKernel> : std::true_type {};
#endif
