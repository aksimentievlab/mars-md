#pragma once

#include "../Random/philox.h"
#include "Constants.h"
#include "Header.h"
#include "Objects/DeviceRigidBody.h"
#include "System/PeriodicBox.h"
#include "Types/Matrix3.h"
#include "Types/Types.h"

namespace ARBD {

/**
 * @brief Port of legacy RigidBody::integrate onto SoA RigidBodyView.
 *
 * Overdamped Brownian rigid-body dynamics: one kernel, one full step, no
 * momentum. Selected by rigidBodyDynamicType values other than "Langevin",
 * which is legacy's default.
 * @see RBBD.md
 */
template<typename TemperatureType = float>
struct RBIntegrateBDKernel {
	RigidBodyView rb;
	RigidBodyTypeView types;
	PeriodicBox sim_box;
	float timestep;
	TemperatureType kT;
	idx_t num_rb;
	uint64_t base_seed;
	size_t current_step;

	/// Philox ctr1, distinct from openrand's 0x12345 default so this kernel's
	/// stream cannot coincide with another kernel's at the same index.
	static constexpr uint32_t rng_stream = 0x52424244u;

	RBIntegrateBDKernel(RigidBodyView rb_view,
						RigidBodyTypeView type_view,
						const PeriodicBox& box,
						float dt,
						TemperatureType temp,
						idx_t n,
						uint64_t seed,
						size_t step)
		: rb(rb_view), types(type_view), sim_box(box), timestep(dt), kT(temp), num_rb(n),
		  base_seed(seed), current_step(step) {}

	KERNEL_FUNC void operator()(idx_t idx) const {
		if (idx >= num_rb)
			return;

		const int type = rb.type_id[idx];
		const float mass = types.mass[type];
		const Vector3 inertia = types.inertia[type];
		const Matrix3 orientation = rb.orientation[idx];

		// Legacy scales these once in RigidBodyType::setDampingCoeffs; arbd2
		// stores them unscaled, so the factor is applied here. No
		// langevin_damp_scale: that literal 10000 belongs to addLangevin's drag
		// term, which this path never runs.
		const Vector3 trans_damping = types.trans_damping[type] * constants::langevin_damping_unit;
		const Vector3 rot_damping = types.rot_damping[type] * constants::langevin_damping_unit;

		// See RBLangevinForceKernel for why the log argument is clamped.
		openrand::Philox rng(base_seed + current_step,
							 static_cast<uint32_t>(idx),
							 openrand::DEFAULT_GLOBAL_SEED,
							 rng_stream);
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

		// Legacy's `diffusion / Temp`, formed directly to avoid a scalar-over-
		// vector division.
		const Vector3 trans_mobility(1.0f / (trans_damping.x * mass),
									 1.0f / (trans_damping.y * mass),
									 1.0f / (trans_damping.z * mass));
		const Vector3 rot_mobility(1.0f / (rot_damping.x * inertia.x),
								   1.0f / (rot_damping.y * inertia.y),
								   1.0f / (rot_damping.z * inertia.z));
		const Vector3 diffusion = trans_mobility * kT;
		const Vector3 rot_diffusion = rot_mobility * kT;

		const Vector3 force = rb.force[idx] + rb.external_force[idx];
		const Vector3 torque = rb.torque[idx] + rb.external_torque[idx];

		const Vector3 offset =
			Vector3::element_mult(trans_mobility, orientation.transpose() * force) * timestep +
			Vector3::element_mult(Vector3::element_sqrt(2.0f * diffusion * timestep), w1);

		rb.position[idx] = sim_box.wrap(rb.position[idx] + orientation * offset);

		const Vector3 rotation_offset =
			Vector3::element_mult(rot_mobility, orientation.transpose() * torque) * timestep +
			Vector3::element_mult(Vector3::element_sqrt(2.0f * rot_diffusion * timestep), w2);

		const Matrix3 rotation = rotation_matrix_z(rotation_offset.z * 0.5f) *
								 rotation_matrix_y(rotation_offset.y * 0.5f) *
								 rotation_matrix_x(rotation_offset.x) *
								 rotation_matrix_y(rotation_offset.y * 0.5f) *
								 rotation_matrix_z(rotation_offset.z * 0.5f);
		rb.orientation[idx] = normalize_orientation(orientation * rotation);
	}
};

} // namespace ARBD

#ifdef USE_CUDA
namespace ARBD {
extern template struct RBIntegrateBDKernel<float>;
extern template Event launch_cuda_kernel(const Resource& resource,
										 const KernelConfig& config,
										 RBIntegrateBDKernel<float> kernel_func);
} // namespace ARBD
#endif

#ifdef USE_SYCL
#include <sycl/sycl.hpp>
template<>
struct sycl::is_device_copyable<ARBD::RBIntegrateBDKernel<float>> : std::true_type {};
#endif
