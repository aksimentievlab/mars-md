#pragma once
#include "Backend/Events.h"
#include "Backend/KernelConfig.h"
#include "Backend/Kernels.h"
#include "Backend/Resource.h"
#include "Header.h"
#include "Objects/DeviceRigidBody.h"
#include "Types/Types.h"

namespace MARS {
/**
 * @brief
 * @todo Needs to accumulate ALL froces on HOST first
 */
struct ExternalForceView {
	CONSTANT_PTR(int) __restrict__ id;
	CONSTANT_PTR(Vector3) __restrict__ external_force;
	CONSTANT_PTR(Vector3) __restrict__ external_torque;
};

/**
 * @brief
 */
struct ApplyExternalForcesKernel {
	RigidBodyView rb;
	ExternalForceView external_force_view;
	idx_t num_rb;
	ApplyExternalForcesKernel(RigidBodyView target_rb,
							  ExternalForceView src_external_force,
							  idx_t n)
		: rb(target_rb), external_force_view(src_external_force), num_rb(n) {}

	KERNEL_FUNC void operator()(idx_t idx) const {
		if (idx >= num_rb)
			return;
		const int id = external_force_view.id[idx];
		const Vector3 external_force = external_force_view.external_force[idx];
		const Vector3 external_torque = external_force_view.external_torque[idx];
		rb.external_force[id] = external_force;
		rb.external_torque[id] = external_torque;
	}
};

} // namespace MARS
#ifdef USE_CUDA
namespace MARS {
extern template Event launch_cuda_kernel(const Resource& resource,
										 const KernelConfig& config,
										 ApplyExternalForcesKernel kernel_func);
} // namespace MARS
#endif

#ifdef USE_SYCL
#include <sycl/sycl.hpp>
template<>
struct sycl::is_device_copyable<MARS::ApplyExternalForcesKernel> : std::true_type {};
#endif
