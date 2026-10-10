// RBEnergyKernels.h (2026)
#pragma once
#include "Interactions/Nonbonded/RigidBodyAttachedParticles.h"
#include "Objects/DeviceParticle.h"
#include "Objects/DeviceRigidBody.h"
#include "Types/Types.h"
#include "Types/Vector3.h"

namespace MARS {
	
using RB_Energy= Vector3; ///< x: translational KE, y: rotational KE, z: grid potential, t: attached-particle potential
/**
 * @brief Reduce rigid-body energies into one RB_Energy at `out`.
 * - x: translational KE, 0.5 p^2/m (raw momentum units)
 * - y: rotational KE, 0.5 sum L_k^2/I_k, body frame (raw units)
 * - z: grid potential, sum rb.force[].t
 * - t: attached-particle potential, sum ForceEnergy.t
 *
 * Thread i handles body i and attached particle i. Caller zeroes `out`.
 * @note Shared memory: block_size * sizeof(RB_Energy).
 */
struct RBEnergyReduceKernel {
	ConstRigidBodyView rb;
	RigidBodyTypeView types;
	ConstParticleView particles;
	const RBAttachedParticle* __restrict__ attached;
	RB_Energy* __restrict__ out;
	idx_t num_rb;
	idx_t num_attached;
	idx_t block_size;

	template<typename WorkItemT>
	KERNEL_FUNC void operator()(size_t, WorkItemT& item) const {
		RB_Energy* acc = item.template get_shared_mem<RB_Energy>(0);

		const idx_t tid = item.local_id();
		const idx_t i = item.global_id();

		RB_Energy e(0.0f);
		if (i < num_rb) {
			const int type = rb.type_id[i];
			const Vector3 inertia = types.inertia[type];
			const Vector3 L = rb.angular_momentum[i];
			e.x = 0.5f * rb.momentum[i].length2() / types.mass[type];
			e.y = 0.5f * (L.x * L.x / inertia.x + L.y * L.y / inertia.y + L.z * L.z / inertia.z);
			e.z = rb.force[i].t;
		}
		if (i < num_attached) {
			e.t = particles.ForceEnergy[attached[i].particle_index].t;
		}
		acc[tid] = e;

		item.barrier();
		for (idx_t offset = block_size / 2; offset > 0; offset >>= 1) {
			if (tid < offset) {
				acc[tid].accumulate(acc[tid + offset]);
			}
			item.barrier();
		}

		if (tid == 0) {
			atomic_add(out, acc[0]);
		}
	}
};

} // namespace MARS

#ifdef USE_CUDA
#include "Backend/CUDA/KernelHelper.cuh"
namespace MARS {
extern template Event launch_cuda_kernel_with_workitem(const Resource& resource,
													   const KernelConfig& config,
													   RBEnergyReduceKernel kernel_func);
} // namespace MARS
#endif

#ifdef USE_SYCL
#include <sycl/sycl.hpp>
template<>
struct sycl::is_device_copyable<MARS::RBEnergyReduceKernel> : std::true_type {};
#endif
