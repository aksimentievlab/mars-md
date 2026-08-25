// RigidBodyCosmeticsKernel.h (2026)
// Placement of a rigid body's visualization-only template atoms.
//
// A template PDB/PSF describes a body's full shape, but only the atoms marked
// with the attached segname become real particles (see
// Interactions/Nonbonded/RigidBodyAttachedParticles.h). The rest carry no
// physics and have no slot in the particle array - they exist so the body reads
// as a molecule in VMD rather than a bare centroid. They are placed by
// transforming the body every output step, never integrated.
//
// This is output-only work, so it belongs on StreamType::Optional: it can
// overlap the next step's physics instead of serializing behind it.
#pragma once
#include "Objects/DeviceRigidBody.h"

namespace MARS {

/**
 * @brief A template atom that exists only to be drawn.
 *
 * Deliberately not an RBAttachedParticle: an attached particle is a real
 * particle with a slot in the particle array, resynced and force-reduced every
 * step. A cosmetic atom has no particle index, no forces, and is placed only
 * when a frame is written - so it stores nothing but where it sits on its body.
 */
struct RBCosmeticParticleView {
	DEVICE_PTR(Vector3) __restrict__ body_offset; ///< Position in the parent body's frame
	DEVICE_PTR(int) rb_id;			 ///< Parent rigid-body instance
};

/**
 * @brief Write cosmetic template atoms' lab-frame positions to a flat buffer.
 *
 * `out[i] = orientation[rb_id] * body_offset + position[rb_id]`, one thread per
 * cosmetic atom. Output goes to a standalone buffer the trajectory writer
 * drains, never into the particle array.
 *
 * Runs once per output period, not once per step.
 */
struct RBCosmeticPositionsKernel {
	ConstRigidBodyView rb;
	RBCosmeticParticleView cosmetic;
	Vector3* __restrict__ out;
	idx_t num_cosmetic;

	KERNEL_FUNC void operator()(idx_t idx) const {
		if (idx >= num_cosmetic)
			return;
		const int id = cosmetic.rb_id[idx];
		out[idx] = rb.orientation[id] * cosmetic.body_offset[idx] + rb.position[id];
	}
};

} // namespace MARS

#ifdef USE_CUDA
#include "Backend/CUDA/KernelHelper.cuh"
namespace MARS {
extern template Event launch_cuda_kernel(const Resource& resource,
										 const KernelConfig& config,
										 RBCosmeticPositionsKernel kernel_func);
} // namespace MARS
#endif

#ifdef USE_SYCL
#include <sycl/sycl.hpp>
template<>
struct sycl::is_device_copyable<MARS::RBCosmeticPositionsKernel> : std::true_type {};
#endif
