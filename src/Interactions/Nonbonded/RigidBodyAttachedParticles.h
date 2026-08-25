// RigidBodyAttachedParticles.h (2026)
// Attached particles: real particles rigidly slaved to a parent rigid body.
//
// They live in the ordinary particle array, so every existing force path
// (pairlist, nonbonded, bonded) sees them with no special casing - matching
// legacy MARS, which counts them in every `num + num_rb_attached_particles`
// kernel bound. Only two things differ, and both live here:
//
//   1. Their position is not integrated. It is rewritten each step from the
//      parent body's transform (RBSyncAttachedPositionsKernel); the
//      integrators skip them via ParticleFlags::FLAG_RB_ATTACHED.
//   2. The force they accumulate does not move them. It is reduced into the
//      parent body's net force and torque (RBReduceAttachedForcesKernel),
//      which is what actually responds.
//
// Ports of legacy update_particle_positions_kernel and
// RigidBody::apply_attached_particle_forces (mars1 RigidBody.cu:108, :336).
#pragma once
#include "Objects/DeviceParticle.h"
#include "Objects/DeviceRigidBody.h"

namespace MARS {

/**
 * @brief One attached particle's binding to its parent rigid body.
 *
 * Built once on the host by RigidBodyManager::prepare_attached_particles() and
 * static thereafter: which particle belongs to which body, and where on that
 * body it sits, never change (bond breaking/formation is a separate, future
 * feature).
 */
struct RBAttachedParticle {
	Vector3 body_offset; ///< Position in the parent body's frame
	int particle_index;	 ///< Index into the owning patch's particle arrays
	int rb_id;			 ///< Parent rigid-body instance
};

/**
 * @brief Rewrite every attached particle's position from its parent body.
 *
 * `pos = orientation * body_offset + position`, one thread per attached
 * particle. Must run before the step's force calculation, since the pairlist
 * and every force kernel read these positions.
 *
 * Momentum is deliberately left untouched: an attached particle has no
 * independent momentum (the parent body carries it), and nothing reads the
 * particle's own - the integrators skip it entirely.
 */
struct RBSyncAttachedPositionsKernel {
	ConstRigidBodyView rb;
	ParticleView particles;
	const RBAttachedParticle* __restrict__ attached;
	idx_t num_attached;

	KERNEL_FUNC void operator()(idx_t idx) const {
		if (idx >= num_attached)
			return;
		const RBAttachedParticle a = attached[idx];
		particles.pos[a.particle_index] =
			rb.orientation[a.rb_id] * a.body_offset + rb.position[a.rb_id];
	}
};

/**
 * @brief Reduce attached-particle forces into their parent bodies.
 *
 * One block per rigid-body instance that has attached particles, striding that
 * instance's contiguous range: `force += F_i` and
 * `torque += (orientation * body_offset_i) x F_i`, block-reduced in shared
 * memory then atomically added into the body. Direct port of legacy
 * apply_attached_particle_forces, which sums exactly these two quantities.
 *
 * The torque arm is the lab-frame offset from the body's own origin, so the
 * body's `position` must be the same physical point its grids are built about
 * (see the `referencePoint` config key).
 *
 * Energy is not reduced: legacy's apply_attached_particle_forces takes only
 * forces, and RigidBodyView has no energy accumulator (same reason
 * RBParticleGridForceKernel drops its energy term on the RB side). The
 * particle keeps its own energy, which the existing output already reports.
 *
 * Must run after all particle forces are complete (nonbonded *and* bonded) and
 * before the rigid-body integration that consumes force/torque.
 *
 * The particle's own force is left in place rather than zeroed: nothing reads
 * it afterwards (the integrators skip these particles, and the next step's
 * nonbonded pass clears the whole force array), and leaving it keeps the
 * force/energy output honest about what actually acted on the particle.
 */
struct RBReduceAttachedForcesKernel {
	RigidBodyView rb;
	ConstParticleView particles;
	const RBAttachedParticle* __restrict__ attached;
	/// Per-block half-open range into `attached`.
	const int* __restrict__ range_start;
	const int* __restrict__ range_count;
	/// Rigid-body instance each block reduces for. Blocks are emitted only for
	/// bodies that actually have attached particles, so this is not the identity.
	const int* __restrict__ block_rb_id;
	idx_t block_size;

	template<typename WorkItemT>
	KERNEL_FUNC void operator()(size_t, WorkItemT& item) const {
		// Shared scratch for the block reduction - named apart from rb.force /
		// rb.torque, which are the per-body totals these finally fold into.
		Vector3* f_shared = item.template get_shared_mem<Vector3>(0);
		Vector3* t_shared = item.template get_shared_mem<Vector3>(block_size * sizeof(Vector3));

		const idx_t block_id = item.group_id();
		const idx_t tid = item.local_id();
		const int rb_id = block_rb_id[block_id];
		const int start = range_start[block_id];
		const int count = range_count[block_id];

		const Matrix3 orientation = rb.orientation[rb_id];

		Vector3 f_acc(0.0f);
		Vector3 t_acc(0.0f);
		for (idx_t i = tid; i < static_cast<idx_t>(count); i += block_size) {
			const RBAttachedParticle a = attached[start + i];
			// ForceEnergy packs energy into Vector3's 4th component, which
			// operator+ would otherwise accumulate into the force total - take
			// the vector part only.
			const Vector3 fe = particles.ForceEnergy[a.particle_index];
			const Vector3 f(fe.x, fe.y, fe.z);
			f_acc += f;
			t_acc += (orientation * a.body_offset).cross(f);
		}
		f_shared[tid] = f_acc;
		t_shared[tid] = t_acc;

		// Tree reduction leaves the block's total in slot 0.
		item.barrier();
		for (idx_t offset = block_size / 2; offset > 0; offset >>= 1) {
			if (tid < offset) {
				f_shared[tid] += f_shared[tid + offset];
				t_shared[tid] += t_shared[tid + offset];
			}
			item.barrier();
		}

		if (tid == 0) {
			atomic_add(&rb.force[rb_id], f_shared[0]);
			atomic_add(&rb.torque[rb_id], t_shared[0]);
		}
	}
};

} // namespace MARS

#ifdef USE_CUDA
#include "Backend/CUDA/KernelHelper.cuh"
namespace MARS {
extern template Event launch_cuda_kernel(const Resource& resource,
										 const KernelConfig& config,
										 RBSyncAttachedPositionsKernel kernel_func);
extern template Event launch_cuda_kernel_with_workitem(const Resource& resource,
													   const KernelConfig& config,
													   RBReduceAttachedForcesKernel kernel_func);
} // namespace MARS
#endif

#ifdef USE_SYCL
#include <sycl/sycl.hpp>
template<>
struct sycl::is_device_copyable<MARS::RBSyncAttachedPositionsKernel> : std::true_type {};
template<>
struct sycl::is_device_copyable<MARS::RBReduceAttachedForcesKernel> : std::true_type {};
#endif
