// RigidBodyParticleGridBatch.h (2026)
// Phase 4.3 of the rigid-body suite: batched particle-RB grid dispatch.
// Different loop shape than 4.1's grid-grid batching (RigidBodyGridBatch.h):
// iterate particles, each sampling one RB potential grid - no pairing
// constraint (one grid in play), so no format-uniformity check is needed
// and every particle can, in principle, sample every (RB, potential-grid)
// candidate.
#pragma once
#include "Objects/DeviceParticle.h"
#include "Objects/DeviceRigidBody.h"
#include "Types/BaseGridDevice.h"

namespace ARBD {

/**
 * @brief One (RB instance, potential-grid) task for the batched
 *        particle-RB force kernel, built on-device by
 *        RBParticleGridBuildKernel.
 */
struct RBParticleGridWork {
	Matrix3 basis_inv;	// (R_rb * grid.basis).inverse() - lab-frame, world->grid-local
	Vector3 origin_lab; // R_rb * grid.origin + rb.position
	float scale;		// GridTerm::scale for this candidate's grid term
	int grid_id;
	int rb_id;
	int scheme;
};

/**
 * @brief Recompute per-candidate lab-frame transforms for the static
 *        (rb_id, grid_id) candidate list built once by
 *        RigidBodyManager::prepare_particle_grid_dispatch().
 *
 * Unlike RBGridCullKernel (RigidBodyGridBatch.h), there is no cull here:
 * which (RB, potential-grid) pairs exist doesn't depend on position - "no
 * pairing constraint" (todo.md Phase 4.3) - only the transform does, so
 * every candidate always becomes exactly one work item, index for index; no
 * atomic counter, capacity, or overflow flag needed. Particles outside a
 * grid's support naturally sample as zero (dense grids zero-pad
 * out-of-bounds - Types/BaseGridDevice.h), so looping every particle per
 * candidate is correct without a broad-phase distance cutoff; whether
 * that's fast enough at scale is a Phase 6 profiling question, not a
 * correctness one.
 */
struct RBParticleGridBuildKernel {
	ConstRigidBodyView rb;
	const int* __restrict__ candidate_rb_id;
	const int* __restrict__ candidate_grid_id;
	const float* __restrict__ candidate_scale;
	idx_t num_candidates;
	const BaseGridView<arbd_real>* __restrict__ grid_views;
	int scheme;

	RBParticleGridWork* __restrict__ work_out;

	KERNEL_FUNC void operator()(idx_t idx) const {
		if (idx >= num_candidates)
			return;

		const int rb_id = candidate_rb_id[idx];
		const int grid_id = candidate_grid_id[idx];
		const Matrix3 R = rb.orientation[rb_id];
		const BaseGridView<arbd_real> grid = grid_views[grid_id];

		RBParticleGridWork w;
		w.basis_inv = (R * grid.basis).inverse();
		w.origin_lab = R * grid.origin + rb.position[rb_id];
		w.scale = candidate_scale[idx];
		w.grid_id = grid_id;
		w.rb_id = rb_id;
		w.scheme = scheme;
		work_out[idx] = w;
	}
};

/**
 * @brief Batched particle-RB force kernel: gridDim.x == num_candidates *
 *        blocks_per_candidate exactly (no over-provisioning needed, unlike
 *        the grid-grid batched kernel - since every candidate here shares
 *        the same num_particles, block counts are uniform across
 *        candidates, so block-to-work-item mapping is plain div/mod instead
 *        of a prefix-sum + binary search).
 *
 * Force on each particle is scattered atomically into
 * ParticleView::ForceEnergy (matching Pmf.h's force+energy packing,
 * force = -scale*gradient, energy = scale*value) - unlike Pmf.h's
 * single-pass one-thread-per-particle kernel, multiple work items (distinct
 * RB/grid candidates) can touch the same particle here, so the write can't
 * be a plain read-modify-write. The RB's Newton's-third-law reaction
 * (force = -force_on_particle, torque about the grid's own lab-frame
 * origin, i.e. R_rb*grid.origin+rb.position - matching
 * RigidBodyGridBatch.h's torque reference-point convention) is block-reduced
 * then atomically added into RigidBodyView's force/torque. No per-RB energy
 * accumulator exists yet (RigidBodyGridBatch.h), so that component is
 * dropped on the RB side but kept on the particle side.
 */
struct RBParticleGridForceKernel {
	RigidBodyView rb;
	ParticleView particles;
	const RBParticleGridWork* __restrict__ work;
	const BaseGridView<arbd_real>* __restrict__ grid_views;
	idx_t num_particles;
	idx_t blocks_per_candidate;
	idx_t block_size;

	template<typename WorkItemT>
	KERNEL_FUNC void operator()(size_t, WorkItemT& item) const {
		Vector3* force = item.template get_shared_mem<Vector3>(0);
		Vector3* torque = item.template get_shared_mem<Vector3>(block_size * sizeof(Vector3));

		const idx_t block_id = item.group_id();
		const idx_t tid = item.local_id();
		const idx_t item_idx = block_id / blocks_per_candidate;
		const idx_t slice = block_id % blocks_per_candidate;
		const RBParticleGridWork w = work[item_idx];
		const BaseGridView<arbd_real> grid = grid_views[w.grid_id];

		Vector3 f_acc(0.0f);
		Vector3 t_acc(0.0f);
		const idx_t stride = block_size * blocks_per_candidate;
		for (idx_t p = slice * block_size + tid; p < num_particles; p += stride) {
			const Vector3 pos = particles.pos[p];
			const Vector3 local = w.basis_inv.transform(pos - w.origin_lab);
			const Matrix3 identity(1.0f);
			const GridSample<arbd_real> sample = (w.scheme == 0)
													 ? sample_grid_linear(grid.data,
																		  local,
																		  Vector3(0.0f),
																		  identity,
																		  identity,
																		  grid.dimensions,
																		  grid.boundary_condition)
													 : sample_grid_cubic(grid.data,
																		 local,
																		 Vector3(0.0f),
																		 identity,
																		 identity,
																		 grid.dimensions,
																		 grid.boundary_condition);

			const Vector3 force_lab =
				w.basis_inv.transpose().transform(sample.gradient * (-w.scale));
			Vector3 fe = force_lab;
			fe.t = w.scale * sample.value;
			atomic_add(&particles.ForceEnergy[p], fe);

			f_acc.x -= force_lab.x;
			f_acc.y -= force_lab.y;
			f_acc.z -= force_lab.z;
			t_acc += (pos - w.origin_lab).cross(-force_lab);
		}
		force[tid] = f_acc;
		torque[tid] = t_acc;

		item.barrier();
		for (idx_t offset = block_size / 2; offset > 0; offset >>= 1) {
			if (tid < offset) {
				force[tid] += force[tid + offset];
				torque[tid] += torque[tid + offset];
			}
			item.barrier();
		}

		if (tid == 0) {
			atomic_add(&rb.force[w.rb_id], force[0]);
			atomic_add(&rb.torque[w.rb_id], torque[0]);
		}
	}
};

} // namespace ARBD

#ifdef USE_CUDA
#include "Backend/CUDA/KernelHelper.cuh"
namespace ARBD {
extern template Event launch_cuda_kernel(const Resource& resource,
										 const KernelConfig& config,
										 RBParticleGridBuildKernel kernel_func);
extern template Event launch_cuda_kernel_with_workitem(const Resource& resource,
													   const KernelConfig& config,
													   RBParticleGridForceKernel kernel_func);
} // namespace ARBD
#endif

#ifdef USE_SYCL
#include <sycl/sycl.hpp>
template<>
struct sycl::is_device_copyable<ARBD::RBParticleGridBuildKernel> : std::true_type {};
template<>
struct sycl::is_device_copyable<ARBD::RBParticleGridForceKernel> : std::true_type {};
#endif
