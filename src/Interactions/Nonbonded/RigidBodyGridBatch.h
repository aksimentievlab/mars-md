// RigidBodyGridBatch.h (2026)
// Phase 4.1 of the rigid-body suite: batched grid-grid dispatch on top of
// Phase 1's per-voxel math (GridGridKernels.h) and Phase 3's type-level pair
// list (RigidBodyForcePairs.h). See todo.md Phase 4.1 for the full design
// rationale (batching vs. per-pair launches, the prefix-sum block mapping).
#pragma once
#include "Interactions/Nonbonded/GridGridKernels.h"
#include "Objects/DeviceRigidBody.h"
#include "Objects/RigidBodyForcePairs.h"
#include "Types/BaseGridDevice.h"

namespace ARBD {

/**
 * @brief One (RB-instance-pair, grid-pair) task for the batched force kernel,
 *        built on-device by RBGridCullKernel.
 *
 * rb_j == -1 marks a type-PMF term (RigidBodyGridPair::is_pmf): the PMF grid
 * is a fixed external field in the lab frame (no orientation/position
 * transform, no reaction body), unlike a true RB-RB pair where rb_j is the
 * potential-owning instance and receives the Newton's-third-law reaction.
 */
struct RBGridWork {
	Matrix3 basis_rho;
	Matrix3 basis_u_inv;
	Vector3 origin_offset; // origin_rho_minus_origin_u, lab frame
	int rho_grid_id;
	int u_grid_id;
	int rb_i;
	int rb_j;
	int scheme;
	idx_t num_blocks;
	idx_t block_offset;
};

/**
 * @brief Cull RB-instance-pair x grid-pair candidates by distance and append
 *        survivors to the worklist, precomputing each item's lab-frame
 *        transform and block count.
 *
 * Candidates are pre-expanded on the host from RigidBodyForcePairList's
 * type-level pairs (see RigidBodyManager::prepare_grid_grid_dispatch) into
 * (rb_a, rb_b) instance pairs - that expansion is static for the run (RB
 * counts/types don't change), so only the per-step distance cull needs to
 * happen here, on-device (a host-side check would force a D2H sync every
 * step - see todo.md Phase 4.1).
 */
struct RBGridCullKernel {
	ConstRigidBodyView rb;
	const int2* __restrict__ candidate_pairs;	// (rb_a, rb_b); rb_b == rb_a for is_pmf candidates
	const int* __restrict__ candidate_pair_idx; // index into grid_pairs
	const RigidBodyGridPair* __restrict__ grid_pairs;
	const BaseGridView<arbd_real>* __restrict__ grid_views; // indexed by grid_id
	idx_t num_candidates;
	float cutoff_squared;
	idx_t step;
	idx_t threads_per_block;
	int scheme;

	RBGridWork* __restrict__ work_out;
	unsigned int* __restrict__ work_count; // atomic counter, caller resets to 0 before launch
	idx_t capacity;
	unsigned int* __restrict__ overflow_flag; // atomic counter, caller resets to 0 before launch

	KERNEL_FUNC void operator()(idx_t idx) const {
		if (idx >= num_candidates)
			return;

		const int2 pair = candidate_pairs[idx];
		const int rb_a = pair.x;
		const int rb_b = pair.y;
		const RigidBodyGridPair& gp = grid_pairs[candidate_pair_idx[idx]];

		if (gp.update_period > 1 && (step % static_cast<size_t>(gp.update_period)) != 0)
			return;

		const Vector3 pos_a = rb.position[rb_a];
		if (!gp.is_pmf) {
			const Vector3 dr = pos_a - rb.position[rb_b];
			if (dr.length2() > cutoff_squared)
				return;
		}

		const Matrix3 R_a = rb.orientation[rb_a];
		const BaseGridView<arbd_real> rho_grid = grid_views[gp.grid_id_rho];
		const BaseGridView<arbd_real> u_grid = grid_views[gp.grid_id_u];

		RBGridWork w;
		w.rho_grid_id = gp.grid_id_rho;
		w.u_grid_id = gp.grid_id_u;
		w.scheme = scheme;
		w.rb_i = rb_a;
		w.basis_rho = R_a * rho_grid.basis;

		if (gp.is_pmf) {
			// External field: fixed in the lab frame, no second body.
			w.basis_u_inv = u_grid.basis_inv;
			w.origin_offset = (R_a * rho_grid.origin + pos_a) - u_grid.origin;
			w.rb_j = -1;
		} else {
			const Matrix3 R_b = rb.orientation[rb_b];
			w.basis_u_inv = (R_b * u_grid.basis).inverse();
			w.origin_offset =
				(R_a * rho_grid.origin + pos_a) - (R_b * u_grid.origin + rb.position[rb_b]);
			w.rb_j = rb_b;
		}

		w.num_blocks = (rho_grid.size() + threads_per_block - 1) / threads_per_block;

		const unsigned int slot = atomic_fetch_add(work_count, 1u);
		if (slot >= static_cast<unsigned int>(capacity)) {
			atomic_fetch_add(overflow_flag, 1u);
			return;
		}
		work_out[slot] = w;
	}
};

/**
 * @brief Exclusive prefix sum of work[].num_blocks -> work[].block_offset,
 *        plus the total block count Kernel B's block-to-work-item search
 *        needs. Single-thread and serial: the worklist is typically
 *        tens-to-low-hundreds of items (see todo.md Phase 4.1 sizing note),
 *        so a parallel scan would add complexity without a measurable win.
 */
struct RBGridPrefixSumKernel {
	RBGridWork* __restrict__ work;
	const unsigned int* __restrict__ work_count;
	unsigned int* __restrict__ total_blocks;

	KERNEL_FUNC void operator()(idx_t idx) const {
		if (idx != 0)
			return;
		idx_t offset = 0;
		const unsigned int n = *work_count;
		for (unsigned int i = 0; i < n; ++i) {
			work[i].block_offset = offset;
			offset += work[i].num_blocks;
		}
		*total_blocks = static_cast<unsigned int>(offset);
	}
};

/**
 * @brief Locate which work item a given block belongs to via binary search
 *        over the (sorted, since it's a prefix sum) block_offset field.
 */
HOST DEVICE inline idx_t
rb_grid_locate_work_item(const RBGridWork* __restrict__ work, unsigned int count, idx_t block_id) {
	idx_t lo = 0;
	idx_t hi = static_cast<idx_t>(count) - 1;
	while (lo < hi) {
		const idx_t mid = lo + (hi - lo + 1) / 2;
		if (work[mid].block_offset <= block_id)
			lo = mid;
		else
			hi = mid - 1;
	}
	return lo;
}

/**
 * @brief Batched grid-grid force kernel: gridDim.x == total_blocks (device-
 *        resident, read without a host round-trip - see
 *        RigidBodyManager::compute_grid_grid_forces for why the launch grid
 *        is sized to the worklist *capacity* instead, with blocks beyond the
 *        real total_blocks doing an early-return no-op).
 *
 * Each work item's rho-grid voxels are striped across its num_blocks blocks
 * (grid-stride within the item), reusing Phase 1's per-voxel math
 * (gridgrid_detail::grid_grid_voxel_force_torque) and block-reduction idiom.
 *
 * Only rb_i's translational force is reduced per-voxel; the per-voxel energy
 * (grid_grid_voxel_force_torque's force.t) is intentionally dropped - there
 * is no per-RB energy accumulator yet (RigidBodyView::force is a plain
 * Vector3, unlike ParticleView::ForceEnergy). rb_j's contribution (when not
 * an external PMF) is the exact Newton's-third-law reaction, computed in
 * closed form from rb_i's totals rather than a second grid pass:
 * force_j = -force_i, torque_j = -(torque_i + origin_offset x force_i) -
 * standard rigid-body torque transfer to a different reference point, valid
 * because the reaction acts at the same world-space point as the action.
 */
struct RBGridBatchedForceKernel {
	RigidBodyView rb;
	const RBGridWork* __restrict__ work;
	const unsigned int* __restrict__ work_count;
	const unsigned int* __restrict__ total_blocks;
	const BaseGridView<arbd_real>* __restrict__ grid_views;
	idx_t block_size;

	template<typename WorkItemT>
	KERNEL_FUNC void operator()(size_t, WorkItemT& item) const {
		Vector3* force = item.template get_shared_mem<Vector3>(0);
		Vector3* torque = item.template get_shared_mem<Vector3>(block_size * sizeof(Vector3));

		const idx_t block_id = item.group_id();
		const idx_t tid = item.local_id();
		const bool active = block_id < static_cast<idx_t>(*total_blocks);

		RBGridWork w{};
		if (active) {
			const idx_t item_idx = rb_grid_locate_work_item(work, *work_count, block_id);
			w = work[item_idx];
		}

		Vector3 f_acc(0.0f);
		Vector3 t_acc(0.0f);
		if (active) {
			const BaseGridView<arbd_real> rho = grid_views[w.rho_grid_id];
			const BaseGridView<arbd_real> u = grid_views[w.u_grid_id];
			const idx_t slice = block_id - w.block_offset;
			const idx_t stride = block_size * w.num_blocks;
			for (idx_t i = slice * block_size + tid; i < rho.size(); i += stride) {
				Vector3 f, t;
				gridgrid_detail::grid_grid_voxel_force_torque(rho,
															  u,
															  w.basis_rho,
															  w.basis_u_inv,
															  w.origin_offset,
															  i,
															  w.scheme,
															  f,
															  t);
				// Energy (f.t) intentionally dropped - see class doc.
				f_acc.x += f.x;
				f_acc.y += f.y;
				f_acc.z += f.z;
				t_acc += t;
			}
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

		if (tid == 0 && active) {
			atomic_add(&rb.force[w.rb_i], force[0]);
			atomic_add(&rb.torque[w.rb_i], torque[0]);
			if (w.rb_j >= 0) {
				const Vector3 reaction_force = -force[0];
				const Vector3 reaction_torque = -(torque[0] + w.origin_offset.cross(force[0]));
				atomic_add(&rb.force[w.rb_j], reaction_force);
				atomic_add(&rb.torque[w.rb_j], reaction_torque);
			}
		}
	}
};

} // namespace ARBD

#ifdef USE_CUDA
#include "Backend/CUDA/KernelHelper.cuh"
namespace ARBD {
extern template Event launch_cuda_kernel(const Resource& resource,
										 const KernelConfig& config,
										 RBGridCullKernel kernel_func);
extern template Event launch_cuda_kernel(const Resource& resource,
										 const KernelConfig& config,
										 RBGridPrefixSumKernel kernel_func);
extern template Event launch_cuda_kernel_with_workitem(const Resource& resource,
													   const KernelConfig& config,
													   RBGridBatchedForceKernel kernel_func);
} // namespace ARBD
#endif

#ifdef USE_SYCL
#include <sycl/sycl.hpp>
template<>
struct sycl::is_device_copyable<ARBD::RBGridCullKernel> : std::true_type {};
template<>
struct sycl::is_device_copyable<ARBD::RBGridPrefixSumKernel> : std::true_type {};
template<>
struct sycl::is_device_copyable<ARBD::RBGridBatchedForceKernel> : std::true_type {};
#endif
