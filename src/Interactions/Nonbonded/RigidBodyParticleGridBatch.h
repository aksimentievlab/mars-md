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

namespace MARS {

/**
 * @brief One (RB instance, potential-grid) task for the batched
 *        particle-RB force kernel, built on-device by
 *        RBParticleGridBuildKernel.
 */
struct RBParticleGridWork {
	Matrix3 basis_inv;	// (R_rb * grid.basis).inverse() - lab-frame, world->grid-local
	Vector3 origin_lab; // R_rb * grid.origin + rb.position
	Vector3 grid_shift; ///< R_rb * grid.origin: rb position -> grid origin, lab frame
	float scale;		// GridTerm::scale for this candidate's grid term
	int grid_id;
	int rb_id;
	InterpolationOrder scheme;
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
	const BaseGridView<mars_real>* __restrict__ grid_views;
	InterpolationOrder scheme;

	RBParticleGridWork* __restrict__ work_out;

	KERNEL_FUNC void operator()(idx_t idx) const {
		if (idx >= num_candidates)
			return;

		const int rb_id = candidate_rb_id[idx];
		const int grid_id = candidate_grid_id[idx];
		const Matrix3 R = rb.orientation[rb_id];
		const BaseGridView<mars_real> grid = grid_views[grid_id];

		RBParticleGridWork w;
		w.basis_inv = (R * grid.basis).inverse();
		w.grid_shift = R * grid.origin;
		w.origin_lab = w.grid_shift + rb.position[rb_id];
		w.scale = candidate_scale[idx];
		w.grid_id = grid_id;
		w.rb_id = rb_id;
		w.scheme = scheme;
		work_out[idx] = w;
	}
};

/**
 * @brief Batched particle-in-RB-grid force; RB gets the reaction force/torque.
 * gridDim.x == num_candidates * blocks_per_candidate.
 * Particle gets force + half the energy; RB gets force/torque + the other half in force.t.
 * Only particle types whose rigidBodyPotential names the grid's key take part.
 * RB torque is about the body position (shifted from the grid origin).
 */
struct RBParticleGridForceKernel {
	RigidBodyView rb;
	ParticleView particles;
	const RBParticleGridWork* __restrict__ work;
	const BaseGridView<mars_real>* __restrict__ grid_views;
	idx_t num_particles;
	idx_t blocks_per_candidate;
	idx_t block_size;
	/// [candidate][particle type] -> 1 if that type samples the candidate's grid.
	const uint8_t* __restrict__ type_mask;
	idx_t num_particle_types;

	template<typename WorkItemT>
	KERNEL_FUNC void operator()(size_t, WorkItemT& item) const {
		Vector3* force = item.template get_shared_mem<Vector3>(0);
		Vector3* torque = item.template get_shared_mem<Vector3>(block_size * sizeof(Vector3));

		const idx_t block_id = item.group_id();
		const idx_t tid = item.local_id();
		const idx_t item_idx = block_id / blocks_per_candidate;
		const idx_t slice = block_id % blocks_per_candidate;
		const RBParticleGridWork w = work[item_idx];
		const BaseGridView<mars_real> grid = grid_views[w.grid_id];
		const uint8_t* mask = type_mask + item_idx * num_particle_types;

		Vector3 f_acc(0.0f);
		Vector3 t_acc(0.0f);
		const idx_t stride = block_size * blocks_per_candidate;
		for (idx_t p = slice * block_size + tid; p < num_particles; p += stride) {
			if (!mask[particles.type_id[p]])
				continue;
			const Vector3 pos = particles.pos[p];
			const Vector3 local = w.basis_inv.transform(pos - w.origin_lab);
			const Matrix3 identity(1.0f);
			const GridSample<mars_real> sample = (w.scheme == InterpolationOrder::Linear)
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
			fe.t = 0.5f * w.scale * sample.value;
			atomic_add(&particles.ForceEnergy[p], fe);

			f_acc -= force_lab;
			f_acc.t += fe.t;
			t_acc += (pos - w.origin_lab).cross(-force_lab);
		}
		force[tid] = f_acc;
		torque[tid] = t_acc;

		item.barrier();
		for (idx_t offset = block_size / 2; offset > 0; offset >>= 1) {
			if (tid < offset) {
				force[tid] += force[tid + offset];
				force[tid].t += force[tid + offset].t;
				torque[tid] += torque[tid + offset];
			}
			item.barrier();
		}

		if (tid == 0) {
			// Torque is reduced about the grid origin; shift to the body position.
			atomic_add(&rb.force[w.rb_id], force[0]);
			atomic_add(&rb.torque[w.rb_id], torque[0] + w.grid_shift.cross(force[0]));
		}
	}
};

} // namespace MARS

#ifdef USE_CUDA
#include "Backend/CUDA/KernelHelper.cuh"
namespace MARS {
extern template Event launch_cuda_kernel(const Resource& resource,
										 const KernelConfig& config,
										 RBParticleGridBuildKernel kernel_func);
extern template Event launch_cuda_kernel_with_workitem(const Resource& resource,
													   const KernelConfig& config,
													   RBParticleGridForceKernel kernel_func);
} // namespace MARS
#endif

#ifdef USE_SYCL
#include <sycl/sycl.hpp>
template<>
struct sycl::is_device_copyable<MARS::RBParticleGridBuildKernel> : std::true_type {};
template<>
struct sycl::is_device_copyable<MARS::RBParticleGridForceKernel> : std::true_type {};
#endif
