#pragma once
#include "../Interactions.h"
#include "Backend/Kernels.h"
#include "Header.h"
#include "Interactions/TabulatedPotential.h"
#include "SimParam.h"
#include "Types/Types.h"

namespace ARBD {

/**
 * @brief Gaussian: U = amp exp(-r^2/sigma^2), zero beyond cutoff
 * @param amp Well depth at r=0
 * @param sigma Width (Angstrom)
 * @param cutoff_squared Squared truncation radius; <= 0 disables the cutoff
 */
struct GaussianPotential {
	arbd_real amp = arbd_real(1);
	arbd_real sigma = arbd_real(1);
	arbd_real cutoff_squared = arbd_real(0);

	DEVICE ScalarForceEnergy compute(arbd_real distance) const {
		const arbd_real d2 = distance * distance;
		if (cutoff_squared > arbd_real(0) && d2 > cutoff_squared) {
			return ScalarForceEnergy{float2{arbd_real(0), arbd_real(0)}};
		}
		const arbd_real inv_s2 = arbd_real(1) / (sigma * sigma);
		const arbd_real energy = amp * math::exp(-d2 * inv_s2);
		// F = -dU/dr = 2 r U / sigma^2
		const arbd_real force = arbd_real(2) * distance * energy * inv_s2;
		return ScalarForceEnergy{float2{force, energy}};
	}
};

struct SoftcoreForceKernel {
	KERNEL_FUNC void operator()(ScalarForceEnergy force_energy,
								const Vector3* positions,
								float eps,
								float rad6,
								const int2& neighbor_indices,
								const PeriodicBox* pbox) {
		Vector3 r_ij =
			pbox->wrapDiff(positions[neighbor_indices.y] - positions[neighbor_indices.x]);
		float distance = r_ij.length();
		ScalarForceEnergy fe = softcoreForce(r_ij, eps, rad6);
	}

	DEVICE static inline ScalarForceEnergy softcoreForce(const Vector3& r, float eps, float rad6) {
		const float d2 = r.length2();
		const float d6 = d2 * d2 * d2;

		float force = -12 * eps * (rad6 * rad6 / (d6 * d6 * d2) - rad6 / (d6 * d2));

		if (d6 < rad6) {
			const float d6_2 = d6 * d6;
			const float rad6_2 = rad6 * rad6;
			float e = eps * ((rad6_2 / (d6_2)) - (2.0f * rad6 / d6)) + eps;
			float f = -12.0f * eps * (rad6_2 / (d6_2 * d2) - rad6 / (d6 * d2));
			return ScalarForceEnergy{float2{f, e}};
		}

		return ScalarForceEnergy{float2{force, 0.0f}};
	};
};

/// Resolve each pair's table index once per rebuild (-1 = skip). See dev_notes.md.
struct ResolvePairTableKernel {
	DEVICE_PTR(const int2) __restrict__ particle_indices;
	DEVICE_PTR(const int) __restrict__ type_ids;
	DEVICE_PTR(const int) __restrict__ pairwise_table_matrix;
	DEVICE_PTR(const int) __restrict__ pairwise_form_matrix;
	idx_t num_particle_types;
	DEVICE_PTR(const int) __restrict__ excl_offsets;   // CSR: per-particle offsets, size n_excl+1
	DEVICE_PTR(const int) __restrict__ excl_neighbors; // CSR: excluded partners, concatenated
	idx_t num_excl_particles;
	DEVICE_PTR(int) table_idx; // output: per-pair table index, -1 = skip
	idx_t num_pairs;

	KERNEL_FUNC void operator()(idx_t i) const {
		if (i >= num_pairs)
			return;
		const int2& indices = particle_indices[i];

		// Excluded (bonded neighbor etc.): each exclusion is stored in both
		// partners' lists, so scanning one endpoint's list is sufficient.
		if (indices.x < static_cast<int>(num_excl_particles)) {
			const int begin = excl_offsets[indices.x];
			const int end = excl_offsets[indices.x + 1];
			for (int e = begin; e < end; ++e) {
				if (excl_neighbors[e] == indices.y) {
					table_idx[i] = -1;
					return;
				}
			}
		}

		const int type_i = type_ids[indices.x];
		const int type_j = type_ids[indices.y];
		const idx_t matrix_idx = type_i * num_particle_types + type_j;
		const int tidx = pairwise_table_matrix[matrix_idx];
		if (tidx < 0 || static_cast<InteractionForm>(pairwise_form_matrix[matrix_idx]) !=
							 InteractionForm::Tabulated) {
			table_idx[i] = -1;
			return;
		}
		table_idx[i] = tidx;
	}
};

struct TabulatedNonBondedComputer {
	// Members
	DEVICE_PTR(const int2) __restrict__ particle_indices;
	DEVICE_PTR(const Vector3) __restrict__ positions;
	DEVICE_PTR(Vector3) force_energy;
	DEVICE_PTR(const int) __restrict__ table_idx; // per-pair, precomputed (-1 = skip)
	DEVICE_PTR(const TabulatedPotential) __restrict__ tables;
	const PeriodicBox* __restrict__ pbox;
	bool get_energy;
	idx_t num_pairs;
	float cutoff_squared;

	// Constructor
	TabulatedNonBondedComputer(DEVICE_PTR(const int2) indices,
							   DEVICE_PTR(Vector3) pos,
							   DEVICE_PTR(Vector3) fe,
							   DEVICE_PTR(const int) tab_idx,
							   DEVICE_PTR(const TabulatedPotential) tabs,
							   const PeriodicBox* box,
							   bool energy,
							   idx_t n_pairs,
							   float cutoff_sq)
		: particle_indices(indices), positions(pos), force_energy(fe), table_idx(tab_idx),
		  tables(tabs), pbox(box), get_energy(energy), num_pairs(n_pairs),
		  cutoff_squared(cutoff_sq) {}

	// Kernel operator
	KERNEL_FUNC void operator()(idx_t i) const {
		if (i >= num_pairs)
			return;

		// Exclusion + type/table resolution was done at rebuild (ResolvePairTableKernel).
		const int tidx = table_idx[i];
		if (tidx < 0)
			return;

		const int2& indices = particle_indices[i];

		// Cutoff stays in-kernel (list carries skin). See dev_notes.md.
		CalcDistance geom = CalcDistance::compute(positions, indices, pbox);
		if (geom.distance < 1e-6f)
			return;
		if (cutoff_squared > 0.0f && geom.distance * geom.distance > cutoff_squared)
			return;

		const ScalarForceEnergy fe = TabulatedPotential::compute(geom.distance, &tables[tidx]);

		const Vector3 force = geom.unit_vector * fe.force_magnitude;
		const float energy = fe.energy * 0.5f;

		atomic_add(&force_energy[indices.x], -force);
		atomic_add(&force_energy[indices.y], force);

		if (get_energy) {
			atomic_add(&force_energy[indices.x].t, energy);
			atomic_add(&force_energy[indices.y].t, energy);
		}
	}
};

/**
 * @brief Resolve per-pair table indices; run once per pairlist rebuild.
 */
inline Event launch_resolve_pair_tables(const Resource& resource,
										DEVICE_PTR(const int2) particle_indices,
										DEVICE_PTR(const int) type_ids,
										DEVICE_PTR(const int) pairwise_table_matrix,
										DEVICE_PTR(const int) pairwise_form_matrix,
										idx_t num_particle_types,
										DEVICE_PTR(const int) excl_offsets,
										DEVICE_PTR(const int) excl_neighbors,
										idx_t num_excl_particles,
										DEVICE_PTR(int) table_idx,
										idx_t num_pairs) {
	if (num_pairs == 0)
		return Event(nullptr, resource);
	KernelConfig config = KernelConfig::for_1d(num_pairs, resource);
	ResolvePairTableKernel resolver{particle_indices,
									type_ids,
									pairwise_table_matrix,
									pairwise_form_matrix,
									num_particle_types,
									excl_offsets,
									excl_neighbors,
									num_excl_particles,
									table_idx,
									num_pairs};
	return launch_kernel(resource, config, resolver);
}

/**
 * @brief Launch pairwise tabulated nonbonded force computation.
 * @note `table_idx` must be filled by launch_resolve_pair_tables after each rebuild.
 */
inline Event launch_pairwise_nonbonded(const Resource& resource,
									   DEVICE_PTR(const int2) particle_indices,
									   DEVICE_PTR(Vector3) positions,
									   DEVICE_PTR(Vector3) force_energy,
									   DEVICE_PTR(const int) table_idx,
									   DEVICE_PTR(const TabulatedPotential) tables,
									   const PeriodicBox* pbox,
									   bool get_energy,
									   idx_t num_pairs,
									   float cutoff_squared) {
	if (num_pairs == 0)
		return Event(nullptr, resource);

	KernelConfig config = KernelConfig::for_1d(num_pairs, resource);

	TabulatedNonBondedComputer computer(particle_indices,
										positions,
										force_energy,
										table_idx,
										tables,
										pbox,
										get_energy,
										num_pairs,
										cutoff_squared);

	return launch_kernel(resource, config, computer);
}

} // namespace ARBD

#ifdef USE_CUDA
#include "Backend/CUDA/KernelHelper.cuh"
namespace ARBD {
extern template Event launch_cuda_kernel(const Resource& resource,
										 const KernelConfig& config,
										 TabulatedNonBondedComputer kernel_func);
extern template Event launch_cuda_kernel(const Resource& resource,
										 const KernelConfig& config,
										 ResolvePairTableKernel kernel_func);
} // namespace ARBD
#endif

#ifdef USE_SYCL
#include <sycl/sycl.hpp>
template<>
struct sycl::is_device_copyable<ARBD::TabulatedNonBondedComputer> : std::true_type {};
template<>
struct sycl::is_device_copyable<ARBD::ResolvePairTableKernel> : std::true_type {};
template<>
struct sycl::is_device_copyable<ARBD::GaussianPotential> : std::true_type {};
#endif
