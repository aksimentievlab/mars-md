#pragma once
#include "../Interactions.h"
#include "Backend/Kernels.h"
#include "Header.h"
#include "Interactions/TabulatedPotential.h"
#include "SimParam.h"
#include "Types/Types.h"

namespace ARBD {

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

/**
 * @brief Pairwise tabulated nonbonded force computer
 *
 * `particle_indices` comes from a Pairlist (see Pairlist.h) - already filtered
 * by cutoff, and already remapped to original particle indices (see
 * ZOrderNeighborKernel). Exclusions are a flat, unsorted list (bonded
 * neighbors etc., see DeviceBondedInteractions::exclusion_pairs()) - checked
 * via a linear scan per pair, which is fine for the small exclusion counts
 * bonded topology produces; a sorted-range merge (as legacy's excludeMap
 * does) would be needed to scale this to much larger systems.
 *
 * Table lookup mirrors DevicePairNonBondedInteractions: a flat
 * num_particle_types*num_particle_types matrix maps (type_i, type_j) to an
 * index into `tables` (-1 = no interaction for that type pair).
 */
struct TabulatedNonBondedComputer {
	// Members
	DEVICE_PTR(const int2) particle_indices;
	DEVICE_PTR(Vector3) positions;
	DEVICE_PTR(Vector3) force_energy;
	DEVICE_PTR(const int) type_ids;
	DEVICE_PTR(const int) pairwise_table_matrix;
	DEVICE_PTR(const int) pairwise_form_matrix;
	DEVICE_PTR(const TabulatedPotential) tables;
	idx_t num_particle_types;
	DEVICE_PTR(const int2) exclusions;
	idx_t num_exclusions;
	const PeriodicBox* pbox;
	bool get_energy;
	idx_t num_pairs;

	// Constructor
	TabulatedNonBondedComputer(DEVICE_PTR(const int2) indices,
							   DEVICE_PTR(Vector3) pos,
							   DEVICE_PTR(Vector3) fe,
							   DEVICE_PTR(const int) type_id_arr,
							   DEVICE_PTR(const int) table_matrix,
							   DEVICE_PTR(const int) form_matrix,
							   DEVICE_PTR(const TabulatedPotential) tabs,
							   idx_t n_types,
							   DEVICE_PTR(const int2) excl,
							   idx_t n_excl,
							   const PeriodicBox* box,
							   bool energy,
							   idx_t n_pairs)
		: particle_indices(indices), positions(pos), force_energy(fe), type_ids(type_id_arr),
		  pairwise_table_matrix(table_matrix), pairwise_form_matrix(form_matrix), tables(tabs),
		  num_particle_types(n_types), exclusions(excl), num_exclusions(n_excl), pbox(box),
		  get_energy(energy), num_pairs(n_pairs) {}

	// Kernel operator
	DEVICE void operator()(idx_t i) const {
		if (i >= num_pairs)
			return;

		const int2& indices = particle_indices[i];

		// Phase 1: Skip excluded pairs (bonded neighbors etc.)
		for (idx_t e = 0; e < num_exclusions; ++e) {
			const int2& ex = exclusions[e];
			if ((ex.x == indices.x && ex.y == indices.y) ||
				(ex.x == indices.y && ex.y == indices.x))
				return;
		}

		// Phase 2: Type-pair -> table lookup
		const int type_i = type_ids[indices.x];
		const int type_j = type_ids[indices.y];
		const idx_t matrix_idx = type_i * num_particle_types + type_j;
		const int table_idx = pairwise_table_matrix[matrix_idx];
		if (table_idx < 0)
			return;
		if (static_cast<InteractionForm>(pairwise_form_matrix[matrix_idx]) !=
			InteractionForm::Tabulated)
			return;

		// Phase 3: Compute geometry
		CalcDistance geom = CalcDistance::compute(positions, indices, pbox);
		if (geom.distance < 1e-6f)
			return;

		// Phase 4: Lookup force from tabulated potential
		const ScalarForceEnergy fe = TabulatedPotential::compute(geom.distance, &tables[table_idx]);

		// Phase 5: Apply forces
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
 * @brief Launch pairwise tabulated nonbonded force computation
 */
inline Event launch_pairwise_nonbonded(const Resource& resource,
									   DEVICE_PTR(const int2) particle_indices,
									   DEVICE_PTR(Vector3) positions,
									   DEVICE_PTR(Vector3) force_energy,
									   DEVICE_PTR(const int) type_ids,
									   DEVICE_PTR(const int) pairwise_table_matrix,
									   DEVICE_PTR(const int) pairwise_form_matrix,
									   DEVICE_PTR(const TabulatedPotential) tables,
									   idx_t num_particle_types,
									   DEVICE_PTR(const int2) exclusions,
									   idx_t num_exclusions,
									   const PeriodicBox* pbox,
									   bool get_energy,
									   idx_t num_pairs) {
	if (num_pairs == 0)
		return Event(nullptr, resource);

	KernelConfig config = KernelConfig::for_1d(num_pairs, resource);

	TabulatedNonBondedComputer computer(particle_indices,
										positions,
										force_energy,
										type_ids,
										pairwise_table_matrix,
										pairwise_form_matrix,
										tables,
										num_particle_types,
										exclusions,
										num_exclusions,
										pbox,
										get_energy,
										num_pairs);

	return launch_kernel(resource, config, computer);
}

} // namespace ARBD

#ifdef USE_CUDA
#include "Backend/CUDA/KernelHelper.cuh"
namespace ARBD {
extern template Event launch_cuda_kernel(const Resource& resource,
										 const KernelConfig& config,
										 TabulatedNonBondedComputer kernel_func);
} // namespace ARBD
#endif

#ifdef USE_SYCL
#include <sycl/sycl.hpp>
template<>
struct sycl::is_device_copyable<ARBD::TabulatedNonBondedComputer> : std::true_type {};
#endif
