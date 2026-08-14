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

/**
 * @brief Pairwise tabulated nonbonded force computer
 * @param particle_indices Particle indices
 * @param positions Particle positions
 * @param force_energy Particle force and energy
 * @param type_ids Particle types
 * @param pairwise_table_matrix Pairwise table matrix
 * @param pairwise_form_matrix Pairwise form matrix
 * @param tables Pairwise tables
 * @param num_particle_types Number of particle types
 * @param excl_offsets Exclusion offsets
 * @param excl_neighbors Exclusion neighbors
 * @param num_excl_particles Number of excluded particles
 * @param pbox Periodic box
 * @param get_energy Get energy
 * @param num_pairs Number of pairs
 * @param cutoff_squared Cutoff squared
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
	DEVICE_PTR(const int) excl_offsets;	  // CSR: per-particle offsets, size num_excl_particles+1
	DEVICE_PTR(const int) excl_neighbors; // CSR: excluded partners, concatenated per particle
	idx_t num_excl_particles;			  // particles covered by excl_offsets
	const PeriodicBox* pbox;
	bool get_energy;
	idx_t num_pairs;
	float cutoff_squared;

	// Constructor
	TabulatedNonBondedComputer(DEVICE_PTR(const int2) indices,
							   DEVICE_PTR(Vector3) pos,
							   DEVICE_PTR(Vector3) fe,
							   DEVICE_PTR(const int) type_id_arr,
							   DEVICE_PTR(const int) table_matrix,
							   DEVICE_PTR(const int) form_matrix,
							   DEVICE_PTR(const TabulatedPotential) tabs,
							   idx_t n_types,
							   DEVICE_PTR(const int) excl_off,
							   DEVICE_PTR(const int) excl_nbr,
							   idx_t n_excl_particles,
							   const PeriodicBox* box,
							   bool energy,
							   idx_t n_pairs,
							   float cutoff_sq)
		: particle_indices(indices), positions(pos), force_energy(fe), type_ids(type_id_arr),
		  pairwise_table_matrix(table_matrix), pairwise_form_matrix(form_matrix), tables(tabs),
		  num_particle_types(n_types), excl_offsets(excl_off), excl_neighbors(excl_nbr),
		  num_excl_particles(n_excl_particles), pbox(box), get_energy(energy), num_pairs(n_pairs),
		  cutoff_squared(cutoff_sq) {}

	// Kernel operator
	KERNEL_FUNC void operator()(idx_t i) const {
		if (i >= num_pairs)
			return;

		const int2& indices = particle_indices[i];

		// Phase 0: Reject pairs outside the interaction cutoff.
		//
		// The pair list is deliberately built at a larger radius (interaction
		// cutoff + skin) so it stays valid for several steps between rebuilds,
		// so it necessarily contains pairs that do not interact. At a 35 A
		// cutoff with a 10 A skin those are (45^3-35^3)/45^3 = 53% of its
		// entries. Evaluating them is not merely wasted work: the tabulated
		// potentials end at the interaction cutoff, so a lookup past that point
		// is out of range and contributes a spurious interaction. This check
		// must therefore live here - the pair list cannot do it, since removing
		// the skin is exactly what would force a rebuild every step.
		//
		// Done before the exclusion scan and the type-pair lookup because it
		// rejects the largest fraction of pairs for the least work.
		CalcDistance geom = CalcDistance::compute(positions, indices, pbox);
		if (geom.distance < 1e-6f)
			return;
		if (cutoff_squared > 0.0f && geom.distance * geom.distance > cutoff_squared)
			return;

		// Phase 1: Skip excluded pairs (bonded neighbors etc.).
		// Scan only particle `indices.x`'s excluded-partner list (CSR), which is
		// ~degree(indices.x) entries rather than the whole exclusion set. Each exclusion is stored
		// in both partners' lists, so checking one endpoint is sufficient. This keeps the per-pair
		// exclusion test O(1) w.r.t. system size; the old full scan was O(N) per
		// pair, i.e. O(N^2) per step, which made large systems appear to hang.
		if (indices.x < static_cast<int>(num_excl_particles)) {
			const int begin = excl_offsets[indices.x];
			const int end = excl_offsets[indices.x + 1];
			for (int e = begin; e < end; ++e) {
				if (excl_neighbors[e] == indices.y)
					return; // excluded
			}
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
									   DEVICE_PTR(const int) excl_offsets,
									   DEVICE_PTR(const int) excl_neighbors,
									   idx_t num_excl_particles,
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
										type_ids,
										pairwise_table_matrix,
										pairwise_form_matrix,
										tables,
										num_particle_types,
										excl_offsets,
										excl_neighbors,
										num_excl_particles,
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
} // namespace ARBD
#endif

#ifdef USE_SYCL
#include <sycl/sycl.hpp>
template<>
struct sycl::is_device_copyable<ARBD::TabulatedNonBondedComputer> : std::true_type {};
template<>
struct sycl::is_device_copyable<ARBD::GaussianPotential> : std::true_type {};
#endif
