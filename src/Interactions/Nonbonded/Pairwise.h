#pragma once
#include "../Interactions.h"
#include "Backend/Kernels.h"
#include "Header.h"
#include "Interactions/Nonbonded/Columb.h"
#include "Interactions/TabulatedPotential.h"
#include "SimParam.h"
#include "Types/Types.h"

namespace MARS {

/**
 * @brief Which nonbonded pair terms apply to one pair, as a bitmask.
 *
 * Analytical terms and the tabulated potential share one mask so a single
 * per-pair tag drives both AnalyticalPairKernel and TabulatedNonBondedComputer.
 * @see AnalyticalPairKernels.md
 */
enum AnalyticalPairTerm : uint32_t {
	PAIR_TERM_NONE = 0u,
	PAIR_TERM_COULOMB = 1u << 0,
	PAIR_TERM_DEBYE_HUCKEL = 1u << 1,
	PAIR_TERM_ONCK = 1u << 2,
	PAIR_TERM_GAUSSIAN = 1u << 3,
	PAIR_TERM_SOFTCORE = 1u << 4,
	PAIR_TERM_TABULATED = 1u << 5
};

/// Low bits of a pair tag hold the term mask; the rest hold the table index.
inline constexpr uint32_t kPairTermMask = 0xFFu;
inline constexpr uint32_t kPairTableShift = 8;

/// Largest table index a pair tag can carry above the term mask.
inline constexpr int kMaxPairTableIndex = static_cast<int>(0xFFFFFFFFu >> kPairTableShift);

/// @brief Pack a term mask and tabulated table index into one pair tag.
inline constexpr uint32_t make_pair_tag(uint32_t terms, int table_index) {
	return (terms & kPairTermMask) |
		   (table_index >= 0 ? (static_cast<uint32_t>(table_index) << kPairTableShift) : 0u);
}

/// @brief Recover the tabulated table index from a pair tag.
DEVICE inline int pair_tag_table_index(uint32_t tag) {
	return static_cast<int>(tag >> kPairTableShift);
}

/**
 * @brief Gaussian: U = amp exp(-r^2/sigma^2), zero beyond cutoff
 * @param amp Well depth at r=0
 * @param sigma Width (Angstrom)
 * @param cutoff_squared Squared truncation radius; <= 0 disables the cutoff
 */
struct GaussianPotential {
	mars_real amp = mars_real(1);
	mars_real sigma = mars_real(1);
	mars_real cutoff_squared = mars_real(0);

	DEVICE ScalarForceEnergy compute(mars_real distance) const {
		const mars_real d2 = distance * distance;
		if (cutoff_squared > mars_real(0) && d2 > cutoff_squared) {
			return ScalarForceEnergy{float2{mars_real(0), mars_real(0)}};
		}
		const mars_real inv_s2 = mars_real(1) / (sigma * sigma);
		const mars_real energy = amp * math::exp(-d2 * inv_s2);
		// F = -dU/dr = 2 r U / sigma^2
		const mars_real force = mars_real(2) * distance * energy * inv_s2;
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
			pbox->wrap_diff(positions[neighbor_indices.y] - positions[neighbor_indices.x]);
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

/// Resolve each pair's term tag once per rebuild (PAIR_TERM_NONE = skip). See dev_notes.md.
/// Exclusions are applied by the pairlist builder and never reach this kernel.
struct ResolvePairTableKernel {
	DEVICE_PTR(const int2) __restrict__ particle_indices;
	DEVICE_PTR(const int) __restrict__ type_ids;
	DEVICE_PTR(const uint32_t) __restrict__ pairwise_term_matrix;
	idx_t num_particle_types;
	DEVICE_PTR(uint32_t) pair_tag; // output: term mask + table index
	idx_t num_pairs;

	KERNEL_FUNC void operator()(idx_t i) const {
		if (i >= num_pairs)
			return;
		const int2& indices = particle_indices[i];
		const int type_i = type_ids[indices.x];
		const int type_j = type_ids[indices.y];
		pair_tag[i] = pairwise_term_matrix[type_i * num_particle_types + type_j];
	}
};

struct TabulatedNonBondedComputer {
	// Members
	DEVICE_PTR(const int2) __restrict__ particle_indices;
	DEVICE_PTR(const Vector3) __restrict__ positions;
	DEVICE_PTR(Vector3) force_energy;
	DEVICE_PTR(const uint32_t) __restrict__ pair_tag; // per-pair term mask + table index
	DEVICE_PTR(const TabulatedPotential) __restrict__ tables;
	const PeriodicBox* __restrict__ pbox;
	bool get_energy;
	idx_t num_pairs;
	float cutoff_squared;

	// Constructor
	TabulatedNonBondedComputer(DEVICE_PTR(const int2) indices,
							   DEVICE_PTR(Vector3) pos,
							   DEVICE_PTR(Vector3) fe,
							   DEVICE_PTR(const uint32_t) tag,
							   DEVICE_PTR(const TabulatedPotential) tabs,
							   const PeriodicBox* box,
							   bool energy,
							   idx_t n_pairs,
							   float cutoff_sq)
		: particle_indices(indices), positions(pos), force_energy(fe), pair_tag(tag), tables(tabs),
		  pbox(box), get_energy(energy), num_pairs(n_pairs), cutoff_squared(cutoff_sq) {}

	// Kernel operator
	KERNEL_FUNC void operator()(idx_t i) const {
		if (i >= num_pairs)
			return;

		// Type/table resolution was done at rebuild (ResolvePairTableKernel).
		const uint32_t tag = pair_tag[i];
		if (!(tag & PAIR_TERM_TABULATED))
			return;
		const int tidx = pair_tag_table_index(tag);

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

		// get_energy is warp-uniform, so this branch costs nothing. Energy rides in
		// .t rather than taking two extra atomics; the force-only path skips .t
		// entirely instead of atomically adding zero. See dev_notes.md.
		if (get_energy) {
			atomic_add(&force_energy[indices.x], Vector3(-force.x, -force.y, -force.z, energy));
			atomic_add(&force_energy[indices.y], Vector3(force.x, force.y, force.z, energy));
		} else {
			atomic_add_xyz(&force_energy[indices.x], -force);
			atomic_add_xyz(&force_energy[indices.y], force);
		}
	}
};
struct PairNonbondedComputer {
	DEVICE_PTR(const int2) __restrict__ neighbor_pairs;
	ParticleView particles;
	DEVICE_PTR(const uint32_t) __restrict__ pair_tag; ///< per-pair terms; null applies every term
	DEVICE_PTR(const TabulatedPotential) tabs;
	ParticleTypeView types;
	const PeriodicBox* __restrict__ pbox;
	bool get_energy;
	idx_t num_pairs;
	mars_real cutoff_squared;
	uint32_t enabled_terms;
	ColumbPotential coulomb;
	DebyeHuckelPotential debye_huckel;
	OnckElecPotential onck;
	GaussianPotential gaussian;

	PairNonbondedComputer(DEVICE_PTR(const int2) pairs,
						  ParticleView particles_,
						  DEVICE_PTR(const uint32_t) tag,
						  DEVICE_PTR(const TabulatedPotential) tables,
						  ParticleTypeView types_,
						  const PeriodicBox* box,
						  bool energy,
						  idx_t n_pairs,
						  mars_real cutoff_sq,
						  uint32_t enabled_terms_)
		: neighbor_pairs(pairs), particles(particles_), pair_tag(tag), tabs(tables), types(types_),
		  pbox(box), get_energy(energy), num_pairs(n_pairs), cutoff_squared(cutoff_sq),
		  enabled_terms(enabled_terms_) {}

	KERNEL_FUNC void operator()(idx_t i) const {
		if (i >= num_pairs)
			return;

		// Per-pair terms gate the run-global mask; a null tag leaves it ungated.
		// The raw tag is kept because its high bits carry the table index.
		const uint32_t tag = pair_tag ? pair_tag[i] : enabled_terms;
		const uint32_t terms = tag & enabled_terms;
		if (terms == PAIR_TERM_NONE)
			return;

		const int2& indices = neighbor_pairs[i];

		// One wrap and one sqrt shared by every term. Cutoff stays in-kernel
		// because the list carries a skin. See dev_notes.md.
		CalcDistance geom = CalcDistance::compute(particles.pos, indices, pbox);
		if (geom.distance < mars_real(1e-6))
			return;
		if (cutoff_squared > mars_real(0) && geom.distance * geom.distance > cutoff_squared)
			return;

		const int type_i = particles.type_id[indices.x];
		const int type_j = particles.type_id[indices.y];

		// Charges are only touched by the electrostatic terms, so a run with none
		// of them enabled never dereferences `types`. See dev_notes.md.
		constexpr uint32_t kChargeTerms =
			PAIR_TERM_COULOMB | PAIR_TERM_DEBYE_HUCKEL | PAIR_TERM_ONCK;
		mars_real qi = mars_real(0);
		mars_real qj = mars_real(0);
		if (terms & kChargeTerms) {
			qi = types.charge[type_i];
			qj = types.charge[type_j];
		}

		mars_real force_magnitude = mars_real(0);
		mars_real energy = mars_real(0);

		if (terms & PAIR_TERM_COULOMB) {
			const ScalarForceEnergy fe = ColumbPotential::compute(geom.r_ij, geom.distance, qi, qj);
			force_magnitude += fe.force_magnitude;
			energy += fe.energy;
		}
		if (terms & PAIR_TERM_DEBYE_HUCKEL) {
			const ScalarForceEnergy fe = debye_huckel.compute(geom.distance, qi, qj);
			force_magnitude += fe.force_magnitude;
			energy += fe.energy;
		}
		if (terms & PAIR_TERM_ONCK) {
			const ScalarForceEnergy fe = onck.compute(geom.distance, qi, qj);
			force_magnitude += fe.force_magnitude;
			energy += fe.energy;
		}
		if (terms & PAIR_TERM_GAUSSIAN) {
			const ScalarForceEnergy fe = gaussian.compute(geom.distance);
			force_magnitude += fe.force_magnitude;
			energy += fe.energy;
		}
		if (terms & PAIR_TERM_SOFTCORE) {
			// softcoreForce returns dU/dr divided by r, so it needs converting
			// to this kernel's -dU/dr. eps and radius are per particle type.
			const mars_real rad = mars_real(0.5) * (types.radius[type_i] + types.radius[type_j]);
			const mars_real rad2 = rad * rad;
			const mars_real rad6 = rad2 * rad2 * rad2;
			const mars_real pair_eps = math::sqrt(types.eps[type_i] * types.eps[type_j]);
			const ScalarForceEnergy fe =
				SoftcoreForceKernel::softcoreForce(geom.r_ij, pair_eps, rad6);
			force_magnitude += -fe.force_magnitude * geom.distance;
			energy += fe.energy;
		}
		if (terms & PAIR_TERM_TABULATED) {
			// The table index rides in the high bits of the unmasked tag.
			const int tidx = pair_tag_table_index(tag);
			const ScalarForceEnergy fe = TabulatedPotential::compute(geom.distance, &tabs[tidx]);
			force_magnitude += fe.force_magnitude;
			energy += fe.energy;
		}

		const Vector3 force = geom.unit_vector * force_magnitude;
		const mars_real half_energy = energy * mars_real(0.5);

		if (get_energy) {
			atomic_add(&particles.ForceEnergy[indices.x],
					   Vector3(-force.x, -force.y, -force.z, half_energy));
			atomic_add(&particles.ForceEnergy[indices.y],
					   Vector3(force.x, force.y, force.z, half_energy));
		} else {
			atomic_add_xyz(&particles.ForceEnergy[indices.x], -force);
			atomic_add_xyz(&particles.ForceEnergy[indices.y], force);
		}
	}
};
} // namespace MARS

#ifdef USE_CUDA
#include "Backend/CUDA/KernelHelper.cuh"
namespace MARS {
extern template Event launch_cuda_kernel(const Resource& resource,
										 const KernelConfig& config,
										 TabulatedNonBondedComputer kernel_func);
extern template Event launch_cuda_kernel(const Resource& resource,
										 const KernelConfig& config,
										 ResolvePairTableKernel kernel_func);
extern template Event launch_cuda_kernel(const Resource& resource,
										 const KernelConfig& config,
										 PairNonbondedComputer kernel_func);
} // namespace MARS
#endif

#ifdef USE_SYCL
#include <sycl/sycl.hpp>
template<>
struct sycl::is_device_copyable<MARS::TabulatedNonBondedComputer> : std::true_type {};
template<>
struct sycl::is_device_copyable<MARS::ResolvePairTableKernel> : std::true_type {};
template<>
struct sycl::is_device_copyable<MARS::PairNonbondedComputer> : std::true_type {};
template<>
struct sycl::is_device_copyable<MARS::GaussianPotential> : std::true_type {};
#endif
