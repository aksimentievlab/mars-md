#pragma once
#include "Backend/Kernels.h"
#include "Columb.h"
#include "Header.h"
#include "Interactions/Interactions.h"
#include "Objects/DeviceParticle.h"
#include "Pairwise.h"
#include "System/PeriodicBox.h"
#include "Types/Types.h"

namespace ARBD {

/**
 * @brief Which analytical pair terms a single AnalyticalPairKernel applies
 * @see AnalyticalPairKernels.md
 */
enum AnalyticalPairTerm : uint32_t {
	PAIR_TERM_NONE = 0u,
	PAIR_TERM_COULOMB = 1u << 0,
	PAIR_TERM_DEBYE_HUCKEL = 1u << 1,
	PAIR_TERM_ONCK = 1u << 2,
	PAIR_TERM_GAUSSIAN = 1u << 3,
	PAIR_TERM_SOFTCORE = 1u << 4
};

/**
 * @brief All analytical nonbonded pair potentials in one pass over the pairlist
 *
 * One kernel rather than one per potential: the pair geometry is the expensive
 * part and is shared, and a single concrete type needs a single explicit CUDA
 * instantiation (see NonbondedInstantiations.cu) instead of one per term.
 *
 * Sign convention matches AnalyticalBondComputer: `force_magnitude` is
 * @f$-dU/dr@f$, positive when the pair repels, applied as `-force` to the first
 * particle and `+force` to the second along the first-to-second unit vector.
 *
 * @see AnalyticalPairKernels.md
 */
struct AnalyticalPairKernel {
	DEVICE_PTR(const int2) neighbor_pairs;
	DEVICE_PTR(Vector3) positions;
	DEVICE_PTR(Vector3) force_energy;
	DEVICE_PTR(const int) type_ids;
	ParticleTypeView types;
	const PeriodicBox* pbox;

	uint32_t enabled_terms;
	ColumbPotential coulomb;
	DebyeHuckelPotential debye_huckel;
	OnckElecPotential onck;
	GaussianPotential gaussian;

	bool get_energy;
	idx_t num_pairs;
	arbd_real cutoff_squared;

	KERNEL_FUNC void operator()(idx_t i) const {
		if (i >= num_pairs)
			return;

		const int2& indices = neighbor_pairs[i];

		CalcDistance geom = CalcDistance::compute(positions, indices, pbox);
		if (geom.distance < arbd_real(1e-6))
			return;
		if (cutoff_squared > arbd_real(0) && geom.distance * geom.distance > cutoff_squared)
			return;

		const int type_i = type_ids[indices.x];
		const int type_j = type_ids[indices.y];
		const arbd_real qi = types.charge[type_i];
		const arbd_real qj = types.charge[type_j];

		arbd_real force_magnitude = arbd_real(0);
		arbd_real energy = arbd_real(0);

		if (enabled_terms & PAIR_TERM_COULOMB) {
			const ScalarForceEnergy fe =
				ColumbPotential::compute(geom.r_ij, geom.distance, qi, qj);
			force_magnitude += fe.force_magnitude;
			energy += fe.energy;
		}
		if (enabled_terms & PAIR_TERM_DEBYE_HUCKEL) {
			const ScalarForceEnergy fe = debye_huckel.compute(geom.distance, qi, qj);
			force_magnitude += fe.force_magnitude;
			energy += fe.energy;
		}
		if (enabled_terms & PAIR_TERM_ONCK) {
			const ScalarForceEnergy fe = onck.compute(geom.distance, qi, qj);
			force_magnitude += fe.force_magnitude;
			energy += fe.energy;
		}
		if (enabled_terms & PAIR_TERM_GAUSSIAN) {
			const ScalarForceEnergy fe = gaussian.compute(geom.distance);
			force_magnitude += fe.force_magnitude;
			energy += fe.energy;
		}
		if (enabled_terms & PAIR_TERM_SOFTCORE) {
			// softcoreForce returns dU/dr divided by r, so it needs converting
			// to this kernel's -dU/dr. eps and radius are per particle type.
			const arbd_real rad = arbd_real(0.5) * (types.radius[type_i] + types.radius[type_j]);
			const arbd_real rad2 = rad * rad;
			const arbd_real rad6 = rad2 * rad2 * rad2;
			const arbd_real pair_eps = math::sqrt(types.eps[type_i] * types.eps[type_j]);
			const ScalarForceEnergy fe =
				SoftcoreForceKernel::softcoreForce(geom.r_ij, pair_eps, rad6);
			force_magnitude += -fe.force_magnitude * geom.distance;
			energy += fe.energy;
		}

		const Vector3 force = geom.unit_vector * force_magnitude;
		atomic_add(&force_energy[indices.x], -force);
		atomic_add(&force_energy[indices.y], force);

		if (get_energy) {
			const arbd_real half = energy * arbd_real(0.5);
			atomic_add(&force_energy[indices.x].t, half);
			atomic_add(&force_energy[indices.y].t, half);
		}
	}
};

/**
 * @brief Launch every enabled analytical nonbonded term in a single pass
 */
inline Event launch_analytical_pair_nonbonded(const Resource& resource,
											  AnalyticalPairKernel kernel,
											  idx_t num_pairs) {
	if (num_pairs == 0 || kernel.enabled_terms == PAIR_TERM_NONE) {
		return Event(nullptr, resource);
	}
	return launch_kernel(resource, KernelConfig::for_1d(num_pairs, resource), kernel);
}

} // namespace ARBD

#ifdef USE_CUDA
#include "Backend/CUDA/KernelHelper.cuh"
namespace ARBD {
// Forces the instantiation into NonbondedInstantiations.cu; without this a
// host-only .cpp launching the kernel instantiates the stub in
// KernelHelper.cuh and throws NotImplementedError.
extern template Event launch_cuda_kernel(const Resource& resource,
										 const KernelConfig& config,
										 AnalyticalPairKernel kernel_func);
} // namespace ARBD
#endif

#ifdef USE_SYCL
#include <sycl/sycl.hpp>
template<>
struct sycl::is_device_copyable<ARBD::AnalyticalPairKernel> : std::true_type {};
#endif
