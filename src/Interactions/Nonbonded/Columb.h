#pragma once
#include "Header.h"
#include "Interactions/Interactions.h"
#include "Objects/DeviceParticle.h"

namespace ARBD {

// ============================================================================
// Columb Potential
// ============================================================================
struct ColumbPotential {
	DEVICE static ScalarForceEnergy
	compute(const Vector3& r_ij, float distance, float qi, float qj) {
		float inv_dist2 = 1.0f / (distance * distance);
		float inv_dist = 1.0f / distance;
		float force = qi * qj * inv_dist2 * constants::COULOMB;
		float energy = qi * qj * inv_dist * constants::COULOMB;
		return ScalarForceEnergy{float2{force, energy}};
	}
};

// ============================================================================
// Columb Force Kernel
// ============================================================================
struct ColumbForceKernel {
	ParticleView particle_view;
	const ParticleTypeView particle_types;
	const int2* neighbor_pairs;
	const PeriodicBox* pbox;
	float epsilon;
	size_t num_pairs;

	KERNEL_FUNC void operator()(idx_t pair_idx) const {
		if (pair_idx >= num_pairs)
			return;

		int2 pair = neighbor_pairs[pair_idx];

		// Access through ParticleView (like your integrators do)
		Vector3 pos_i = particle_view.pos[pair.x];
		Vector3 pos_j = particle_view.pos[pair.y];
		int type_i = particle_view.type_id[pair.x];
		int type_j = particle_view.type_id[pair.y];

		// Get geometry
		Vector3 r_ij = pbox->wrapDiff(pos_j - pos_i);
		float distance = r_ij.length();
		Vector3 unit_vec = r_ij / distance;

		// Get charges from types
		float qi = particle_types.charge[type_i];
		float qj = particle_types.charge[type_j];

		// Compute
		auto fe = ColumbPotential::compute(r_ij, distance, qi, qj);

		// Apply forces to ParticleView
		Vector3 force_vec = fe.force_magnitude * unit_vec;
		atomic_add(&particle_view.ForceEnergy[pair.x].x, force_vec.x);
		atomic_add(&particle_view.ForceEnergy[pair.x].y, force_vec.y);
		atomic_add(&particle_view.ForceEnergy[pair.x].z, force_vec.z);
		atomic_add(&particle_view.ForceEnergy[pair.y].x, -force_vec.x);
		atomic_add(&particle_view.ForceEnergy[pair.y].y, -force_vec.y);
		atomic_add(&particle_view.ForceEnergy[pair.y].z, -force_vec.z);
	}
};

} // namespace ARBD
#ifdef USE_SYCL
#include <sycl/sycl.hpp>
template<>
struct sycl::is_device_copyable<ARBD::ColumbPotential> : std::true_type {};
template<>
struct sycl::is_device_copyable<ARBD::ColumbForceKernel> : std::true_type {};
#endif
