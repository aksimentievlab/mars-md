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
// Debye-Huckel (screened Coulomb)
// ============================================================================
/**
 * @brief Screened Coulomb: U = A exp(-r/lambda) / r, A = COULOMB qi qj / eps
 * @param screen_length Debye length lambda (Angstrom)
 * @param epsilon Relative dielectric
 */
struct DebyeHuckelPotential {
	arbd_real screen_length = arbd_real(10);
	arbd_real epsilon = arbd_real(80);

	DEVICE ScalarForceEnergy compute(arbd_real distance, arbd_real qi, arbd_real qj) const {
		const arbd_real inv_r = arbd_real(1) / distance;
		const arbd_real a = qi * qj * constants::COULOMB / epsilon;
		const arbd_real screen = math::exp(-distance / screen_length);
		const arbd_real energy = a * screen * inv_r;
		// F = -dU/dr = A exp(-r/l) (1/r^2 + 1/(l r))
		const arbd_real force = energy * (inv_r + arbd_real(1) / screen_length);
		return ScalarForceEnergy{float2{force, energy}};
	}
};

// ============================================================================
// ONC electrostatics (distance-dependent sigmoidal dielectric)
// ============================================================================
/**
 * @brief U = COULOMB qi qj exp(-kappa r) / (eps(r) r), with
 *        eps(r) = Sz (1 - (r/z)^2 e^(r/z) / (e^(r/z) - 1)^2)
 * @param kappa Inverse screening length (1/Angstrom)
 * @param sz Bulk dielectric plateau
 * @param z Sigmoid width (Angstrom)
 * @note eps(0) is singular; distance is floored at MIN_DISTANCE, matching the
 *       reference implementation's d = 1e-2 guard.
 */
struct OnckElecPotential {
	static constexpr arbd_real MIN_DISTANCE = arbd_real(1e-2);

	arbd_real kappa = arbd_real(0.1);
	arbd_real sz = arbd_real(80);
	arbd_real z = arbd_real(6.86);

	/// Below this, eps -> Sz v^2/3 and deps/dr -> Sz v/(3z); avoids 0/0 at r=0.
	static constexpr arbd_real SMALL_V = arbd_real(1e-5);

	/// @brief Sigmoidal dielectric, eps = Sz (sinh v - v)(sinh v + v)/sinh^2 v
	DEVICE arbd_real dielectric(arbd_real r) const {
		const arbd_real v = r / (arbd_real(2) * z);
		if (v < SMALL_V)
			return sz * v * v / arbd_real(3);
		const arbd_real s = math::sinh(v);
		return sz * math::sinh_minus_x(v) * (s + v) / (s * s);
	}

	/// @brief d(eps)/dr = Sz v (v cosh v - sinh v) / (z sinh^3 v)
	DEVICE arbd_real dielectric_deriv(arbd_real r) const {
		const arbd_real v = r / (arbd_real(2) * z);
		if (v < SMALL_V)
			return sz * v / (arbd_real(3) * z);
		const arbd_real s = math::sinh(v);
		return sz * v * math::x_cosh_minus_sinh(v) / (z * s * s * s);
	}

	DEVICE ScalarForceEnergy compute(arbd_real distance, arbd_real qi, arbd_real qj) const {
		const arbd_real r = distance < MIN_DISTANCE ? MIN_DISTANCE : distance;
		const arbd_real a = qi * qj * constants::COULOMB;
		const arbd_real screen = math::exp(-kappa * r);
		const arbd_real eps = dielectric(r);
		const arbd_real energy = a * screen / (eps * r);
		// F = -dU/dr = A exp(-kr) [ k/(eps r) + (eps' r + eps)/(eps^2 r^2) ]
		const arbd_real eps_p = dielectric_deriv(r);
		const arbd_real force =
			a * screen * (kappa / (eps * r) + (eps_p * r + eps) / (eps * eps * r * r));
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
		ScalarForceEnergy fe = ColumbPotential::compute(r_ij, distance, qi, qj);

		// Apply forces to ParticleView. unit_vec points from x to y and
		// force_magnitude is +dU/dr negated, i.e. positive when the pair
		// repels, so the repelled x must move *away* from y.
		Vector3 force_vec = fe.force_magnitude * unit_vec;
		atomic_add(&particle_view.ForceEnergy[pair.x].x, -force_vec.x);
		atomic_add(&particle_view.ForceEnergy[pair.x].y, -force_vec.y);
		atomic_add(&particle_view.ForceEnergy[pair.x].z, -force_vec.z);
		atomic_add(&particle_view.ForceEnergy[pair.y].x, force_vec.x);
		atomic_add(&particle_view.ForceEnergy[pair.y].y, force_vec.y);
		atomic_add(&particle_view.ForceEnergy[pair.y].z, force_vec.z);
	}
};

} // namespace ARBD
#ifdef USE_SYCL
#include <sycl/sycl.hpp>
template<>
struct sycl::is_device_copyable<ARBD::ColumbPotential> : std::true_type {};
template<>
struct sycl::is_device_copyable<ARBD::ColumbForceKernel> : std::true_type {};
template<>
struct sycl::is_device_copyable<ARBD::DebyeHuckelPotential> : std::true_type {};
template<>
struct sycl::is_device_copyable<ARBD::OnckElecPotential> : std::true_type {};
#endif
