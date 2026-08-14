#pragma once
#include "../Interactions.h"
#include "Header.h"
#include "SimParam.h"
#include "Types/Math.h"
#include "Types/Types.h"

namespace ARBD {

// ============================================================================
// Force Computation Templates - One per bond type
// Using uniform arbd_real* parameter arrays for flexibility
// ============================================================================

template<int TypeId>
struct AnalyticalForceComputer;

template<>
struct AnalyticalForceComputer<0> {
	static constexpr int NUM_PARAMS = 2;

	DEVICE static inline ScalarForceEnergy compute(arbd_real distance, const arbd_real* params) {
		const arbd_real k = params[0];	// Spring constant
		const arbd_real r0 = params[1]; // Equilibrium distance
		arbd_real energy = arbd_real(0.5) * k * (distance - r0) * (distance - r0);
		arbd_real force = -k * (distance - r0);
		return ScalarForceEnergy{Vec2<arbd_real>(force, energy)};
	}
};

/*Morse potential: V(r) = D0*[1-exp(-a(r-r0))]^2
 * Force: F = -dV/dr = -2*D0*a*exp(-a(r-r0))*[1-exp(-a(r-r0))]
 * (negative for r>r0: attractive, pulling the stretched bond back together -
 * same sign convention as AnalyticalForceComputer<0>'s -k*(distance-r0))
 * Parameters: [D0, a, r0]
 */
template<>
struct AnalyticalForceComputer<1> {
	static constexpr int NUM_PARAMS = 3;

	DEVICE static inline ScalarForceEnergy compute(arbd_real distance, const arbd_real* params) {
		const arbd_real D0 = params[0]; // Dissociation energy
		const arbd_real a = params[1];	// Width parameter
		const arbd_real r0 = params[2]; // Equilibrium distance
		const arbd_real exp_term = math::exp(-a * (distance - r0));
		const arbd_real one_minus_exp = arbd_real(1.0) - exp_term;
		arbd_real force = -arbd_real(2.0) * D0 * a * exp_term * one_minus_exp;
		arbd_real energy = D0 * one_minus_exp * one_minus_exp;
		return ScalarForceEnergy{Vec2<arbd_real>(force, energy)};
	}
};

// FENE Force: F = -k*r*(1-r/r0)
template<>
struct AnalyticalForceComputer<2> {
	static constexpr int NUM_PARAMS = 2;

	DEVICE static inline ScalarForceEnergy compute(arbd_real distance, const arbd_real* params) {
		const arbd_real k = params[0];	// Spring constant
		const arbd_real r0 = params[1]; // Equilibrium distance
		arbd_real force = -k * distance * (arbd_real(1.0) - distance / r0);
		arbd_real energy =
			arbd_real(0.5) * k * (distance - r0) * (distance - r0); // TODO: check if this is correct
		return ScalarForceEnergy{Vec2<arbd_real>(force, energy)};
	}
};

// Half Harmonic Force: F = -k*(r-r0) for r > r0, 0 for r <= r0
template<>
struct AnalyticalForceComputer<3> {
	static constexpr int NUM_PARAMS = 2;

	DEVICE static inline ScalarForceEnergy compute(arbd_real distance, const arbd_real* params) {
		const arbd_real k = params[0];	// Spring constant
		const arbd_real r0 = params[1]; // Equilibrium distance
		arbd_real force = distance > r0 ? -k * (distance - r0) : arbd_real(0.0);
		arbd_real energy = distance > r0 ? arbd_real(0.5) * k * (distance - r0) * (distance - r0) : arbd_real(0.0);
		return ScalarForceEnergy{Vec2<arbd_real>(force, energy)};
	}
};
// WLCSK Force (Worm-Like Chain with Shear and Kink)
// Parameters: [d, lp, kT]
template<>
struct AnalyticalForceComputer<4> {
	static constexpr int NUM_PARAMS = 3;

	DEVICE static inline ScalarForceEnergy compute(arbd_real distance, const arbd_real* params) {
		const arbd_real d = params[0];	// Contour length
		const arbd_real lp = params[1]; // Persistence length
		const arbd_real kT = params[2]; // Thermal energy

		const arbd_real nk = distance / (arbd_real(2.0) * lp);
		const arbd_real q2 = (distance / d) * (distance / d);
		const arbd_real a1 = arbd_real(1.0);
		const arbd_real a2 = -arbd_real(7.0) / (arbd_real(2.0) * nk);
		const arbd_real a3 = arbd_real(3.0) / arbd_real(32.0) - arbd_real(3.0) / (arbd_real(8.0) * nk) - arbd_real(6.0) / (arbd_real(4.0) * nk * nk);
		const arbd_real p0 = arbd_real(13.0) / arbd_real(32.0);
		const arbd_real p1 = arbd_real(3.4719);
		const arbd_real p2 = arbd_real(2.5064);
		const arbd_real p3 = -arbd_real(1.2906);
		const arbd_real p4 = arbd_real(0.6482);
		const arbd_real a4 = (p0 + p1 / (arbd_real(2.0) * nk) + p2 / (arbd_real(4.0) * nk * nk)) /
						 (arbd_real(1.0) + p3 / (arbd_real(2.0) * nk) + p4 / (arbd_real(4.0) * nk * nk));
		const arbd_real force_magnitude =
			kT * nk *
			(a1 / (arbd_real(1.0) - q2) - a2 * math::log(arbd_real(1.0) - q2) + a3 * q2 - arbd_real(0.5) * a4 * q2 * (q2 - arbd_real(2.0)));
		const arbd_real energy = kT * nk * (arbd_real(1.0) - q2); // TODO: check if this is correct
		return ScalarForceEnergy{Vec2<arbd_real>(force_magnitude, energy)};
	}
};
} // namespace ARBD
#ifdef USE_SYCL
#include <sycl/sycl.hpp>
template<int T>
struct sycl::is_device_copyable<ARBD::AnalyticalForceComputer<T>> : std::true_type {};
#endif
