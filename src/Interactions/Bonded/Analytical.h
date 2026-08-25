#pragma once
#include "../Interactions.h"
#include "Header.h"
#include "SimParam.h"
#include "Types/Math.h"
#include "Types/Types.h"

namespace MARS {

// ============================================================================
// Force Computation Templates - One per bond type
// Using uniform mars_real* parameter arrays for flexibility
// ============================================================================

template<int TypeId>
struct AnalyticalForceComputer;

template<>
struct AnalyticalForceComputer<0> {
	static constexpr int NUM_PARAMS = 2;

	DEVICE static inline ScalarForceEnergy compute(mars_real distance, const mars_real* params) {
		const mars_real k = params[0];	// Spring constant
		const mars_real r0 = params[1]; // Equilibrium distance
		mars_real energy = mars_real(0.5) * k * (distance - r0) * (distance - r0);
		mars_real force = -k * (distance - r0);
		return ScalarForceEnergy{Vec2<mars_real>(force, energy)};
	}
};

/// Morse: V(r) = D0*[1-exp(-a(r-r0))]^2, params [D0, a, r0]. See dev_notes.md.
template<>
struct AnalyticalForceComputer<1> {
	static constexpr int NUM_PARAMS = 3;

	DEVICE static inline ScalarForceEnergy compute(mars_real distance, const mars_real* params) {
		const mars_real D0 = params[0]; // Dissociation energy
		const mars_real a = params[1];	// Width parameter
		const mars_real r0 = params[2]; // Equilibrium distance
		const mars_real exp_term = math::exp(-a * (distance - r0));
		const mars_real one_minus_exp = mars_real(1.0) - exp_term;
		mars_real force = -mars_real(2.0) * D0 * a * exp_term * one_minus_exp;
		mars_real energy = D0 * one_minus_exp * one_minus_exp;
		return ScalarForceEnergy{Vec2<mars_real>(force, energy)};
	}
};

// FENE Force: F = -k*r*(1-r/r0)
template<>
struct AnalyticalForceComputer<2> {
	static constexpr int NUM_PARAMS = 2;

	DEVICE static inline ScalarForceEnergy compute(mars_real distance, const mars_real* params) {
		const mars_real k = params[0];	// Spring constant
		const mars_real r0 = params[1]; // Equilibrium distance
		mars_real force = -k * distance * (mars_real(1.0) - distance / r0);
		mars_real energy =
			mars_real(0.5) * k * (distance - r0) * (distance - r0); // TODO: check if this is correct
		return ScalarForceEnergy{Vec2<mars_real>(force, energy)};
	}
};

// Half Harmonic Force: F = -k*(r-r0) for r > r0, 0 for r <= r0
template<>
struct AnalyticalForceComputer<3> {
	static constexpr int NUM_PARAMS = 2;

	DEVICE static inline ScalarForceEnergy compute(mars_real distance, const mars_real* params) {
		const mars_real k = params[0];	// Spring constant
		const mars_real r0 = params[1]; // Equilibrium distance
		mars_real force = distance > r0 ? -k * (distance - r0) : mars_real(0.0);
		mars_real energy = distance > r0 ? mars_real(0.5) * k * (distance - r0) * (distance - r0) : mars_real(0.0);
		return ScalarForceEnergy{Vec2<mars_real>(force, energy)};
	}
};
// WLCSK Force (Worm-Like Chain with Shear and Kink)
// Parameters: [d, lp, kT]
template<>
struct AnalyticalForceComputer<4> {
	static constexpr int NUM_PARAMS = 3;

	DEVICE static inline ScalarForceEnergy compute(mars_real distance, const mars_real* params) {
		const mars_real d = params[0];	// Contour length
		const mars_real lp = params[1]; // Persistence length
		const mars_real kT = params[2]; // Thermal energy

		const mars_real nk = distance / (mars_real(2.0) * lp);
		const mars_real q2 = (distance / d) * (distance / d);
		const mars_real a1 = mars_real(1.0);
		const mars_real a2 = -mars_real(7.0) / (mars_real(2.0) * nk);
		const mars_real a3 = mars_real(3.0) / mars_real(32.0) - mars_real(3.0) / (mars_real(8.0) * nk) - mars_real(6.0) / (mars_real(4.0) * nk * nk);
		const mars_real p0 = mars_real(13.0) / mars_real(32.0);
		const mars_real p1 = mars_real(3.4719);
		const mars_real p2 = mars_real(2.5064);
		const mars_real p3 = -mars_real(1.2906);
		const mars_real p4 = mars_real(0.6482);
		const mars_real a4 = (p0 + p1 / (mars_real(2.0) * nk) + p2 / (mars_real(4.0) * nk * nk)) /
						 (mars_real(1.0) + p3 / (mars_real(2.0) * nk) + p4 / (mars_real(4.0) * nk * nk));
		const mars_real force_magnitude =
			kT * nk *
			(a1 / (mars_real(1.0) - q2) - a2 * math::log(mars_real(1.0) - q2) + a3 * q2 - mars_real(0.5) * a4 * q2 * (q2 - mars_real(2.0)));
		const mars_real energy = kT * nk * (mars_real(1.0) - q2); // TODO: check if this is correct
		return ScalarForceEnergy{Vec2<mars_real>(force_magnitude, energy)};
	}
};
} // namespace MARS
#ifdef USE_SYCL
#include <sycl/sycl.hpp>
template<int T>
struct sycl::is_device_copyable<MARS::AnalyticalForceComputer<T>> : std::true_type {};
#endif
