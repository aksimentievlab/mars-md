#pragma once
#include "../Interactions.h"
#include "Header.h"
#include "SimParam.h"
#include "Types/Types.h"

namespace ARBD {

// ============================================================================
// Force Computation Templates - One per bond type
// Using uniform float* parameter arrays for flexibility
// ============================================================================

template<int TypeId>
struct AnalyticalForceComputer;

template<>
struct AnalyticalForceComputer<0> {
	static constexpr int NUM_PARAMS = 2;

	DEVICE static inline ScalarForceEnergy compute(float distance, const float* params) {
		const float k = params[0];	// Spring constant
		const float r0 = params[1]; // Equilibrium distance
		float energy = 0.5f * k * (distance - r0) * (distance - r0);
		float force = -k * (distance - r0);
		return ScalarForceEnergy{float2(force, energy)};
	}
};

// Morse Force: F = 2*D0*a*exp(-a(r-r0))*[1-exp(-a(r-r0))]
// Parameters: [D0, a, r0]
template<>
struct AnalyticalForceComputer<1> {
	static constexpr int NUM_PARAMS = 3;

	DEVICE static inline ScalarForceEnergy compute(float distance, const float* params) {
		const float D0 = params[0]; // Dissociation energy
		const float a = params[1];	// Width parameter
		const float r0 = params[2]; // Equilibrium distance
		const float exp_term = expf(-a * (distance - r0));
		float force = 2.0f * D0 * a * exp_term * (1.0f - exp_term);
		float energy = D0 * (1.0f - exp_term);
		return ScalarForceEnergy{float2(force, energy)};
	}
};

// FENE Force: F = -k*r*(1-r/r0)
template<>
struct AnalyticalForceComputer<2> {
	static constexpr int NUM_PARAMS = 2;

	DEVICE static inline ScalarForceEnergy compute(float distance, const float* params) {
		const float k = params[0];	// Spring constant
		const float r0 = params[1]; // Equilibrium distance
		float force = -k * distance * (1.0f - distance / r0);
		float energy =
			0.5f * k * (distance - r0) * (distance - r0); // TODO: check if this is correct
		return ScalarForceEnergy{float2(force, energy)};
	}
};

// Half Harmonic Force: F = -k*(r-r0) for r > r0, 0 for r <= r0
template<>
struct AnalyticalForceComputer<3> {
	static constexpr int NUM_PARAMS = 2;

	DEVICE static inline ScalarForceEnergy compute(float distance, const float* params) {
		const float k = params[0];	// Spring constant
		const float r0 = params[1]; // Equilibrium distance
		float force = distance > r0 ? -k * (distance - r0) : 0.0f;
		float energy = distance > r0 ? 0.5f * k * (distance - r0) * (distance - r0) : 0.0f;
		return ScalarForceEnergy{float2(force, energy)};
	}
};
// WLCSK Force (Worm-Like Chain with Shear and Kink)
// Parameters: [d, lp, kT]
template<>
struct AnalyticalForceComputer<4> {
	static constexpr int NUM_PARAMS = 3;

	DEVICE static inline ScalarForceEnergy compute(float distance, const float* params) {
		const float d = params[0];	// Contour length
		const float lp = params[1]; // Persistence length
		const float kT = params[2]; // Thermal energy

		const float nk = distance / (2.0f * lp);
		const float q2 = (distance / d) * (distance / d);
		const float a1 = 1.0f;
		const float a2 = -7.0f / (2.0f * nk);
		const float a3 = 3.0f / 32.0f - 3.0f / (8.0f * nk) - 6.0f / (4.0f * nk * nk);
		const float p0 = 13.0f / 32.0f;
		const float p1 = 3.4719f;
		const float p2 = 2.5064f;
		const float p3 = -1.2906f;
		const float p4 = 0.6482f;
		const float a4 = (p0 + p1 / (2.0f * nk) + p2 / (4.0f * nk * nk)) /
						 (1.0f + p3 / (2.0f * nk) + p4 / (4.0f * nk * nk));
		const float force_magnitude =
			kT * nk *
			(a1 / (1.0f - q2) - a2 * logf(1.0f - q2) + a3 * q2 - 0.5f * a4 * q2 * (q2 - 2.0f));
		const float energy = kT * nk * (1.0f - q2); // TODO: check if this is correct
		return ScalarForceEnergy{float2(force_magnitude, energy)};
	}
};
} // namespace ARBD
#ifdef USE_SYCL
#include <sycl/sycl.hpp>
template<int T>
struct sycl::is_device_copyable<ARBD::AnalyticalForceComputer<T>> : std::true_type {};
#endif
