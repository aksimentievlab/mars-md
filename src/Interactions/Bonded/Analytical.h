#pragma once
#include "Header.h"
#include "SimParam.h"
#include "Types/Types.h"

namespace ARBD {

enum class AnalyticalBondType { Harmonic = 0, Morse = 1, WLCSK = 2, FENE = 3 };
enum class AnalyticalAngleType { Harmonic = 0, Morse = 1, WLCSK = 2, FENE = 3 };
enum class AnalyticalDihedralType { Harmonic = 0, Morse = 1, WLCSK = 2, FENE = 3 };

// ============================================================================
// Force Computation Templates - One per bond type
// Using uniform float* parameter arrays for flexibility
// ============================================================================

template<int TypeId>
struct AnalyticalBondComputer;

template<>
struct AnalyticalBondComputer<0> {
	static constexpr int NUM_PARAMS = 2;

	DEVICE static inline float compute(float distance, const float* params) {
		const float k = params[0];	// Spring constant
		const float r0 = params[1]; // Equilibrium distance
		return -k * (distance - r0);
	}
};

// Morse Bond: F = 2*D0*a*exp(-a(r-r0))*[1-exp(-a(r-r0))]
// Parameters: [D0, a, r0]
template<>
struct AnalyticalBondComputer<1> {
	static constexpr int NUM_PARAMS = 3;

	DEVICE static inline float compute(float distance, const float* params) {
		const float D0 = params[0]; // Dissociation energy
		const float a = params[1];	// Width parameter
		const float r0 = params[2]; // Equilibrium distance
		const float exp_term = expf(-a * (distance - r0));
		return 2.0f * D0 * a * exp_term * (1.0f - exp_term);
	}
};

// WLCSK Bond (Worm-Like Chain with Shear and Kink)
// Parameters: [d, lp, kT]
template<>
struct AnalyticalBondComputer<2> {
	static constexpr int NUM_PARAMS = 3;

	DEVICE static inline float compute(float distance, const float* params) {
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
		const float u =
			kT * nk *
			(a1 / (1.0f - q2) - a2 * logf(1.0f - q2) + a3 * q2 - 0.5f * a4 * q2 * (q2 - 2.0f));
		return u;
	}
};

// FENE Bond: F = -k*r*(1-r/r0)
template<>
struct AnalyticalBondComputer<3> {
	static constexpr int NUM_PARAMS = 2;

	DEVICE static inline float compute(float distance, const float* params) {
		const float k = params[0];	// Spring constant
		const float r0 = params[1]; // Equilibrium distance
		return -k * distance * (1.0f - distance / r0);
	}
};

template<int TypeId>
struct AnalyticalBondFunctor {
	HOST DEVICE void operator()(idx_t i,
								DEVICE_PTR(Vector3) positions,
								DEVICE_PTR(Vector3) forces,
								DEVICE_PTR(const int2) particle_indices,
								DEVICE_PTR(const float) params, // Flat parameter array
								int offset = 0) const {
		// Get particle indices
		const int p1 = particle_indices[i + offset].x;
		const int p2 = particle_indices[i + offset].y;

		// Compute bond vector and distance
		const Vector3 r_ij = positions[p2] - positions[p1];
		const float distance = r_ij.length();

		if (distance < 1e-6f)
			return; // Avoid division by zero

		// Get parameters for this bond
		constexpr int num_params = AnalyticalBondComputer<TypeId>::NUM_PARAMS;
		const float* bond_params = params + (i * num_params);

		// Compute force magnitude using specialized template
		const float force_magnitude =
			AnalyticalBondComputer<TypeId>::compute(distance, bond_params);

		// Apply force atomically
		const Vector3 force = (r_ij / distance) * force_magnitude;
		atomic_add(&forces[p1].x, -force.x);
		atomic_add(&forces[p1].y, -force.y);
		atomic_add(&forces[p1].z, -force.z);
		atomic_add(&forces[p2].x, force.x);
		atomic_add(&forces[p2].y, force.y);
		atomic_add(&forces[p2].z, force.z);
	}
};

struct AnalyticalAngleFunctor {
	HOST DEVICE void operator()(idx_t i,
								DEVICE_PTR(Vector3) positions,
								DEVICE_PTR(Vector3) forces,
								DEVICE_PTR(Vector3) particle_indices,
								DEVICE_PTR(const float) params, // Flat parameter array
								int offset = 0) const {
		const int p1 = particle_indices[i + offset].x;
		const int p2 = particle_indices[i + offset].y;
		const int p3 = particle_indices[i + offset].z;

		const Vector3 r_ij = positions[p2] - positions[p1];
		const Vector3 r_ik = positions[p3] - positions[p1];
		const float distance_ij = r_ij.length();
		const float distance_ik = r_ik.length();
	};
}; // namespace ARBD;

struct AnalyticalDihedralFunctor {
	HOST DEVICE void operator()(idx_t i,
								DEVICE_PTR(Vector3) positions,
								DEVICE_PTR(Vector3) forces,
								DEVICE_PTR(Vector3) particle_indices,
								DEVICE_PTR(const float) params, // Flat parameter array
								int offset = 0) const {
		const int p1 = particle_indices[i + offset].x;
		const int p2 = particle_indices[i + offset].y;
		const int p3 = particle_indices[i + offset].z;
		const int p4 = particle_indices[i + offset].w;

		const Vector3 r_ij = positions[p2] - positions[p1];
		const Vector3 r_ik = positions[p3] - positions[p1];
		const float distance_ij = r_ij.length();
		const float distance_ik = r_ik.length();
	};
}; // namespace ARBD;
} // namespace ARBD
