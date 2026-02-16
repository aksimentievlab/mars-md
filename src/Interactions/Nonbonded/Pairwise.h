#pragma once
#include "../Interactions.h"
#include "Header.h"
#include "Interactions/TabulatedPotential.h"
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
 * @brief Tabulated bond force computer
 */
struct TabulatedNonBondedComputer {
	// Members
	DEVICE_PTR(const int2) particle_indices;
	DEVICE_PTR(Vector3) positions;
	DEVICE_PTR(Vector3) force_energy;
	DEVICE_PTR(const TabulatedPotential) tables;
	const PeriodicBox* pbox;
	bool get_energy;
	idx_t num_pairs;

	// Constructor
	TabulatedNonBondedComputer(DEVICE_PTR(const int2) indices,
							   DEVICE_PTR(Vector3) pos,
							   DEVICE_PTR(Vector3) fe,
							   DEVICE_PTR(const TabulatedPotential) tabs,
							   const PeriodicBox* box,
							   bool energy,
							   idx_t n)
		: particle_indices(indices), positions(pos), force_energy(fe), tables(tabs), pbox(box),
		  get_energy(energy), num_pairs(n) {}

	// Kernel operator
	DEVICE void operator()(idx_t i) const {
		if (i >= num_pairs)
			return;

		const int2& indices = particle_indices[i];

		// Phase 1: Compute geometry
		CalcDistance geom = CalcDistance::compute(positions, indices, pbox);
		if (geom.distance < 1e-6f)
			return;

		// Phase 2: Lookup force from tabulated potential
		const ScalarForceEnergy fe = TabulatedPotential::compute(geom.distance, &tables[i]);

		// Phase 3: Apply forces
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
} // namespace ARBD
