#pragma once
#include "Analytical.h"
#include "BondGeometry.h"
#include "Interactions/Bonded/TabulatedPotential.h"
#include "Interactions/BondedInteraction.h"
#include "System/PeriodicBox.h"
#include "Types/Types.h"

namespace ARBD {

// ============================================================================
// DEVICE BOND COMPUTER - Uses 2-step approach with BondGeometry
// ============================================================================

template<int BondTypeId>
struct BondComputer {
	DEVICE static inline void compute(idx_t i,
									  DEVICE_PTR(const int2) particle_indices,
									  DEVICE_PTR(Vector3) positions,
									  DEVICE_PTR(Vector3) forces,
									  DEVICE_PTR(float) energies,
									  DEVICE_PTR(const float) params,
									  const PeriodicBox* pbox,
									  bool get_energy,
									  int offset = 0) {

		// Get particle indices
		int2 indices = particle_indices[i + offset];

		// Phase 1: Compute geometry using the excellent BondGeometry approach
		BondGeometry geom = BondGeometry::compute(positions, indices, pbox);

		if (geom.distance < 1e-6f)
			return;

		// Phase 2: Compute force and energy using analytical bond computer
		constexpr int num_params = AnalyticalBondComputer<BondTypeId>::NUM_PARAMS;
		const float* bond_params = params + (i * num_params);
		const ForceEnergy fe =
			AnalyticalBondComputer<BondTypeId>::compute(geom.distance, bond_params);

		// Phase 3: Apply forces using precomputed geometry
		const Vector3 force = geom.unit_vector * fe.force_magnitude;

		atomic_add(&forces[indices.x].x, -force.x);
		atomic_add(&forces[indices.x].y, -force.y);
		atomic_add(&forces[indices.x].z, -force.z);
		atomic_add(&forces[indices.y].x, force.x);
		atomic_add(&forces[indices.y].y, force.y);
		atomic_add(&forces[indices.y].z, force.z);

		if (get_energy) {
			atomic_add(&energies[indices.x], fe.energy * 0.5f);
			atomic_add(&energies[indices.y], fe.energy * 0.5f);
		}
	}
};

// Tabulated bond computer using same 2-step approach
struct TabulatedBondComputer {
	DEVICE static inline void compute(idx_t i,
									  DEVICE_PTR(const int2) indices,
									  DEVICE_PTR(Vector3) positions,
									  DEVICE_PTR(Vector3) forces,
									  DEVICE_PTR(float) energies,
									  DEVICE_PTR(const TabulatedPotential) tables,
									  const PeriodicBox* pbox,
									  bool get_energy,
									  int offset = 0) {
		// Phase 1: Compute geometry
		BondGeometry geom = BondGeometry::compute(positions, indices[i + offset], pbox);

		if (geom.distance < 1e-6f)
			return;

		// Phase 2: Compute force and energy using tabulated potential
		const ForceEnergy fe = TabulatedPotential::compute(geom.distance, &tables[i]);

		// Phase 3: Apply forces using precomputed geometry
		const Vector3 force = geom.unit_vector * fe.force_magnitude;

		atomic_add(&forces[indices->x].x, -force.x);
		atomic_add(&forces[indices->x].y, -force.y);
		atomic_add(&forces[indices->x].z, -force.z);
		atomic_add(&forces[indices->y].x, force.x);
		atomic_add(&forces[indices->y].y, force.y);
		atomic_add(&forces[indices->y].z, force.z);

		if (get_energy) {
			atomic_add(&energies[indices->x], fe.energy * 0.5f);
			atomic_add(&energies[indices->y], fe.energy * 0.5f);
		}
	}
};

// ============================================================================
// DEVICE ANGLE COMPUTER - Uses 2-step approach with AngleGeometry
// ============================================================================

template<int AngleTypeId>
struct AngleComputer {
	DEVICE static inline void compute(idx_t i,
									  DEVICE_PTR(const int3) particle_indices,
									  DEVICE_PTR(Vector3) positions,
									  DEVICE_PTR(Vector3) forces,
									  DEVICE_PTR(float) energies,
									  DEVICE_PTR(const float) params,
									  const PeriodicBox* pbox,
									  bool get_energy,
									  int offset = 0) {

		// Get particle indices
		const int3 indices = particle_indices[i + offset];

		// Phase 1: Compute geometry using the excellent AngleGeometry approach
		AngleGeometry geom = AngleGeometry::compute(positions, indices, pbox);

		// Phase 2: Compute force and energy (placeholder - implement angle potentials)
		// TODO: Implement AnalyticalAngleComputer templates similar to bond computers
		const float force_magnitude = 0.0f; // Placeholder
		const float energy = 0.0f;			// Placeholder

		// Phase 3: Apply forces (would use angle force distribution)
		// TODO: Implement proper angle force distribution

		if (get_energy) {
			atomic_add(&energies[indices.x], energy / 3.0f);
			atomic_add(&energies[indices.y], energy / 3.0f);
			atomic_add(&energies[indices.z], energy / 3.0f);
		}
	}
};

// ============================================================================
// DEVICE DIHEDRAL COMPUTER - Uses 2-step approach with DihedralGeometry
// ============================================================================

template<int DihedralTypeId>
struct DihedralComputer {
	DEVICE static inline void compute(idx_t i,
									  DEVICE_PTR(const int4) particle_indices,
									  DEVICE_PTR(Vector3) positions,
									  DEVICE_PTR(Vector3) forces,
									  DEVICE_PTR(float) energies,
									  DEVICE_PTR(const float) params,
									  const PeriodicBox* pbox,
									  bool get_energy,
									  int offset = 0) {

		// Get particle indices
		const int4 indices = particle_indices[i + offset];

		// Phase 1: Compute geometry using the excellent DihedralGeometry approach
		DihedralGeometry geom = DihedralGeometry::compute(positions, indices, pbox);

		// Phase 2: Compute force and energy (placeholder - implement dihedral potentials)
		// TODO: Implement AnalyticalDihedralComputer templates similar to bond computers
		const float force_magnitude = 0.0f; // Placeholder
		const float energy = 0.0f;			// Placeholder

		// Phase 3: Apply forces (would use dihedral force distribution)
		// TODO: Implement proper dihedral force distribution

		if (get_energy) {
			atomic_add(&energies[indices.x], energy * 0.25f);
			atomic_add(&energies[indices.y], energy * 0.25f);
			atomic_add(&energies[indices.z], energy * 0.25f);
			atomic_add(&energies[indices.w], energy * 0.25f);
		}
	}
};

} // namespace ARBD
