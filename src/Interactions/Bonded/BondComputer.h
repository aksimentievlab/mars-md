#pragma once
#include "Analytical.h"
#include "Interactions/BondedInteraction.h"
#include "Types/Types.h"

namespace ARBD {

// ============================================================================
// UNIFIED BOND COMPUTATION INTERFACE
// ============================================================================

// Forward declarations
struct BondGeometry;
struct AngleGeometry;
struct DihedralGeometry;

// ============================================================================
// GEOMETRY COMPUTATION STRUCTURES
// ============================================================================

struct BondGeometry {
	Vector3 r_ij;		 // Bond vector
	float distance;		 // Bond distance
	Vector3 unit_vector; // Normalized direction

	DEVICE static BondGeometry
	compute(const Vector3* positions, const int2& particle_indices, const PeriodicBox* pbox) {
		BondGeometry geom;
		geom.r_ij = pbox->wrapDiff(positions[particle_indices.y] - positions[particle_indices.x]);
		geom.distance = geom.r_ij.length();
		if (geom.distance > 1e-6f) {
			geom.unit_vector = geom.r_ij / geom.distance;
		} else {
			geom.unit_vector = Vector3(0.0f);
		}
		return geom;
	}
};

struct AngleGeometry {
	Vector3 ab, bc, ac; // Vectors
	float angle;		// Computed angle
	float cos_angle;	// Cosine of angle
	float sin_angle;	// Sine of angle

	DEVICE static AngleGeometry
	compute(const Vector3* positions, const int4& particle_indices, const PeriodicBox* pbox) {
		AngleGeometry geom;
		geom.ab = pbox->wrapDiff(positions[particle_indices.y] - positions[particle_indices.x]);
		geom.bc = pbox->wrapDiff(positions[particle_indices.z] - positions[particle_indices.y]);
		geom.ac = pbox->wrapDiff(positions[particle_indices.z] - positions[particle_indices.x]);

		// Compute angle using law of cosines
		float distab2 = geom.ab.length2();
		float distbc2 = geom.bc.length2();
		float distac2 = geom.ac.length2();

		geom.cos_angle = (distab2 + distbc2 - distac2) * 0.5f / (sqrt(distab2) * sqrt(distbc2));

		// Clamp cosine to valid range
		if (geom.cos_angle < -1.0f)
			geom.cos_angle = -1.0f;
		if (geom.cos_angle > 1.0f)
			geom.cos_angle = 1.0f;

		geom.angle = acos(geom.cos_angle);
		geom.sin_angle = sqrt(1.0f - geom.cos_angle * geom.cos_angle);

		return geom;
	}
};

struct DihedralGeometry {
	Vector3 ab, bc, cd;			// Vectors
	float dihedral_angle;		// Computed dihedral angle
	Vector3 crossABC, crossBCD; // Cross products

	DEVICE static DihedralGeometry
	compute(const Vector3* positions, const int4& particle_indices, const PeriodicBox* pbox) {
		DihedralGeometry geom;
		geom.ab = pbox->wrapDiff(positions[particle_indices.y] - positions[particle_indices.x]);
		geom.bc = pbox->wrapDiff(positions[particle_indices.z] - positions[particle_indices.y]);
		geom.cd = pbox->wrapDiff(positions[particle_indices.w] - positions[particle_indices.z]);

		geom.crossABC = geom.ab.cross(geom.bc);
		geom.crossBCD = geom.bc.cross(geom.cd);
		Vector3 crossX = geom.bc.cross(geom.crossABC);

		float cos_phi =
			geom.crossABC.dot(geom.crossBCD) / (geom.crossABC.length() * geom.crossBCD.length());
		float sin_phi = crossX.dot(geom.crossBCD) / (crossX.length() * geom.crossBCD.length());

		geom.dihedral_angle = -atan2(sin_phi, cos_phi);

		return geom;
	}
};

// ============================================================================
// FORCE COMPUTATION INTERFACE
// ============================================================================

// Base interface for force computation
template<typename PotentialType>
struct BondComputer {
	DEVICE static inline float compute_force(float distance, const PotentialType& potential);
	DEVICE static inline float compute_energy(float distance, const PotentialType& potential);
};

// Specialization for tabulated potentials
template<>
struct BondComputer<BondedPotential> {
	DEVICE static inline float compute_force(float distance, const BondedPotential* table) {
		// Table lookup with linear interpolation
		float t = distance * table->step_inv;
		int home = int(floorf(t));
		t = t - home;

		if (home < 0)
			home = 0;
		if (home >= table->size)
			home = table->size - 1;

		int home1 = (home + 1 >= table->size) ? table->size - 1 : home + 1;

		float U0 = table->pot[home];
		float dU = table->pot[home1] - U0;

		return -dU * table->step_inv; // Force magnitude
	}

	DEVICE static inline float compute_energy(float distance, const BondedPotential* table) {
		// Table lookup with linear interpolation
		float t = distance * table->step_inv;
		int home = int(floorf(t));
		t = t - home;

		if (home < 0)
			home = 0;
		if (home >= table->size)
			home = table->size - 1;

		int home1 = (home + 1 >= table->size) ? table->size - 1 : home + 1;

		float U0 = table->pot[home];
		float dU = table->pot[home1] - U0;

		return dU * t + U0; // Energy
	}
};

// Specialization for analytical potentials - this is a placeholder
// The actual analytical bond computation is handled by AnalyticalBondComputer<TypeId>
template<>
struct BondComputer<AnalyticalBondType> {
	DEVICE static inline float compute_force(float distance, const float* params) {
		// This should not be called directly - use AnalyticalBondComputer<TypeId> instead
		return 0.0f;
	}

	DEVICE static inline float compute_energy(float distance, const float* params) {
		// This should not be called directly - use AnalyticalBondComputer<TypeId> instead
		return 0.0f;
	}
};

// ============================================================================
// UNIFIED KERNEL FUNCTORS
// ============================================================================

// Bond computation functor
struct BondFunctor {
	DEVICE inline void operator()(const DeviceBond& bond,
								  const BondedPotential* potentials,
								  const Vector3* positions,
								  const PeriodicBox* pbox,
								  Vector3* forces,
								  float* energy,
								  bool get_energy,
								  size_t num_local_particles) const {

		// Skip bonds involving ghost particles (only process local-local bonds)
		// Ghost particles are at the end: [capacity-num_ghost_particles ... capacity-1]
		if (bond.particle_indices.x >= num_local_particles ||
			bond.particle_indices.y >= num_local_particles) {
			return;
		}

		// Phase 1: Compute geometry
		BondGeometry geom = BondGeometry::compute(positions, bond.particle_indices, pbox);

		if (geom.distance < 1e-6f)
			return;

		// Phase 2: Compute force magnitude
		const BondedPotential* potential = &potentials[bond.function_index];
		const float force_magnitude =
			BondComputer<BondedPotential>::compute_force(geom.distance, potential);

		// Phase 3: Apply forces
		const Vector3 force = geom.unit_vector * force_magnitude;
		atomic_add(&forces[bond.particle_indices.x].x, -force.x);
		atomic_add(&forces[bond.particle_indices.x].y, -force.y);
		atomic_add(&forces[bond.particle_indices.x].z, -force.z);
		atomic_add(&forces[bond.particle_indices.y].x, force.x);
		atomic_add(&forces[bond.particle_indices.y].y, force.y);
		atomic_add(&forces[bond.particle_indices.y].z, force.z);

		if (get_energy) {
			const float e = BondComputer<BondedPotential>::compute_energy(geom.distance, potential);
			atomic_add(&energy[bond.particle_indices.x], e * 0.5f);
			atomic_add(&energy[bond.particle_indices.y], e * 0.5f);
		}
	}
};

// Angle computation functor
struct AngleFunctor {
	DEVICE inline void operator()(const DeviceAngle& angle,
								  const BondedPotential* potentials,
								  const Vector3* positions,
								  const PeriodicBox* pbox,
								  Vector3* forces,
								  float* energy,
								  bool get_energy,
								  size_t num_local_particles) const {

		// Skip angles involving ghost particles (only process local-local-local angles)
		if (angle.particle_indices.x >= num_local_particles ||
			angle.particle_indices.y >= num_local_particles ||
			angle.particle_indices.z >= num_local_particles) {
			return;
		}

		// Phase 1: Compute geometry
		AngleGeometry geom = AngleGeometry::compute(positions, angle.particle_indices, pbox);

		// Phase 2: Compute force magnitude
		const BondedPotential* potential = &potentials[angle.function_index];
		const float force_magnitude =
			BondComputer<BondedPotential>::compute_force(geom.angle, potential);

		// Phase 3: Apply forces (using existing angle force calculation)
		// This would use the existing angle force calculation logic
		// from your TabulatedKernels.h

		// Skip angles involving ghost particles (only process local-local angles)
		if (angle.particle_indices.x >= num_local_particles ||
			angle.particle_indices.y >= num_local_particles ||
			angle.particle_indices.z >= num_local_particles) {
			return;
		}

		if (get_energy) {
			const float e = BondComputer<BondedPotential>::compute_energy(geom.angle, potential);
			atomic_add(&energy[angle.particle_indices.x], e * 0.3333333333f);
			atomic_add(&energy[angle.particle_indices.y], e * 0.3333333333f);
			atomic_add(&energy[angle.particle_indices.z], e * 0.3333333333f);
		}
	}
};

// Dihedral computation functor
struct DihedralFunctor {
	DEVICE inline void operator()(const DeviceDihedral& dihedral,
								  const BondedPotential* potentials,
								  const Vector3* positions,
								  const PeriodicBox* pbox,
								  Vector3* forces,
								  float* energy,
								  bool get_energy,
								  size_t num_local_particles) const {

		// Skip dihedrals involving ghost particles (only process local-local-local-local dihedrals)
		if (dihedral.particle_indices.x >= num_local_particles ||
			dihedral.particle_indices.y >= num_local_particles ||
			dihedral.particle_indices.z >= num_local_particles ||
			dihedral.particle_indices.w >= num_local_particles) {
			return;
		}

		// Phase 1: Compute geometry
		DihedralGeometry geom =
			DihedralGeometry::compute(positions, dihedral.particle_indices, pbox);

		// Phase 2: Compute force magnitude
		const BondedPotential* potential = &potentials[dihedral.function_index];
		const float force_magnitude =
			BondComputer<BondedPotential>::compute_force(geom.dihedral_angle, potential);

		// Phase 3: Apply forces (using existing dihedral force calculation)
		// This would use the existing dihedral force calculation logic
		// from your TabulatedKernels.h

		if (get_energy) {
			const float e =
				BondComputer<BondedPotential>::compute_energy(geom.dihedral_angle, potential);
			atomic_add(&energy[dihedral.particle_indices.x], e * 0.25f);
			atomic_add(&energy[dihedral.particle_indices.y], e * 0.25f);
			atomic_add(&energy[dihedral.particle_indices.z], e * 0.25f);
			atomic_add(&energy[dihedral.particle_indices.w], e * 0.25f);
		}
	}
};

} // namespace ARBD
