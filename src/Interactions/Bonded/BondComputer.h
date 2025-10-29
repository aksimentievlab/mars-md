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
		const int2& indices = particle_indices[i + offset];
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
									  DEVICE_PTR(const int2) particle_indices,
									  DEVICE_PTR(Vector3) positions,
									  DEVICE_PTR(Vector3) forces,
									  DEVICE_PTR(float) energies,
									  DEVICE_PTR(const TabulatedPotential) tables,
									  const PeriodicBox* pbox,
									  bool get_energy,
									  int offset = 0) {
		// Phase 1: Compute geometry
		const int2& indices = particle_indices[i + offset];
		BondGeometry geom = BondGeometry::compute(positions, indices, pbox);

		if (geom.distance < 1e-6f)
			return;

		// Phase 2: Compute force and energy using tabulated potential
		const ForceEnergy fe = TabulatedPotential::compute(geom.distance, &tables[i]);

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

// ============================================================================
// DEVICE ANGLE COMPUTER - Uses 2-step approach with AngleGeometry
// ============================================================================

struct TabulatedAngleComputer {
	DEVICE static inline void compute(idx_t i,
									  DEVICE_PTR(const int3) particle_indices,
									  DEVICE_PTR(Vector3) positions,
									  DEVICE_PTR(Vector3) forces,
									  DEVICE_PTR(float) energies,
									  DEVICE_PTR(const TabulatedPotential) tables,
									  const PeriodicBox* pbox,
									  bool get_energy,
									  int offset = 0) {

		// Get particle indices
		const int3 indices = particle_indices[i + offset];

		// Phase 1: Compute geometry using the excellent AngleGeometry approach
		AngleGeometry geom = AngleGeometry::compute(positions, indices, pbox);

		// Phase 2: Compute force and energy (placeholder - implement angle potentials)
		const ForceEnergy fe = TabulatedPotential::compute(geom.angle, &tables[i]);

		// Phase 3: Apply forces using precomputed geometry
		const Vector3 force1 = geom.ab.cross(geom.bc) * fe.force_magnitude;
		const Vector3 force3 = geom.bc.cross(geom.ab) * fe.force_magnitude;

		atomic_add(&forces[indices.x].x, -force1.x);
		atomic_add(&forces[indices.x].y, -force1.y);
		atomic_add(&forces[indices.x].z, -force1.z);
		atomic_add(&forces[indices.y].x, force1.x + force3.x);
		atomic_add(&forces[indices.y].y, force1.y + force3.y);
		atomic_add(&forces[indices.y].z, force1.z + force3.z);
		atomic_add(&forces[indices.z].x, -force3.x);
		atomic_add(&forces[indices.z].y, -force3.y);
		atomic_add(&forces[indices.z].z, -force3.z);

		if (get_energy) {
			atomic_add(&energies[indices.x], fe.energy * 0.3333333333f);
			atomic_add(&energies[indices.y], fe.energy * 0.3333333333f);
			atomic_add(&energies[indices.z], fe.energy * 0.3333333333f);
		}
	}
};

// ============================================================================
// DEVICE DIHEDRAL COMPUTER - Uses 2-step approach with DihedralGeometry
// ============================================================================

struct TabulatedDihedralComputer {
	DEVICE static inline void compute(idx_t i,
									  DEVICE_PTR(const int4) particle_indices,
									  DEVICE_PTR(Vector3) positions,
									  DEVICE_PTR(Vector3) forces,
									  DEVICE_PTR(float) energies,
									  DEVICE_PTR(const TabulatedPotential) tables,
									  const PeriodicBox* pbox,
									  bool get_energy,
									  int offset = 0) {

		// Get particle indices
		const int4 indices = particle_indices[i + offset];

		// Phase 1: Compute geometry using the excellent DihedralGeometry approach
		DihedralGeometry geom = DihedralGeometry::compute(positions, indices, pbox);

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

// ============================================================================
// PRODUCT POTENTIAL COMPUTER
// ============================================================================

/**
 * @brief Computer for product potentials (coupled bonded interactions)
 *
 * This template handles various types of product potentials:
 * - BondAngle: 2 angles + 1 bond (4 particles)
 * - AngleAngle: 2 angles sharing a bond (4 particles)
 * - BondBond: 2 bonds sharing an atom (3 particles)
 */
// template<int ProductTypeId>
// struct ProductPotentialComputer {
// 	DEVICE static inline void compute(idx_t i,
// 									  DEVICE_PTR(const int4) particle_indices,
// 									  DEVICE_PTR(Vector3) positions,
// 									  DEVICE_PTR(Vector3) forces,
// 									  DEVICE_PTR(float) energies,
// 									  DEVICE_PTR(const TabulatedPotential) angle_tables_1,
// 									  DEVICE_PTR(const TabulatedPotential) bond_tables,
// 									  DEVICE_PTR(const TabulatedPotential) angle_tables_2,
// 									  DEVICE_PTR(const int)
// 										  potential_indices, // 3 indices per product
// 									  const PeriodicBox* pbox,
// 									  bool get_energy,
// 									  int offset = 0) {
// 		// Get particle indices
// 		int4 indices = particle_indices[i + offset];
// 		int3 pot_idx = int3(potential_indices[i * 3 + 0],
// 							potential_indices[i * 3 + 1],
// 							potential_indices[i * 3 + 2]);

// 		// Compute geometry based on product type
// 		ProductPotentialGeometry geom;
// 		if constexpr (ProductTypeId == 0) { // BondAngle
// 			geom = ProductPotentialGeometry::compute_bond_angle(positions, indices, pbox);
// 		} else if constexpr (ProductTypeId == 1) { // AngleAngle
// 			geom = ProductPotentialGeometry::compute_angle_angle(positions, indices, pbox);
// 		} else if constexpr (ProductTypeId == 2) { // BondBond
// 			// For BondBond, we only need 3 particles
// 			int3 indices_3 = int3(indices.x, indices.y, indices.z);
// 			geom = ProductPotentialGeometry::compute_bond_bond(positions, indices_3, pbox);
// 		}

// 		// Early exit if singular
// 		if (geom.is_singular)
// 			return;

// 		// Compute forces from tabulated potentials
// 		// This matches your legacy computeBondAngle logic
// 		ForceEnergy fe1, fe_bond, fe2;

// 		if constexpr (ProductTypeId == 0) { // BondAngle
// 			fe1 = angle_tables_1[pot_idx.x].compute(geom.angle1.angle, &angle_tables_1[pot_idx.x]);
// 			fe_bond = bond_tables[pot_idx.y].compute(geom.bond.distance, &bond_tables[pot_idx.y]);
// 			fe2 = angle_tables_2[pot_idx.z].compute(geom.angle2.angle, &angle_tables_2[pot_idx.z]);
// 		} else if constexpr (ProductTypeId == 1) { // AngleAngle
// 			fe1 = angle_tables_1[pot_idx.x].compute(geom.angle_a.angle, &angle_tables_1[pot_idx.x]);
// 			fe2 = angle_tables_2[pot_idx.z].compute(geom.angle_b.angle, &angle_tables_2[pot_idx.z]);
// 			fe_bond = {0.0f, 0.0f};				   // No bond term for AngleAngle
// 		} else if constexpr (ProductTypeId == 2) { // BondBond
// 			fe1 = bond_tables[pot_idx.x].compute(geom.bond_a.distance, &bond_tables[pot_idx.x]);
// 			fe2 = bond_tables[pot_idx.y].compute(geom.bond_b.distance, &bond_tables[pot_idx.y]);
// 			fe_bond = {0.0f, 0.0f}; // No third term for BondBond
// 		}

// 		// Apply forces using geometry basis vectors
// 		apply_product_forces(indices, geom, fe1, fe_bond, fe2, forces);

// 		if (get_energy) {
// 			float total_energy = fe1.energy + fe_bond.energy + fe2.energy;
// 			float energy_per_particle = total_energy / 4.0f;
// 			atomic_add(&energies[indices.x], energy_per_particle);
// 			atomic_add(&energies[indices.y], energy_per_particle);
// 			atomic_add(&energies[indices.z], energy_per_particle);
// 			atomic_add(&energies[indices.w], energy_per_particle);
// 		}
// 	}

//   private:
// 	/**
// 	 * @brief Apply forces for BondAngle product potential
// 	 */
// 	DEVICE static inline void apply_bond_angle_forces(int4 indices,
// 													  const ProductPotentialGeometry& geom,
// 													  const ForceEnergy& fe1,	  // angle 1 force
// 													  const ForceEnergy& fe_bond, // bond force
// 													  const ForceEnergy& fe2,	  // angle 2 force
// 													  DEVICE_PTR(Vector3) forces) {
// 		// Apply angle 1 forces (i-j-k)
// 		Vector3 f1_angle1 = geom.angle1.f1_basis * fe1.force_magnitude;
// 		Vector3 f2_angle1 = geom.angle1.f2_basis * fe1.force_magnitude;
// 		Vector3 f3_angle1 = geom.angle1.f3_basis * fe1.force_magnitude;

// 		// Apply bond forces (j-k)
// 		Vector3 f1_bond = -geom.bond.unit_vector * fe_bond.force_magnitude;
// 		Vector3 f2_bond = geom.bond.unit_vector * fe_bond.force_magnitude;

// 		// Apply angle 2 forces (j-k-l)
// 		Vector3 f1_angle2 = geom.angle2.f1_basis * fe2.force_magnitude;
// 		Vector3 f2_angle2 = geom.angle2.f2_basis * fe2.force_magnitude;
// 		Vector3 f3_angle2 = geom.angle2.f3_basis * fe2.force_magnitude;

// 		// Combine forces (particles j and k get forces from multiple terms)
// 		atomic_add(&forces[indices.x], f1_angle1); // i: angle1 only
// 		atomic_add(&forces[indices.y],
// 				   f2_angle1 + f1_bond + f1_angle2); // j: angle1 + bond + angle2
// 		atomic_add(&forces[indices.z],
// 				   f3_angle1 + f2_bond + f2_angle2); // k: angle1 + bond + angle2
// 		atomic_add(&forces[indices.w], f3_angle2);	 // l: angle2 only
// 	}

// 	/**
// 	 * @brief Apply forces for AngleAngle product potential
// 	 */
// 	DEVICE static inline void apply_angle_angle_forces(int4 indices,
// 													   const ProductPotentialGeometry& geom,
// 													   const ForceEnergy& fe1, // angle 1 force
// 													   const ForceEnergy&,	   // unused
// 													   const ForceEnergy& fe2, // angle 2 force
// 													   DEVICE_PTR(Vector3) forces) {
// 		// Apply angle 1 forces (i-j-k)
// 		Vector3 f1_angle1 = geom.angle_a.f1_basis * fe1.force_magnitude;
// 		Vector3 f2_angle1 = geom.angle_a.f2_basis * fe1.force_magnitude;
// 		Vector3 f3_angle1 = geom.angle_a.f3_basis * fe1.force_magnitude;

// 		// Apply angle 2 forces (j-k-l)
// 		Vector3 f1_angle2 = geom.angle_b.f1_basis * fe2.force_magnitude;
// 		Vector3 f2_angle2 = geom.angle_b.f2_basis * fe2.force_magnitude;
// 		Vector3 f3_angle2 = geom.angle_b.f3_basis * fe2.force_magnitude;

// 		// Combine forces
// 		atomic_add(&forces[indices.x], f1_angle1);			   // i: angle1 only
// 		atomic_add(&forces[indices.y], f2_angle1 + f1_angle2); // j: angle1 + angle2
// 		atomic_add(&forces[indices.z], f3_angle1 + f2_angle2); // k: angle1 + angle2
// 		atomic_add(&forces[indices.w], f3_angle2);			   // l: angle2 only
// 	}

// 	/**
// 	 * @brief Apply forces for BondBond product potential
// 	 */
// 	DEVICE static inline void apply_bond_bond_forces(int4 indices,
// 													 const ProductPotentialGeometry& geom,
// 													 const ForceEnergy& fe1, // bond 1 force
// 													 const ForceEnergy&,	 // unused
// 													 const ForceEnergy& fe2, // bond 2 force
// 													 DEVICE_PTR(Vector3) forces) {
// 		// Apply bond 1 forces (i-j)
// 		Vector3 f1_bond1 = -geom.bond_a.unit_vector * fe1.force_magnitude;
// 		Vector3 f2_bond1 = geom.bond_a.unit_vector * fe1.force_magnitude;

// 		// Apply bond 2 forces (j-k)
// 		Vector3 f1_bond2 = -geom.bond_b.unit_vector * fe2.force_magnitude;
// 		Vector3 f2_bond2 = geom.bond_b.unit_vector * fe2.force_magnitude;

// 		// Combine forces
// 		atomic_add(&forces[indices.x], f1_bond1);			 // i: bond1 only
// 		atomic_add(&forces[indices.y], f2_bond1 + f1_bond2); // j: bond1 + bond2
// 		atomic_add(&forces[indices.z], f2_bond2);			 // k: bond2 only
// 	}

// 	/**
// 	 * @brief Dispatch to appropriate force application method
// 	 */
// 	DEVICE static inline void apply_product_forces(int4 indices,
// 												   const ProductPotentialGeometry& geom,
// 												   const ForceEnergy& fe1,
// 												   const ForceEnergy& fe_bond,
// 												   const ForceEnergy& fe2,
// 												   DEVICE_PTR(Vector3) forces) {
// 		if constexpr (ProductTypeId == 0) { // BondAngle
// 			apply_bond_angle_forces(indices, geom, fe1, fe_bond, fe2, forces);
// 		} else if constexpr (ProductTypeId == 1) { // AngleAngle
// 			apply_angle_angle_forces(indices, geom, fe1, fe_bond, fe2, forces);
// 		} else if constexpr (ProductTypeId == 2) { // BondBond
// 			apply_bond_bond_forces(indices, geom, fe1, fe_bond, fe2, forces);
// 		}
// 	}
// };

} // namespace ARBD
