#pragma once

#include "../BondedInteraction.h"
#include <string>
#include <vector>

namespace ARBD {

/**
 * @brief to be deprecated, reference only.
 */

// Forward declaration for legacy BondAngle (from legacy codebase)
struct LegacyBondAngle {
	int ind1, ind2, ind3, ind4;
	std::string angleFileName1;
	std::string bondFileName;
	std::string angleFileName2;
	int tabFileIndex1;
	int tabFileIndex2;
	int tabFileIndex3;

	LegacyBondAngle()
		: ind1(-1), ind2(-1), ind3(-1), ind4(-1), tabFileIndex1(-1), tabFileIndex2(-1),
		  tabFileIndex3(-1) {}

	LegacyBondAngle(int i1,
					int i2,
					int i3,
					int i4,
					const std::string& a1,
					const std::string& b,
					const std::string& a2)
		: ind1(i1), ind2(i2), ind3(i3), ind4(i4), angleFileName1(a1), bondFileName(b),
		  angleFileName2(a2), tabFileIndex1(-1), tabFileIndex2(-1), tabFileIndex3(-1) {}
};

/**
 * @brief Convert legacy BondAngle to ProductPotential
 * @param legacy_bond_angle Legacy BondAngle structure
 * @return ProductPotential equivalent
 */
inline ProductPotential
migrate_bond_angle_to_product_potential(const LegacyBondAngle& legacy_bond_angle) {
	return ProductPotential::from_bond_angle(legacy_bond_angle.ind1,
											 legacy_bond_angle.ind2,
											 legacy_bond_angle.ind3,
											 legacy_bond_angle.ind4,
											 legacy_bond_angle.tabFileIndex1, // angle 1 table index
											 legacy_bond_angle.tabFileIndex2, // bond table index
											 legacy_bond_angle.tabFileIndex3, // angle 2 table index
											 "MigratedBondAngle");
}

/**
 * @brief Convert vector of legacy BondAngles to ProductPotentials
 * @param legacy_bond_angles Vector of legacy BondAngle structures
 * @return Vector of ProductPotential structures
 */
inline std::vector<ProductPotential>
migrate_bond_angles_to_product_potentials(const std::vector<LegacyBondAngle>& legacy_bond_angles) {
	std::vector<ProductPotential> product_potentials;
	product_potentials.reserve(legacy_bond_angles.size());

	for (const auto& legacy : legacy_bond_angles) {
		product_potentials.push_back(migrate_bond_angle_to_product_potential(legacy));
	}

	return product_potentials;
}

/**
 * @brief Create ProductPotential from individual components
 * @param i,j,k,l Particle indices
 * @param angle1_table_index Index for angle 1 table
 * @param bond_table_index Index for bond table
 * @param angle2_table_index Index for angle 2 table
 * @param name Optional name for the product potential
 * @return ProductPotential structure
 */
inline ProductPotential create_bond_angle_product_potential(int i,
															int j,
															int k,
															int l,
															int angle1_table_index,
															int bond_table_index,
															int angle2_table_index,
															const std::string& name = "BondAngle") {
	return ProductPotential::from_bond_angle(i,
											 j,
											 k,
											 l,
											 angle1_table_index,
											 bond_table_index,
											 angle2_table_index,
											 name);
}

/**
 * @brief Create AngleAngle product potential
 * @param i,j,k,l Particle indices
 * @param angle1_table_index Index for angle 1 table
 * @param angle2_table_index Index for angle 2 table
 * @param name Optional name for the product potential
 * @return ProductPotential structure
 */
inline ProductPotential
create_angle_angle_product_potential(int i,
									 int j,
									 int k,
									 int l,
									 int angle1_table_index,
									 int angle2_table_index,
									 const std::string& name = "AngleAngle") {
	return ProductPotential(i,
							j,
							k,
							l,
							ProductPotentialType::AngleAngle,
							angle1_table_index,
							-1, // No bond term
							angle2_table_index,
							name);
}

/**
 * @brief Create BondBond product potential
 * @param i,j,k Particle indices
 * @param bond1_table_index Index for bond 1 table
 * @param bond2_table_index Index for bond 2 table
 * @param name Optional name for the product potential
 * @return ProductPotential structure
 */
inline ProductPotential create_bond_bond_product_potential(int i,
														   int j,
														   int k,
														   int bond1_table_index,
														   int bond2_table_index,
														   const std::string& name = "BondBond") {
	return ProductPotential(i,
							j,
							k,
							-1, // 4th particle unused for BondBond
							ProductPotentialType::BondBond,
							bond1_table_index,
							bond2_table_index,
							-1, // No third term
							name);
}

/**
 * @brief Validate ProductPotential structure
 * @param pp ProductPotential to validate
 * @return true if valid, false otherwise
 */
inline bool validate_product_potential(const ProductPotential& pp) {
	// Check particle indices
	if (pp.particle_indices.x < 0 || pp.particle_indices.y < 0 || pp.particle_indices.z < 0 ||
		pp.particle_indices.w < 0) {
		return false;
	}

	// Check potential indices based on type
	switch (pp.type) {
	case ProductPotentialType::BondAngle:
		return (pp.potential_index_1 >= 0 && pp.potential_index_2 >= 0 &&
				pp.potential_index_3 >= 0);
	case ProductPotentialType::AngleAngle:
		return (pp.potential_index_1 >= 0 && pp.potential_index_3 >= 0);
	case ProductPotentialType::BondBond:
		return (pp.potential_index_1 >= 0 && pp.potential_index_2 >= 0);
	case ProductPotentialType::Custom:
		return true; // Custom validation logic would go here
	default:
		return false;
	}
}

/**
 * @brief Print ProductPotential information for debugging
 * @param pp ProductPotential to print
 */
inline void print_product_potential(const ProductPotential& pp) {
	printf("ProductPotential: %s\n", pp.name.c_str());
	printf("  Type: %d\n", static_cast<int>(pp.type));
	printf("  Particles: %d-%d-%d-%d\n",
		   pp.particle_indices.x,
		   pp.particle_indices.y,
		   pp.particle_indices.z,
		   pp.particle_indices.w);
	printf("  Potential indices: %d, %d, %d\n",
		   pp.potential_index_1,
		   pp.potential_index_2,
		   pp.potential_index_3);
	printf("  Form: %d, FunctionIndex: %d\n", static_cast<int>(pp.form), pp.functionIndex);
}

} // namespace ARBD
