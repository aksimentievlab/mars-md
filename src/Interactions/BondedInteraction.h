/**
 * @file BondedInteraction.h
 * @brief Host-side data structures for bonded interactions and device data preparation
 */

#pragma once
#include "Header.h"
#include "IO/Reader.h"
#include "Objects/ParticleProperties.h"
#include "SimParam.h"

namespace ARBD {

enum class BondFlag { DEFAULT = 1, REPLACE = 1, ADD = 2 };
enum BondedPotentialType { UNSET, BOND, ANGLE, DIHEDRAL, VECANGLE };
constexpr auto BD_PI = constants::PI;
namespace AnalyticalNameList {
const std::vector<std::string> bond_types = {"Harmonic", "Morse", "FENE", "Half_Harmonic", "WLCSK"};
const std::vector<std::string> angle_types = {"Harmonic",
											  "Morse",
											  "FENE",
											  "Half_Harmonic",
											  "WLCSK"};
const std::vector<std::string> dihedral_types = {"Harmonic",
												 "Morse",
												 "FENE",
												 "Half_Harmonic",
												 "WLCSK"};
} // namespace AnalyticalNameList
enum class AnalyticalBondType {
	Harmonic = 0,
	Morse = 1,
	FENE = 2,
	Half_Harmonic = 3,
	WLCSK = 4,
	COUNT
};
enum class AnalyticalAngleType {
	Harmonic = 0,
	Morse = 1,
	FENE = 2,
	Half_Harmonic = 3,
	WLCSK = 4,
	COUNT
};
enum class AnalyticalDihedralType {
	Harmonic = 0,
	Morse = 1,
	FENE = 2,
	Half_Harmonic = 3,
	WLCSK = 4,
	COUNT
};

// Host-Side Exclusion Definition
class Exclude {
  public:
	int ind1;
	int ind2;
	Exclude() : ind1(-1), ind2(-1) {}
	Exclude(int ind1, int ind2) : ind1(ind1), ind2(ind2) {}
	bool operator!=(const Exclude& other) const {
		return ind1 != other.ind1 || ind2 != other.ind2;
	}
	bool operator<(const Exclude& other) const {
		if (ind1 != other.ind1)
			return ind1 < other.ind1;
		return ind2 < other.ind2;
	}
	bool operator==(const Exclude& other) const {
		return ind1 == other.ind1 && ind2 == other.ind2;
	}
};

// Host-side bond definition
class Bond {
  public:
	Bond() : ind1(-1), ind2(-1), function_index(-1), flag(BondFlag::REPLACE) {}
	Bond(int ind1, int ind2, std::vector<Exclude>& exclusions)
		: ind1(ind1), ind2(ind2), function_index(-1), flag(BondFlag::DEFAULT) {
		if (!(flag == BondFlag::REPLACE)) {
			add_exclusion(exclusions);
		}
	}
	Bond(int ind1, int ind2, AnalyticalBondType type)
		: ind1(ind1), ind2(ind2), form(InteractionForm::Analytical), flag(BondFlag::DEFAULT) {
		switch (type) {
		case AnalyticalBondType::Harmonic:
			function_name = "Harmonic";
			function_index = 0;
			break;
		case AnalyticalBondType::Morse:
			function_name = "Morse";
			function_index = 1;
			break;
		case AnalyticalBondType::FENE:
			function_name = "FENE";
			function_index = 2;
			break;
		case AnalyticalBondType::Half_Harmonic:
			function_name = "Half_Harmonic";
			function_index = 3;
			break;
		case AnalyticalBondType::WLCSK:
			function_name = "WLCSK";
			function_index = 4;
			break;
		default:
			throw Exception(ExceptionType::ValueError, "Invalid analytical bond type");
		}
	}
	int ind1, ind2;
	std::string function_name{""}; // file name or function name
	InteractionForm form{InteractionForm::Tabulated};
	int function_index{-1};
	BondFlag flag{BondFlag::DEFAULT};
	void add_exclusion(std::vector<Exclude>& excludes) {
		excludes.push_back(Exclude(ind1, ind2));
	}
};

// Host-side angle definition
struct Angle {
	int ind1, ind2, ind3;
	std::string function_name;
	InteractionForm form;
	int function_index{-1};
};

// Host-side dihedral definition
struct Dihedral {
	int ind1, ind2, ind3, ind4;
	std::string function_name;
	InteractionForm form;
	int function_index{-1};
};

// Host-side restraint definition, harmonic restraint
struct Restraint {
  public:
	Restraint() : ind(-1) {}
	Restraint(int id, Vector3 r0, float k) : ind(id), r0(r0), k(k) {}
	int ind;
	Vector3 r0;
	float k;
};

struct VecAngle {
	int ind1, ind2, ind3, ind4;
	std::string function_name;
	InteractionForm form;
	int functionIndex;
};

/**
 * @brief Product potential types for coupled bonded interactions
 */
enum class ProductPotentialType {
	BondAngle,	// 2 angles + 1 bond (4 particles: i-j-k-l)
	AngleAngle, // 2 angles sharing a bond (4 particles: i-j-k-l)
	BondBond,	// 2 bonds sharing an atom (3 particles: i-j-k)
	Custom		// User-defined combination
};

/**
 * @brief Product potential coupling multiple bonded terms
 *
 * Supports various combinations of bonded interactions:
 * - BondAngle: 2 angles + 1 bond (4 particles: i-j-k-l)
 * - AngleAngle: 2 angles sharing a bond (4 particles: i-j-k-l)
 * - BondBond: 2 bonds sharing an atom (3 particles: i-j-k)
 * - Custom: User-defined combination
 */
struct ProductPotential {
	int4 particle_indices; // Up to 4 particles
	ProductPotentialType type;

	// Indices into tabulated potential arrays
	int potential_index_1; // First component (e.g., angle 1)
	int potential_index_2; // Second component (e.g., bond)
	int potential_index_3; // Third component (e.g., angle 2)

	std::string function_name;
	InteractionForm form;
	int functionIndex;

	ProductPotential()
		: particle_indices{-1, -1, -1, -1}, type(ProductPotentialType::BondAngle),
		  potential_index_1(-1), potential_index_2(-1), potential_index_3(-1),
		  form(InteractionForm::Tabulated), functionIndex(-1) {}

	ProductPotential(int i,
					 int j,
					 int k,
					 int l,
					 ProductPotentialType type,
					 int pot1,
					 int pot2,
					 int pot3,
					 const std::string& function_name = "ProductPotential")
		: particle_indices{i, j, k, l}, type(type), potential_index_1(pot1),
		  potential_index_2(pot2), potential_index_3(pot3), function_name(function_name),
		  form(InteractionForm::Tabulated), functionIndex(-1) {}

	// Helper to create BondAngle product potential
	static ProductPotential from_bond_angle(int i,
											int j,
											int k,
											int l,
											int angle1_idx,
											int bond_idx,
											int angle2_idx,
											const std::string& name = "BondAngle") {
		return ProductPotential(i,
								j,
								k,
								l,
								ProductPotentialType::BondAngle,
								angle1_idx,
								bond_idx,
								angle2_idx,
								name);
	}
};

// ============================================================================
// BONDED INTERACTION MANAGER
// ============================================================================

/**
 * @brief Host-Side Bonded interaction manager
 * @param sys_bonds System bonds
 * @param sys_angles System angles
 * @param sys_dihedrals System dihedrals
 */
class BondedInteractions {
  public:
	BondedInteractions(std::vector<Bond> sys_bonds,
					   std::vector<Angle> sys_angles,
					   std::vector<Dihedral> sys_dihedrals,
					   std::vector<Exclude> sys_excludes,
					   std::vector<Restraint> sys_restraints)
		: bonds_(sys_bonds), angles_(sys_angles), dihedrals_(sys_dihedrals),
		  exclusions_(sys_excludes), restraints_(sys_restraints) {}
	BondedInteractions() = default;
	~BondedInteractions() = default;

	// Host-side data management
	void add_bond(const Bond& bond) {
		bonds_.push_back(bond);
	}
	void add_angle(const Angle& angle) {
		angles_.push_back(angle);
	}
	void add_dihedral(const Dihedral& dihedral) {
		dihedrals_.push_back(dihedral);
	}
	void add_exclude(const Exclude& exclude) {
		exclusions_.push_back(exclude);
	}
	void add_restraint(const Restraint& restraint) {
		restraints_.push_back(restraint);
	}

	size_t get_num_bonds() const {
		return bonds_.size();
	}
	size_t get_num_angles() const {
		return angles_.size();
	}
	size_t get_num_dihedrals() const {
		return dihedrals_.size();
	}
	const std::vector<Bond>& get_bonds() const {
		return bonds_;
	}
	const std::vector<Angle>& get_angles() const {
		return angles_;
	}
	const std::vector<Dihedral>& get_dihedrals() const {
		return dihedrals_;
	}
	const std::vector<Exclude>& get_exclusions() const {
		return exclusions_;
	}
	const std::vector<Restraint>& get_restraints() const {
		return restraints_;
	}
	void make_exclusions(int num_particles, int exclusion_depth);
	void clear() {
		bonds_.clear();
		angles_.clear();
		dihedrals_.clear();
		exclusions_.clear();
		restraints_.clear();
	}

  private:
	std::vector<Bond> bonds_{};
	std::vector<Angle> angles_{};
	std::vector<Dihedral> dihedrals_{};
	std::vector<Exclude> exclusions_{};
	std::vector<Restraint> restraints_{};
};

} // namespace ARBD
