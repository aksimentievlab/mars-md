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
	Bond() : ind1(-1), ind2(-1), functionIndex(-1), flag(BondFlag::REPLACE) {}
	Bond(int ind1, int ind2, std::vector<Exclude>& exclusions)
		: ind1(ind1), ind2(ind2), functionIndex(-1), flag(BondFlag::DEFAULT) {
		if (!(flag == BondFlag::REPLACE)) {
			add_exclusion(exclusions);
		}
	}

	int ind1, ind2;
	std::string name; // file name or function name
	InteractionForm form;
	int functionIndex;
	BondFlag flag;

	void add_exclusion(std::vector<Exclude>& excludes) {
		excludes.push_back(Exclude(ind1, ind2));
	}
};

// Host-side angle definition
struct Angle {
	int ind1, ind2, ind3;
	std::string name;
	InteractionForm form;
	int functionIndex;
};

// Host-side dihedral definition
struct Dihedral {
	int ind1, ind2, ind3, ind4;
	std::string name;
	InteractionForm form;
	int functionIndex;
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

// ============================================================================
// BONDED INTERACTION MANAGER
// ============================================================================

class BondedInteraction {
  public:
	BondedInteraction(std::vector<Bond> sys_bonds,
					  std::vector<Angle> sys_angles,
					  std::vector<Dihedral> sys_dihedrals)
		: bonds_(sys_bonds), angles_(sys_angles), dihedrals_(sys_dihedrals) {}
	~BondedInteraction() = default;

	// Host-side data management
	void addBond(const Bond& bond) {
		bonds_.push_back(bond);
	}
	void addAngle(const Angle& angle) {
		angles_.push_back(angle);
	}
	void addDihedral(const Dihedral& dihedral) {
		dihedrals_.push_back(dihedral);
	}

	// Device data preparation
	void prepareDeviceData();
	void cleanupDeviceData();

	size_t getNumBonds() const {
		return bonds_.size();
	}
	size_t getNumAngles() const {
		return angles_.size();
	}
	size_t getNumDihedrals() const {
		return dihedrals_.size();
	}

  private:
	// Host-side data
	std::vector<Bond> bonds_;
	std::vector<Angle> angles_;
	std::vector<Dihedral> dihedrals_;
};

// ============================================================================
// POTENTIAL REGISTRATION
// ============================================================================

/** Future pybind class for new analytical potential registration */
class Register_Potential {
  public:
	int register_potential(const std::string& name) {
		if (m_name_to_id.find(name) == m_name_to_id.end()) {
			int new_id = m_name_to_id.size();
			m_name_to_id[name] = new_id;
		}
		return m_name_to_id[name];
	}

	int get_id(const std::string& name) const {
		return m_name_to_id.at(name);
	}

  private:
	std::unordered_map<std::string, int> m_name_to_id;
};
} // namespace ARBD
