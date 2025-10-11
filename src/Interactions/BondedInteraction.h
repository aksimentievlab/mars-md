/**
 * @file LocalInteraction.h
 * @brief Defines the LocalInteraction class and its related structures. Host side AoS.
 */

#pragma once
#include "Header.h"
#include "IO/Reader.h"
#include "Objects/ParticleProperties.h"
#include "SimParam.h"

namespace ARBD {
class Patch;
enum class BondFlag { DEFAULT = 1, REPLACE = 1, ADD = 2 };

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

class Bond {
  public:
	Bond() : ind1(-1), ind2(-1) {}
	int ind1, ind2;
	std::string name; // file name or function name
	InteractionForm form;
	int functionIndex;
	BondFlag flag;
	void add_exclusion(std::vector<Exclude>& excludes) {
		excludes.push_back(Exclude(ind1, ind2));
	}
};

struct Angle {
	int ind1, ind2, ind3;
	std::string name;
	InteractionForm form;
	int functionIndex;
};

struct Dihedral {
	int ind1, ind2, ind3, ind4;
	std::string name;
	InteractionForm form;
	int functionIndex; // can be user-defined function index or tabulated file id.
};

struct Restraint {
  public:
	Restraint() : id(-1) {}
	Restraint(int id, Vector3 r0, float k) : id(id), r0(r0), k(k) {}
	int id;
	Vector3 r0;
	float k;
};

} // namespace ARBD
