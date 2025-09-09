#pragma once

#include "Backend/Header.h"
#include "Math/BaseGrid.h"
#include "Math/Types.h"
#include "System/SimSystem.h"

namespace ARBD {

class Dihedral {
  public:
	int ind1, ind2, ind3, ind4;
	int tabFileIndex;
	// This will be assigned after ComputeForce.cu loads the TabulatedDihedralPotential objects.
	// The tabFileIndex is used by ComputeForce to discern which TabDiPot this Dihedral object uses.
	String fileName;
	Dihedral() : ind1(-1), ind2(-1), ind3(-1), ind4(-1), tabFileIndex(-1) {}
	Dihedral(int ind1, int ind2, int ind3, int ind4, String fileName);
	Dihedral(const Dihedral& d);
	HOST DEVICE inline int getIndex(int index) const {
		if (index == ind1)
			return 1;
		if (index == ind2)
			return 2;
		if (index == ind3)
			return 3;
		if (index == ind4)
			return 4;
		return -1;
	}
	Dihedral(int ind1, int ind2, int ind3, int ind4, String fileName)
		: ind1(ind1), ind2(ind2), ind3(ind3), ind4(ind4), fileName(fileName) {}

	Dihedral(const Dihedral& d)
		: ind1(d.ind1), ind2(d.ind2), ind3(d.ind3), ind4(d.ind4), tabFileIndex(d.tabFileIndex),
		  fileName(d.fileName) {}

	String toString() {
		return String("DIHEDRAL ") + std::to_string(ind1) + " " + std::to_string(ind2) + " " +
			   std::to_string(ind3) + " " + std::to_string(ind4) + " " + fileName;
	}

	void print() {
		printf("DIHEDRAL (%d %d %d %d) %s\n", ind1, ind2, ind3, ind4, fileName.c_str());
	}
};
} // namespace ARBD
