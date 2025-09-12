#pragma once

#include "Header.h"
#include "IO/Reader.h"
#include "Types/Types.h"

namespace ARBD {

enum class BondFlag { DEFAULT = 1, REPLACE = 1, ADD = 2 };

class Bond {
  public:
	BondFlag flag;

	Bond() : flag(BondFlag::DEFAULT), ind1(-1), ind2(-1) {}

	Bond(BondFlag flag, int ind1, int ind2, std::string fileName)
		: flag(flag), ind1(ind1), ind2(ind2), fileName(fileName) {}

	Bond(BondFlag flag, int ind1, int ind2, std::string fileName);

	int ind1;
	int ind2;
	int tabFileIndex;
	std::string fileName;
	Bond(std::string strflag, int ind1, int ind2, std::string fileName)
		: ind1(ind1), ind2(ind2), fileName(fileName) {
		if (strflag == "REPLACE") {
			flag = BondFlag::REPLACE;
		} else if (strflag == "ADD") {
			flag = BondFlag::ADD;
		} else {
			printf("WARNING: Invalid operation flag found:"
				   "\"BOND %s %d %d\"\n",
				   strflag.c_str(),
				   ind1,
				   ind2);
			printf("sing default flag\n");
			flag = BondFlag::DEFAULT;
		}
		tabFileIndex = -1;
	}

	void print() {
		printf("BOND %s %d %d %s\n", flags[flag].c_str(), ind1, ind2, fileName.c_str());
	}

	std::string tostd::string() {
		return "BOND " + flags[flag] + " " + std::to_std::string(ind1) + " " +
			   std::to_std::string(ind2) + " " + fileName;
	}
};

} // namespace ARBD
