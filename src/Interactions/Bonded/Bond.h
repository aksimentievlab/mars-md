#pragma once

#include "Backend/Header.h"
#include "Math/Types.h"
#include "System/SimSystem.h"

namespace ARBD {

const String flags[] = {"DEFAULT", "REPLACE", "ADD"};

class Bond {
  public:
	enum { DEFAULT = 1, REPLACE = 1, ADD = 2 };

	Bond() : flag(DEFAULT), ind1(-1), ind2(-1) {}

	Bond(int flag, int ind1, int ind2, String fileName)
		: flag(flag), ind1(ind1), ind2(ind2), fileName(fileName) {}

	Bond(String strflag, int ind1, int ind2, String fileName);

	int flag;
	int ind1;
	int ind2;
	int tabFileIndex;
	String fileName;
	Bond(String strflag, int ind1, int ind2, String fileName)
		: ind1(ind1), ind2(ind2), fileName(fileName) {
		if (strflag == "REPLACE") {
			flag = REPLACE;
		} else if (strflag == "ADD") {
			flag = ADD;
		} else {
			printf("WARNING: Invalid operation flag found:"
				   "\"BOND %s %d %d\"\n",
				   strflag.c_str(),
				   ind1,
				   ind2);
			printf("sing default flag\n");
			flag = DEFAULT;
		}
		tabFileIndex = -1;
	}

	void print() {
		printf("BOND %s %d %d %s\n", flags[flag].c_str(), ind1, ind2, fileName.c_str());
	}

	String toString() {
		return "BOND " + flags[flag] + " " + std::to_string(ind1) + " " + std::to_string(ind2) +
			   " " + fileName;
	}
};

} // namespace ARBD
