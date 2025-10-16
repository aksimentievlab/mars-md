#pragma once
#include "../BondedInteraction.h"
#include "Header.h"
#include "IO/Reader.h"
#include "TablesRegistry.h"

namespace ARBD {

class BondedReader {
  public:
	explicit BondedReader(std::string_view fileName) {
		readFile(fileName);
	}

	const std::vector<Angle>& getAngles() const {
		return angles_;
	}
	const std::vector<Dihedral>& getDihedrals() const {
		return dihedrals_;
	}
	const std::vector<Bond>& getBonds() const {
		return bonds_;
	}

  private:
	std::vector<Angle> angles_;
	std::vector<Dihedral> dihedrals_;
	std::vector<Bond> bonds_;

	void readFile(std::string_view fileName) {
		Reader reader(fileName);

		for (size_t i = 0; i < reader.length(); ++i) {
			std::string param = reader.getParameter(i); // ANGLE/DIHEDRAL/BOND
			std::string value = reader.getValue(i);		// Rest of the line

			if (param == "ANGLE") {
				parseAngleLine(value);
			} else if (param == "DIHEDRAL") {
				parseDihedralLine(value);
			} else if (param == "BOND") {
				parseBondLine(value);
			}
		}

		LOGINFO("BondedReader.h: Loaded {} angles, {} dihedrals, {} bonds from '{}'",
				angles_.size(),
				dihedrals_.size(),
				bonds_.size(),
				fileName);
	}

	void parseAngleLine(const std::string& line) {
		std::istringstream iss(line);
		Angle angle;

		if (iss >> angle.ind1 >> angle.ind2 >> angle.ind3 >> angle.name) {
			angle.form = InteractionForm::Tabulated;
			// Resolve via registry: assumes name is a filename or key and path == name
			angle.functionIndex = TablesRegistry::instance().getOrLoadAngle(angle.name, angle.name);
			angles_.push_back(angle);
		} else {
			LOGWARN("BondedReader.h: Failed to parse ANGLE line: {}", line);
		}
	}

	void parseDihedralLine(const std::string& line) {
		std::istringstream iss(line);
		Dihedral dihedral;

		if (iss >> dihedral.ind1 >> dihedral.ind2 >> dihedral.ind3 >> dihedral.ind4 >>
			dihedral.name) {
			dihedral.form = InteractionForm::Tabulated;
			dihedral.functionIndex = TablesRegistry::instance().getOrLoadDihedral(dihedral.name, dihedral.name);
			dihedrals_.push_back(dihedral);
		} else {
			LOGWARN("BondedReader.h: Failed to parse DIHEDRAL line: {}", line);
		}
	}

	void parseBondLine(const std::string& line) {
		std::istringstream iss(line);
		Bond bond;

		if (iss >> bond.ind1 >> bond.ind2 >> bond.name) {
			bond.form = InteractionForm::Tabulated;
			bond.functionIndex = TablesRegistry::instance().getOrLoadBond(bond.name, bond.name);
			bond.flag = BondFlag::DEFAULT;
			bonds_.push_back(bond);
		} else {
			LOGWARN("BondedReader.h: Failed to parse BOND line: {}", line);
		}
	}
};
} // namespace ARBD
