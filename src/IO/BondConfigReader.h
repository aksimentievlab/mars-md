#pragma once
#include "Header.h"
#include "IO/Reader.h"
#include "Interactions/BondedInteraction.h"
#include <string>
#include <string_view>
#include <vector>

namespace ARBD {
/**
 * @brief Reader for bond configuration files
 * @param fileName Name of the configuration file to read.
 * @return Returns angles, dihedrals, and bonds from the configuration file.
 */
// Registry to load and cache tabulated potentials by name
class TablesRegistry {
  public:
	static TablesRegistry& instance();

	int getOrLoadAngle(const std::string& name, const std::string& path);
	int getOrLoadDihedral(const std::string& name, const std::string& path);
	int getOrLoadBond(const std::string& name, const std::string& path);

  private:
	TablesRegistry() = default;

	std::unordered_map<std::string, int> angleNameToIdx_;
	std::unordered_map<std::string, int> dihedralNameToIdx_;
	std::unordered_map<std::string, int> bondNameToIdx_;
};

class BondConfigReader {
  public:
	explicit BondConfigReader(std::string_view fileName) {
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

		LOGINFO("BondConfigReader.h: Loaded {} angles, {} dihedrals, {} bonds from '{}'",
				angles_.size(),
				dihedrals_.size(),
				bonds_.size(),
				fileName);
	}

	void parseAngleLine(const std::string& line) {
		std::istringstream iss(line);
		Angle angle;

		if (iss >> angle.ind1 >> angle.ind2 >> angle.ind3 >> angle.function_name) {
			angle.form = InteractionForm::Tabulated;
			// Resolve via registry: assumes name is a filename or key and path == name
			angle.function_index =
				TablesRegistry::instance().getOrLoadAngle(angle.function_name, angle.function_name);
			angles_.push_back(angle);
		} else {
			LOGWARN("BondConfigReader.h: Failed to parse ANGLE line: {}", line);
		}
	}

	void parseDihedralLine(const std::string& line) {
		std::istringstream iss(line);
		Dihedral dihedral;

		if (iss >> dihedral.ind1 >> dihedral.ind2 >> dihedral.ind3 >> dihedral.ind4 >>
			dihedral.function_name) {
			dihedral.form = InteractionForm::Tabulated;
			dihedral.function_index =
				TablesRegistry::instance().getOrLoadDihedral(dihedral.function_name,
															 dihedral.function_name);
			dihedrals_.push_back(dihedral);
		} else {
			LOGWARN("BondConfigReader.h: Failed to parse DIHEDRAL line: {}", line);
		}
	}

	void parseBondLine(const std::string& line) {
		std::istringstream iss(line);
		Bond bond;

		if (iss >> bond.ind1 >> bond.ind2 >> bond.function_name) {
			bond.form = InteractionForm::Tabulated;
			bond.function_index =
				TablesRegistry::instance().getOrLoadBond(bond.function_name, bond.function_name);
			bond.flag = BondFlag::DEFAULT;
			bonds_.push_back(bond);
		} else {
			LOGWARN("BondConfigReader.h: Failed to parse BOND line: {}", line);
		}
	}
};
} // namespace ARBD
