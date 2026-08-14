#pragma once
#include "Header.h"
#include "IO/Reader.h"
#include "Interactions/Bonded/Analytical.h"
#include "Interactions/BondedInteraction.h"
#include "Objects/Tables.h"
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace ARBD {
/**
 * @brief Reader for bond configuration files
 * @param fileName Name of the configuration file to read.
 * @return Returns angles, dihedrals, and bonds from the configuration file.
 */
class BondConfigReader {
  public:
	explicit BondConfigReader(BondedInteractions& bonded_interactions,
							  TablesRegistry& tables_registry)
		: bonded_interactions_(bonded_interactions), tables_registry_(tables_registry) {}

	HOST void read_file(std::string_view fileName, std::string_view config_file_path = "") {
		config_file_path_ = config_file_path;
		std::string resolved_path =
			config_file_path.empty()
				? std::string(fileName)
				: resolve_file_path(std::string(fileName), std::string(config_file_path));
		Reader reader(resolved_path);
		for (auto [key, value] : reader) {
			std::string line = value;
			if (key == "ANGLE") {
				parse_angle_line(value);
			} else if (key == "DIHEDRAL") {
				parse_dihedral_line(value);
			} else if (key == "BOND") {
				parse_bond_line(value);
			} else if (key == "EXCLUDE") {
				parse_exclude_line(value);
			}
		}
	}

  private:
	BondedInteractions& bonded_interactions_;
	TablesRegistry& tables_registry_;
	std::string config_file_path_;

	void parse_angle_line(const std::string& line) {
		std::istringstream iss(line);
		Angle angle;

		if (iss >> angle.ind1 >> angle.ind2 >> angle.ind3 >> angle.function_name) {
			auto it = std::find(AnalyticalNameList::angle_types.begin(),
								AnalyticalNameList::angle_types.end(),
								angle.function_name);
			if (it != AnalyticalNameList::angle_types.end()) {
				angle.form = InteractionForm::Analytical;
				angle.function_index = std::distance(AnalyticalNameList::angle_types.begin(), it);
			} else {
				angle.form = InteractionForm::Tabulated;
				angle.function_index =
					tables_registry_.get_or_load_angle(angle.function_name, config_file_path_);
			}
			bonded_interactions_.add_angle(angle);
		} else {
			LOGWARN("BondConfigReader.h: Failed to parse ANGLE line: {}", line);
		}
	}

	void parse_dihedral_line(const std::string& line) {
		std::istringstream iss(line);
		Dihedral dihedral;

		if (iss >> dihedral.ind1 >> dihedral.ind2 >> dihedral.ind3 >> dihedral.ind4 >>
			dihedral.function_name) {
			auto it = std::find(AnalyticalNameList::dihedral_types.begin(),
								AnalyticalNameList::dihedral_types.end(),
								dihedral.function_name);
			if (it != AnalyticalNameList::dihedral_types.end()) {
				dihedral.form = InteractionForm::Analytical;
				dihedral.function_index =
					std::distance(AnalyticalNameList::dihedral_types.begin(), it);
			} else {
				dihedral.form = InteractionForm::Tabulated;
				dihedral.function_index =
					tables_registry_.get_or_load_dihedral(dihedral.function_name,
														  config_file_path_);
			}
			bonded_interactions_.add_dihedral(dihedral);
		} else {
			LOGWARN("BondConfigReader.h: Failed to parse DIHEDRAL line: {}", line);
		}
	}

	void parse_bond_line(const std::string& line, bool add_exclusions = false) {
		std::istringstream iss(line);
		Bond bond;

		// Legacy format is: BOND | OPERATION_FLAG | INDEX1 | INDEX2 | FILENAME
		// where OPERATION_FLAG is REPLACE or ADD. The BOND key is stripped by
		// the caller, so the first token here is the flag. Files written
		// without a flag start straight at INDEX1, so treat it as optional
		// rather than rejecting the line.
		std::string first_token;
		if (!(iss >> first_token)) {
			LOGWARN("BondConfigReader.h: Failed to parse BOND line: {}", line);
			return;
		}

		// A REPLACE bond supersedes the nonbonded interaction for the pair, so
		// the pair gets excluded from nonbonded evaluation (legacy readBonds()).
		bool replaces_nonbonded = true;
		if (first_token == "REPLACE" || first_token == "ADD") {
			bond.flag = (first_token == "ADD") ? BondFlag::ADD : BondFlag::REPLACE;
			replaces_nonbonded = (first_token == "REPLACE");
			if (!(iss >> bond.ind1)) {
				LOGWARN("BondConfigReader.h: Failed to parse BOND line: {}", line);
				return;
			}
		} else {
			bond.flag = BondFlag::DEFAULT;
			try {
				size_t consumed = 0;
				bond.ind1 = std::stoi(first_token, &consumed);
				if (consumed != first_token.size()) {
					throw std::invalid_argument("trailing characters");
				}
			} catch (const std::exception&) {
				LOGWARN("BondConfigReader.h: Failed to parse BOND line: {}", line);
				return;
			}
		}

		if (!(iss >> bond.ind2 >> bond.function_name)) {
			LOGWARN("BondConfigReader.h: Failed to parse BOND line: {}", line);
			return;
		}

		if (bond.ind1 < 0 || bond.ind2 < 0 || bond.ind1 == bond.ind2) {
			LOGWARN("BondConfigReader.h: Invalid bond indices in BOND line: {}", line);
			return;
		}

		auto it = std::find(AnalyticalNameList::bond_types.begin(),
							AnalyticalNameList::bond_types.end(),
							bond.function_name);
		if (it != AnalyticalNameList::bond_types.end()) {
			bond.form = InteractionForm::Analytical;
			bond.function_index = std::distance(AnalyticalNameList::bond_types.begin(), it);
		} else {
			bond.form = InteractionForm::Tabulated;
			bond.function_index =
				tables_registry_.get_or_load_bond(bond.function_name, config_file_path_);
		}

		bonded_interactions_.add_bond(bond);

		if (add_exclusions || replaces_nonbonded) {
			bonded_interactions_.add_exclude(Exclude(bond.ind1, bond.ind2));
		}
	}

	void parse_exclude_line(const std::string& line) {
		std::istringstream iss(line);
		Exclude exclude;

		if (iss >> exclude.ind1 >> exclude.ind2) {
			bonded_interactions_.add_exclude(exclude);
		} else {
			LOGWARN("BondConfigReader.h: Failed to parse EXCLUDE line: {}", line);
		}
	}

	void parse_restraint_line(const std::string& line) {
		std::istringstream iss(line);
		Restraint restraint;
		float x0, y0, z0;

		if (iss >> restraint.ind >> restraint.k >> x0 >> y0 >> z0) {
			restraint.r0 = Vector3{x0, y0, z0};
			bonded_interactions_.add_restraint(restraint);
		} else {
			LOGWARN("BondConfigReader.h: Failed to parse RESTRAINT line: {}", line);
		}
	}
};
} // namespace ARBD
