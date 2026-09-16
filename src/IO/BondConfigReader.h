#pragma once
#include "Header.h"
#include "IO/Reader.h"
#include "Interactions/Bonded/Analytical.h"
#include "Interactions/BondedInteraction.h"
#include "Objects/Tables.h"
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace MARS {
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
		bond_lines_ = 0;
		duplicate_bonds_ = 0;
		conflicting_bonds_ = 0;
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
			} else if (key == "RESTRAINT") {
				parse_restraint_line(value);
			}
		}
		if (duplicate_bonds_ > 0) {
			LOGWARN("BondConfigReader.h: {}: {} of {} BOND lines duplicate a pair already "
					"defined ({:.2f}%) - {} unique bonds kept, {} had a conflicting potential. "
					"One line per bond is expected; both directions are added internally, so "
					"listing a pair twice would double its force.",
					resolved_path,
					duplicate_bonds_,
					bond_lines_,
					100.0 * static_cast<double>(duplicate_bonds_) /
						static_cast<double>(bond_lines_),
					bond_lines_ - duplicate_bonds_,
					conflicting_bonds_);
		}
	}

  private:
	BondedInteractions& bonded_interactions_;
	TablesRegistry& tables_registry_;
	std::string config_file_path_;
	/// Unordered pair (min,max) -> index in bonds_, to drop bidirectional duplicates.
	std::unordered_map<uint64_t, size_t> seen_bond_slot_;
	size_t bond_lines_{0};		 ///< BOND lines accepted from the current file
	size_t duplicate_bonds_{0};	 ///< of those, collapsed onto an existing pair
	size_t conflicting_bonds_{0}; ///< of those, carrying a different potential

	/// Order-independent key for a bonded pair.
	static uint64_t canonical_bond_key(int a, int b) {
		const uint32_t lo = static_cast<uint32_t>(a < b ? a : b);
		const uint32_t hi = static_cast<uint32_t>(a < b ? b : a);
		return (static_cast<uint64_t>(lo) << 32) | hi;
	}

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

		++bond_lines_;
		const uint64_t bond_key = canonical_bond_key(bond.ind1, bond.ind2);
		auto slot = seen_bond_slot_.find(bond_key);
		if (slot != seen_bond_slot_.end()) {
			++duplicate_bonds_;
			const Bond& existing = bonded_interactions_.get_bonds()[slot->second];
			if (existing.function_name != bond.function_name) {
				++conflicting_bonds_;
				LOGWARN("BondConfigReader.h: duplicate bond ({}, {}) with conflicting "
						"potential '{}' vs '{}' - keeping the latter",
						bond.ind1,
						bond.ind2,
						existing.function_name,
						bond.function_name);
				bonded_interactions_.set_bond(slot->second, bond);
			}
			return;
		}
		seen_bond_slot_.emplace(bond_key, bonded_interactions_.get_num_bonds());
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
} // namespace MARS
