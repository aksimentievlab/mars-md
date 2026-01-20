#pragma once
/**
 * @file LocalInteraction.h
 * @brief Defines the LocalInteraction class and its related structures
 */

#include "BondedInteraction.h"
#include "Header.h"
#include "IO/Reader.h"
#include "Objects/DeviceParticle.h"
#include "Objects/ParticleProperties.h"
#include "SimParam.h"

namespace ARBD {
namespace AnalyticalNameList {
const std::vector<std::string> pair_nonbonded_types = {"LJ"};
const std::vector<std::string> long_range_nonbonded_types = {"E-field", "Pmf"};
} // namespace AnalyticalNameList
// Tabulated interactions, such as LJ and Columb
struct PairNonBonded {
	int type_id_1{-1};
	int type_id_2{-1};
	int id{-1};
	std::string function_name{""}; // tabulated file name or function name
	InteractionForm form{InteractionForm::Tabulated};
	int function_index{-1};
	PairNonBonded(int type_id_1, int type_id_2, const std::string& function_name)
		: function_name(function_name) {
		this->type_id_1 = std::min(type_id_1, type_id_2);
		this->type_id_2 = std::max(type_id_1, type_id_2);
		auto it = std::find(AnalyticalNameList::pair_nonbonded_types.begin(),
							AnalyticalNameList::pair_nonbonded_types.end(),
							function_name);
		if (it != AnalyticalNameList::pair_nonbonded_types.end()) {
			form = InteractionForm::Analytical;
			function_index = std::distance(AnalyticalNameList::pair_nonbonded_types.begin(), it);
		} else {
			form = InteractionForm::Tabulated;
			this->function_name = function_name;
		}
	}
};

// Grids, such as Pmf and E-field
struct LongRangeNonBonded {
	int type_id{-1};
	std::string function_name{""}; // grid file name or function name
	InteractionForm form{InteractionForm::Grid};
	int function_index{-1};
};

/**
 * @brief Host-Side Nonbonded interaction manager
 */
class NonBondedInteractions {
  public:
	NonBondedInteractions(std::vector<PairNonBonded> pair_nonbonded,
						  std::vector<LongRangeNonBonded> long_range_nonbonded)
		: pair_nonbonded_(pair_nonbonded), long_range_nonbonded_(long_range_nonbonded) {}
	NonBondedInteractions() = default;
	~NonBondedInteractions() = default;

	// Host-side data management
	void add_pair_nonbonded(const PairNonBonded& pair_nonbonded) {
		pair_nonbonded_.push_back(pair_nonbonded);
	}
	void add_long_range_nonbonded(const LongRangeNonBonded& long_range_nonbonded) {
		long_range_nonbonded_.push_back(long_range_nonbonded);
	}
	// Device data preparation
	void prepare_device_data();
	void cleanup_device_data();

	size_t get_num_pair_nonbonded() const {
		return pair_nonbonded_.size();
	}
	size_t get_num_long_range_nonbonded() const {
		return long_range_nonbonded_.size();
	}
	// Assign ids to pair and long range nonbonded interactions
	void assign_id() {
		for (int i = 0; i < pair_nonbonded_.size(); i++) {
			pair_nonbonded_[i].type_id_1 = i;
			pair_nonbonded_[i].type_id_2 = i;
		}
		for (int i = 0; i < long_range_nonbonded_.size(); i++) {
			long_range_nonbonded_[i].type_id = i;
		}
	}

  private:
	std::vector<PairNonBonded> pair_nonbonded_{};
	std::vector<LongRangeNonBonded> long_range_nonbonded_{};
};
} // namespace ARBD
