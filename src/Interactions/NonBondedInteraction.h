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

// Tabulated interactions, such as LJ and Columb
struct PairNonBonded {
	int type_id_1{-1};
	int type_id_2{-1};
	int id{-1};
	std::string function_name{""}; // tabulated file name or function name
	InteractionForm form{InteractionForm::Tabulated};
	int function_index{-1};
	PairNonBonded(int type_id_1, int type_id_2, const std::string& function_name)
		: type_id_1(type_id_1), type_id_2(type_id_2), function_name(function_name) {}

	PairNonBonded(int type_id_1, int type_id_2, InteractionForm form = InteractionForm::Analytical)
		: type_id_1(type_id_1), type_id_2(type_id_2), form(form) {}
};

// Grids, such as Pmf and E-field
struct LongRangeNonBonded {
	int id{-1};
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
	~NonBondedInteractions() = default;

	// Host-side data management
	void addPairNonBonded(const PairNonBonded& pair_nonbonded) {
		pair_nonbonded_.push_back(pair_nonbonded);
	}
	void addLongRangeNonBonded(const LongRangeNonBonded& long_range_nonbonded) {
		long_range_nonbonded_.push_back(long_range_nonbonded);
	}
	// Device data preparation
	void prepareDeviceData();
	void cleanupDeviceData();

	size_t getNumPairNonBonded() const {
		return pair_nonbonded_.size();
	}
	size_t getNumLongRangeNonBonded() const {
		return long_range_nonbonded_.size();
	}
	// Assign ids to pair and long range nonbonded interactions
	void assign_id() {
		for (int i = 0; i < pair_nonbonded_.size(); i++) {
			pair_nonbonded_[i].id = i;
		}
		for (int i = 0; i < long_range_nonbonded_.size(); i++) {
			long_range_nonbonded_[i].id = i;
		}
	}

  private:
	std::vector<PairNonBonded> pair_nonbonded_;
	std::vector<LongRangeNonBonded> long_range_nonbonded_;
};
} // namespace ARBD
