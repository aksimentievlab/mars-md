#pragma once
/**
 * @file NonBondedInteraction.h
 * @brief Host-side nonbonded interaction definitions
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

struct PairNonBonded {
	int type_id_1{-1};
	int type_id_2{-1};
	// Alternate, deferred-resolution path for callers that don't already know
	// the numeric type ids (e.g. the Python bindings, which take ParticleType
	// objects) - mirrors ParticleIO::type_name / RigidBodyIO::type_name. When
	// non-empty, SimSystem::build_name_to_id_maps() resolves these into
	// type_id_1/type_id_2 (see resolve_type_names below); the legacy
	// `tabulatedFile i@j@file` config path supplies ids directly and leaves
	// these empty. Safe to defer because the ids aren't read until
	// Patch::calculate_nonbonded_forces builds the type-pair matrix on the
	// first step, long after type ids are assigned.
	std::string type_name_1{};
	std::string type_name_2{};
	int id{-1};
	std::string function_name{""};
	InteractionForm form{InteractionForm::Tabulated};
	int function_index{-1};

	PairNonBonded(int type_id_1, int type_id_2, const std::string& function_name) {
		this->type_id_1 = std::min(type_id_1, type_id_2);
		this->type_id_2 = std::max(type_id_1, type_id_2);
		set_function(function_name);
	}

	PairNonBonded(const std::string& type_name_1,
				  const std::string& type_name_2,
				  const std::string& function_name)
		: type_name_1(type_name_1), type_name_2(type_name_2) {
		set_function(function_name);
	}

	/**
	 * @brief Resolve type_name_1/type_name_2 into type_id_1/type_id_2.
	 * @param name_to_id Callable mapping a type name to its assigned id.
	 * @note No-op when the names are empty (ids were supplied directly).
	 *       Templated on the lookup so this header stays independent of
	 *       SimSystem, which includes it transitively via Objects/Tables.h.
	 */
	template<typename NameToId>
	void resolve_type_names(NameToId&& name_to_id) {
		if (type_name_1.empty() || type_name_2.empty()) {
			return;
		}
		const int a = name_to_id(type_name_1);
		const int b = name_to_id(type_name_2);
		type_id_1 = std::min(a, b);
		type_id_2 = std::max(a, b);
	}

  private:
	void set_function(const std::string& function_name) {
		auto it = std::find(AnalyticalNameList::pair_nonbonded_types.begin(),
							AnalyticalNameList::pair_nonbonded_types.end(),
							function_name);
		if (it != AnalyticalNameList::pair_nonbonded_types.end()) {
			form = InteractionForm::Analytical;
			function_index = std::distance(AnalyticalNameList::pair_nonbonded_types.begin(), it);
		} else {
			form = InteractionForm::Tabulated;
		}
		this->function_name = function_name;
	}
};

struct LongRangeNonBonded {
	int type_id{-1};
	std::string function_name{""};
	InteractionForm form{InteractionForm::Grid};
	int function_index{-1};
};

/**
 * @brief Host-side nonbonded interaction registry
 *
 * Owns configuration metadata only. Device buffers live in
 * DevicePairNonBondedInteractions (pairwise) and GridManager (PMF/force grids).
 */
class NonBondedInteractions {
  public:
	NonBondedInteractions(std::vector<PairNonBonded> pair_nonbonded,
						  std::vector<LongRangeNonBonded> long_range_nonbonded)
		: pair_nonbonded_(std::move(pair_nonbonded)),
		  long_range_nonbonded_(std::move(long_range_nonbonded)) {}
	NonBondedInteractions() = default;
	~NonBondedInteractions() = default;

	void add_pair_nonbonded(const PairNonBonded& pair_nonbonded) {
		pair_nonbonded_.push_back(pair_nonbonded);
	}
	void add_long_range_nonbonded(const LongRangeNonBonded& long_range_nonbonded) {
		long_range_nonbonded_.push_back(long_range_nonbonded);
	}

	void prepare_device_data();
	void cleanup_device_data();

	const std::vector<PairNonBonded>& get_pair_nonbonded() const {
		return pair_nonbonded_;
	}
	const std::vector<LongRangeNonBonded>& get_long_range_nonbonded() const {
		return long_range_nonbonded_;
	}

	size_t get_num_pair_nonbonded() const {
		return pair_nonbonded_.size();
	}
	size_t get_num_long_range_nonbonded() const {
		return long_range_nonbonded_.size();
	}

	void assign_id() {
		for (size_t i = 0; i < pair_nonbonded_.size(); ++i) {
			pair_nonbonded_[i].id = static_cast<int>(i);
		}
		for (size_t i = 0; i < long_range_nonbonded_.size(); ++i) {
			long_range_nonbonded_[i].type_id = static_cast<int>(i);
		}
	}

	/**
	 * @brief Resolve every name-specified pair into numeric type ids.
	 * @see PairNonBonded::resolve_type_names
	 */
	template<typename NameToId>
	void resolve_type_names(NameToId&& name_to_id) {
		for (auto& pair : pair_nonbonded_) {
			pair.resolve_type_names(name_to_id);
		}
	}

  private:
	std::vector<PairNonBonded> pair_nonbonded_{};
	std::vector<LongRangeNonBonded> long_range_nonbonded_{};
};
} // namespace ARBD
