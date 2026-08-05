/**
 * @file RigidBodyForcePairs.cpp
 * @brief Tests for Phase 3 host-side force-pair construction
 *        (Objects/RigidBodyForcePairs.h), the port of legacy
 *        RigidBodyController::initializeForcePairs.
 *
 * Pure host bookkeeping - no GridManager, no grid loading. Grid identity is
 * plain grid_ids the caller supplies (standing in for what GridManager would
 * have assigned at config-parse time); format is a simple lookup table rather
 * than a real GridManager, matching the style of Interactions/Pmf.cpp.
 */

#include "Objects/RigidBodyForcePairs.h"
#include "../catch_boiler.h"
#include <unordered_map>

using namespace ARBD;

namespace {

// grid_id -> GridFormat lookup, standing in for GridManager::get_grid_format.
std::function<GridFormat(int)> make_format_table(std::unordered_map<int, GridFormat> table) {
	return [table = std::move(table)](int grid_id) {
		auto it = table.find(grid_id);
		REQUIRE(it != table.end());
		return it->second;
	};
}

// Mirrors legacy's two-type config: a "Protein" type carrying density grids
// ("Elec", "SF6") and a PMF ("Elec"), paired against a "Membrane" type
// carrying a potential grid ("Elec"). Only "Elec" is a shared key name.
std::vector<RigidBodyType> two_type_config() {
	std::vector<RigidBodyType> types(2);

	types[0].name = "Protein";
	types[0].id = 0;
	types[0].density_grid_keys = {"Elec", "SF6"};
	types[0].density_grids = {GridTerm{100}, GridTerm{101}};
	types[0].pmf_keys = {"Elec"};
	types[0].pmf_grids = {GridTerm{300}};

	types[1].name = "Membrane";
	types[1].id = 1;
	types[1].potential_grid_keys = {"Elec"};
	types[1].potential_grids = {GridTerm{200}};

	return types;
}

} // namespace

TEST_CASE("RigidBodyForcePairList: grid-key matching reproduces legacy pairing",
		  "[rigidbody][forcepairs]") {
	const std::vector<RigidBodyType> types = two_type_config();
	const auto grid_format = make_format_table({{100, GridFormat::Dense},
												{101, GridFormat::Dense},
												{200, GridFormat::Dense},
												{300, GridFormat::Dense}});

	RigidBodyForcePairList pairs;
	pairs.build(types, grid_format, /*update_period=*/4);

	// One type-type match ("Elec" between Protein's density and Membrane's
	// potential) and one type-PMF match ("Elec" between Protein's density and
	// its own PMF). "SF6" has no counterpart anywhere, so it contributes
	// nothing - legacy only pairs on an exact key-name match.
	REQUIRE(pairs.size() == 2);

	bool found_type_pair = false;
	bool found_pmf_pair = false;
	for (const RigidBodyGridPair& p : pairs.dense_pairs()) {
		CHECK(p.update_period == 4);
		if (!p.is_pmf) {
			found_type_pair = true;
			CHECK(p.type_i == 0);
			CHECK(p.type_j == 1);
			CHECK(p.grid_id_rho == 100); // Protein's "Elec" density
			CHECK(p.grid_id_u == 200);	 // Membrane's "Elec" potential
		} else {
			found_pmf_pair = true;
			CHECK(p.type_i == 0);
			CHECK(p.type_j == 0);
			CHECK(p.grid_id_rho == 100); // Protein's "Elec" density
			CHECK(p.grid_id_u == 300);	 // Protein's "Elec" pmf
		}
	}
	CHECK(found_type_pair);
	CHECK(found_pmf_pair);
}

TEST_CASE("RigidBodyForcePairList: legacy only checks density(ti) against potential(tj>=ti)",
		  "[rigidbody][forcepairs]") {
	// Swap which type carries density vs potential relative to index order:
	// type 0 ("Membrane") only has a potential grid, type 1 ("Protein") only
	// has a density grid. Legacy's loop (tj starts at ti) never checks
	// density(t1) against potential(t0), so no pair should be emitted.
	std::vector<RigidBodyType> types(2);
	types[0].name = "Membrane";
	types[0].potential_grid_keys = {"Elec"};
	types[0].potential_grids = {GridTerm{200}};
	types[1].name = "Protein";
	types[1].density_grid_keys = {"Elec"};
	types[1].density_grids = {GridTerm{100}};

	const auto grid_format =
		make_format_table({{100, GridFormat::Dense}, {200, GridFormat::Dense}});

	RigidBodyForcePairList pairs;
	pairs.build(types, grid_format);
	CHECK(pairs.size() == 0);
}

TEST_CASE("RigidBodyForcePairList: format-uniformity check fires on a mismatched pair",
		  "[rigidbody][forcepairs]") {
	const std::vector<RigidBodyType> types = two_type_config();
	// Protein's "Elec" density (100) is Dense, Membrane's "Elec" potential
	// (200) is (hypothetically) Sparse - the type-type pair must throw before
	// ever reaching the PMF pass.
	const auto grid_format = make_format_table({{100, GridFormat::Dense},
												{101, GridFormat::Dense},
												{200, GridFormat::Sparse},
												{300, GridFormat::Dense}});

	RigidBodyForcePairList pairs;
	REQUIRE_THROWS_AS(pairs.build(types, grid_format), ARBD::Exception);
}
