/**
 * @file NonbondedEnergy.cpp
 * @brief Energy tests for launch_pairwise_nonbonded (tabulated pair potentials).
 * @see NonbondedEnergy.md
 */

#include "../catch_boiler.h"

#include "Backend/Resource.h"
#include "Interactions/Nonbonded/Pairwise.h"
#include "System/PeriodicBox.h"
#include "Types/Types.h"

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

using namespace ARBD;
// ============================================================================
// TABULATED PAIRWISE NONBONDED ENERGY
// See NonbondedEnergy.md
// ============================================================================

namespace {

constexpr size_t NB_TABLE_N = 2048;
constexpr arbd_real NB_RMAX = arbd_real(10.0);

/// @brief Tabulated pair potential sampled from an arbitrary U(r).
struct PairTable {
	std::vector<arbd_real> pot;
	arbd_real step;

	template<typename Fn>
	explicit PairTable(Fn u) : pot(NB_TABLE_N), step(NB_RMAX / arbd_real(NB_TABLE_N)) {
		for (size_t i = 0; i < NB_TABLE_N; ++i) {
			pot[i] = u(step * arbd_real(i));
		}
	}

	/// @brief Energy the kernel's linear interpolation will report at r.
	arbd_real interpolated(arbd_real r) const {
		const arbd_real w = r / step;
		const int home = static_cast<int>(std::floor(w));
		if (home >= static_cast<int>(NB_TABLE_N) - 1) {
			return pot[NB_TABLE_N - 1];
		}
		const int h = home < 0 ? 0 : home;
		return (pot[h + 1] - pot[h]) * (w - static_cast<arbd_real>(home)) + pot[h];
	}

	TabulatedPotential descriptor(const arbd_real* device_pot) const {
		TabulatedPotential t{};
		t.pot = const_cast<arbd_real*>(device_pot);
		t.step_inv = arbd_real(1) / step;
		t.size = static_cast<unsigned int>(pot.size());
		t.start = arbd_real(0);
		t.is_periodic = false;
		return t;
	}
};

/// @brief Total nonbonded energy from launch_pairwise_nonbonded.
/// @return {sum over particles of ForceEnergy.t, per-particle energies}
std::pair<arbd_real, std::vector<Vector3>> run_pairwise(const Resource& res,
														const std::vector<Vector3>& positions,
														const std::vector<int>& type_ids,
														const std::vector<ARBD::int2>& pairs,
														const PairTable& table,
														int num_types,
														const std::vector<int>& excl_offsets,
														const std::vector<int>& excl_neighbors,
														arbd_real cutoff) {
	const idx_t n = static_cast<idx_t>(positions.size());

	DeviceBuffer<arbd_real> pot(table.pot.size(), res);
	pot.copy_from_host(table.pot.data(), table.pot.size());
	const TabulatedPotential desc = table.descriptor(pot.data());
	DeviceBuffer<TabulatedPotential> tables(1, res);
	tables.copy_from_host(&desc, 1);

	DeviceBuffer<Vector3> pos_buf(n, res);
	pos_buf.copy_from_host(positions.data(), n);
	std::vector<Vector3> zeroed(n, Vector3(0.0f, 0.0f, 0.0f));
	DeviceBuffer<Vector3> force_buf(n, res);
	force_buf.copy_from_host(zeroed.data(), n);

	DeviceBuffer<int> types_buf(n, res);
	types_buf.copy_from_host(type_ids.data(), n);

	DeviceBuffer<ARBD::int2> pairs_buf(pairs.size(), res);
	pairs_buf.copy_from_host(pairs.data(), pairs.size());

	// Every type pair maps to table 0 and is Tabulated.
	std::vector<int> table_matrix(static_cast<size_t>(num_types) * num_types, 0);
	std::vector<int> form_matrix(table_matrix.size(), static_cast<int>(InteractionForm::Tabulated));
	DeviceBuffer<int> table_matrix_buf(table_matrix.size(), res);
	table_matrix_buf.copy_from_host(table_matrix.data(), table_matrix.size());
	DeviceBuffer<int> form_matrix_buf(form_matrix.size(), res);
	form_matrix_buf.copy_from_host(form_matrix.data(), form_matrix.size());

	// Both buffers are allocated non-empty even when there are no exclusions:
	// a zero-length DeviceBuffer is null, and copy_from_host rejects that.
	// num_excl_particles below is what actually disables the exclusion scan.
	DeviceBuffer<int> excl_off_buf(std::max<size_t>(excl_offsets.size(), 1), res);
	if (!excl_offsets.empty()) {
		excl_off_buf.copy_from_host(excl_offsets.data(), excl_offsets.size());
	}
	DeviceBuffer<int> excl_nbr_buf(std::max<size_t>(excl_neighbors.size(), 1), res);
	if (!excl_neighbors.empty()) {
		excl_nbr_buf.copy_from_host(excl_neighbors.data(), excl_neighbors.size());
	}
	const idx_t num_excl_particles =
		excl_offsets.empty() ? 0 : static_cast<idx_t>(excl_offsets.size() - 1);

	PeriodicBox box_host(Vector3(1000.0f, 1000.0f, 1000.0f));
	DeviceBuffer<PeriodicBox> box_buf(1, res);
	box_buf.copy_from_host(&box_host, 1);

	launch_pairwise_nonbonded(res,
							  pairs_buf.data(),
							  pos_buf.data(),
							  force_buf.data(),
							  types_buf.data(),
							  table_matrix_buf.data(),
							  form_matrix_buf.data(),
							  tables.data(),
							  static_cast<idx_t>(num_types),
							  excl_off_buf.data(),
							  excl_nbr_buf.data(),
							  num_excl_particles,
							  box_buf.data(),
							  /*get_energy=*/true,
							  static_cast<idx_t>(pairs.size()),
							  cutoff > 0 ? cutoff * cutoff : arbd_real(0))
		.wait();

	std::vector<Vector3> out(n);
	force_buf.copy_to_host(out.data(), n);
	arbd_real total = arbd_real(0);
	for (const auto& v : out) {
		total += v.t;
	}
	return {total, out};
}

} // namespace

TEST_CASE("Pairwise nonbonded energy of one pair equals U(r)",
		  "[forces][nonbonded][tabulated][energy]") {
	initialize_backend_once();
	Resource res(Global::single_resource_id);

	// U(r) = (r - 3)^2 - 1: has a genuine negative well, so this also pins
	// down that a negative table yields a negative reported energy.
	const PairTable table(
		[](arbd_real r) { return (r - arbd_real(3)) * (r - arbd_real(3)) - arbd_real(1); });
	const arbd_real r = arbd_real(3.5);

	auto [total, per_particle] =
		run_pairwise(res,
					 {Vector3(0.0f, 0.0f, 0.0f), Vector3(float(r), 0.0f, 0.0f)},
					 {0, 0},
					 {ARBD::int2{0, 1}},
					 table,
					 1,
					 {},
					 {},
					 arbd_real(0));

	const arbd_real expected = table.interpolated(r);
	REQUIRE(total == Catch::Approx(expected).epsilon(1e-4));
	// Each endpoint carries exactly half, so the particle sum is U, not 2U.
	REQUIRE(per_particle[0].t == Catch::Approx(expected / 2).epsilon(1e-4));
	REQUIRE(per_particle[1].t == Catch::Approx(expected / 2).epsilon(1e-4));
}

TEST_CASE("Pairwise nonbonded energy is negative inside an attractive well",
		  "[forces][nonbonded][tabulated][energy]") {
	initialize_backend_once();
	Resource res(Global::single_resource_id);

	const PairTable table(
		[](arbd_real r) { return (r - arbd_real(3)) * (r - arbd_real(3)) - arbd_real(1); });

	auto [at_min, ignored1] = run_pairwise(res,
										   {Vector3(0.0f, 0.0f, 0.0f), Vector3(3.0f, 0.0f, 0.0f)},
										   {0, 0},
										   {ARBD::int2{0, 1}},
										   table,
										   1,
										   {},
										   {},
										   arbd_real(0));
	auto [on_wall, ignored2] = run_pairwise(res,
											{Vector3(0.0f, 0.0f, 0.0f), Vector3(1.0f, 0.0f, 0.0f)},
											{0, 0},
											{ARBD::int2{0, 1}},
											table,
											1,
											{},
											{},
											arbd_real(0));
	(void)ignored1;
	(void)ignored2;

	REQUIRE(at_min == Catch::Approx(arbd_real(-1)).epsilon(1e-3));
	REQUIRE(on_wall > arbd_real(0));
}

TEST_CASE("Pairwise nonbonded energy matches a direct CPU pair sum",
		  "[forces][nonbonded][tabulated][energy]") {
	initialize_backend_once();
	Resource res(Global::single_resource_id);

	const PairTable table(
		[](arbd_real r) { return (r - arbd_real(3)) * (r - arbd_real(3)) - arbd_real(1); });

	// A jittered lattice: many distinct separations, none degenerate.
	std::vector<Vector3> pos;
	for (int i = 0; i < 4; ++i) {
		for (int j = 0; j < 4; ++j) {
			pos.emplace_back(2.1f * i + 0.13f * j, 1.9f * j - 0.07f * i, 0.31f * (i - j));
		}
	}
	const int n = static_cast<int>(pos.size());
	std::vector<int> types(n, 0);

	std::vector<ARBD::int2> pairs;
	for (int i = 0; i < n; ++i) {
		for (int j = i + 1; j < n; ++j) {
			pairs.push_back(ARBD::int2{i, j});
		}
	}

	auto [total, per_particle] =
		run_pairwise(res, pos, types, pairs, table, 1, {}, {}, arbd_real(0));

	arbd_real expected = arbd_real(0);
	for (const auto& p : pairs) {
		const Vector3 d = pos[p.y] - pos[p.x];
		expected += table.interpolated(static_cast<arbd_real>(d.length()));
	}

	INFO("pairs = " << pairs.size());
	REQUIRE(total == Catch::Approx(expected).epsilon(1e-3));
}

TEST_CASE("Pairwise nonbonded energy drops excluded pairs",
		  "[forces][nonbonded][tabulated][energy][exclusions]") {
	initialize_backend_once();
	Resource res(Global::single_resource_id);

	const PairTable table(
		[](arbd_real r) { return (r - arbd_real(3)) * (r - arbd_real(3)) - arbd_real(1); });
	const std::vector<Vector3> pos{Vector3(0.0f, 0.0f, 0.0f),
								   Vector3(3.5f, 0.0f, 0.0f),
								   Vector3(7.0f, 0.0f, 0.0f)};
	const std::vector<int> types(3, 0);
	const std::vector<ARBD::int2> pairs{ARBD::int2{0, 1}, ARBD::int2{0, 2}, ARBD::int2{1, 2}};

	auto [unexcluded, ignored] =
		run_pairwise(res, pos, types, pairs, table, 1, {}, {}, arbd_real(0));
	(void)ignored;

	// CSR over 3 particles excluding the (0,1) pair, stored on both endpoints.
	const std::vector<int> excl_offsets{0, 1, 2, 2};
	const std::vector<int> excl_neighbors{1, 0};
	auto [excluded, ignored2] =
		run_pairwise(res, pos, types, pairs, table, 1, excl_offsets, excl_neighbors, arbd_real(0));
	(void)ignored2;

	const arbd_real dropped = table.interpolated(arbd_real(3.5));
	REQUIRE(excluded == Catch::Approx(unexcluded - dropped).epsilon(1e-3));
}

TEST_CASE("Pairwise nonbonded energy drops pairs beyond the cutoff",
		  "[forces][nonbonded][tabulated][energy][cutoff]") {
	initialize_backend_once();
	Resource res(Global::single_resource_id);

	const PairTable table(
		[](arbd_real r) { return (r - arbd_real(3)) * (r - arbd_real(3)) - arbd_real(1); });
	const std::vector<Vector3> pos{Vector3(0.0f, 0.0f, 0.0f),
								   Vector3(3.5f, 0.0f, 0.0f),
								   Vector3(8.0f, 0.0f, 0.0f)};
	const std::vector<int> types(3, 0);
	const std::vector<ARBD::int2> pairs{ARBD::int2{0, 1}, ARBD::int2{0, 2}, ARBD::int2{1, 2}};

	// Cutoff 5 keeps (0,1) at 3.5 and (1,2) at 4.5, drops (0,2) at 8.
	auto [total, ignored] = run_pairwise(res, pos, types, pairs, table, 1, {}, {}, arbd_real(5));
	(void)ignored;

	const arbd_real expected =
		table.interpolated(arbd_real(3.5)) + table.interpolated(arbd_real(4.5));
	REQUIRE(total == Catch::Approx(expected).epsilon(1e-3));
}
