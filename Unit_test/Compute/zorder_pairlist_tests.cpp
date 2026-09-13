#include "../catch_boiler.h"
#include "PatchOperation/PairListKernels/ZOrderPairlist.h"
#include "PatchOperation/ZOrderKernels/ZOrderSort.h"
#include <algorithm>
#include <random>
#include <utility>
#include <vector>

using namespace MARS;

TEST_CASE("ZOrderPairlist Integration", "[zorder][pairlist][integration]") {
	initialize_backend_once();
	const auto& resource = Resource(Global::single_resource_id);
	const size_t max_particles = 200;
	const size_t max_pairs = 1000;
	const float cutoff = 2.0f;

	ZOrderPairlist pairlist(resource, max_particles, max_pairs);
	// The box is the Morton domain and is mandatory; open on every axis here so
	// no wrapping is applied, matching the old particle-extent behaviour.
	PeriodicBox test_box(Vector3(60.0f, 60.0f, 60.0f), false, false, false);
	test_box.set_origin(Vector3(-30.0f, -30.0f, -30.0f));
	pairlist.set_periodic_box(test_box);

	SECTION("Basic Pairlist Construction") {
		REQUIRE(pairlist.get_type() == PairlistBuilderType::ZOrder);
		REQUIRE(std::string(pairlist.get_name()) == "Z-Order Pairlist");
	}

	SECTION("Pairlist Build and Update") {
		// Create a grid of particles
		const size_t num_particles = 64; // 4x4x4 grid
		std::vector<Vector3> positions(num_particles);

		size_t idx = 0;
		for (int i = 0; i < 4; ++i) {
			for (int j = 0; j < 4; ++j) {
				for (int k = 0; k < 4; ++k) {
					positions[idx++] = Vector3(static_cast<float>(i) * 1.5f,
											   static_cast<float>(j) * 1.5f,
											   static_cast<float>(k) * 1.5f);
				}
			}
		}

		DeviceBuffer<Vector3> device_positions(num_particles, resource);
		device_positions.copy_from_host(positions.data(), num_particles);

		// Build initial pairlist
		pairlist.build_pairlist(device_positions, num_particles, cutoff);

		// Check that we found some pairs
		size_t num_pairs = pairlist.get_num_pairs();
		REQUIRE(num_pairs > 0);
		REQUIRE(num_pairs <= max_pairs);

		// Get statistics
		auto stats = pairlist.get_statistics();
		REQUIRE(stats.num_particles == num_particles);
		REQUIRE(stats.num_pairs == num_pairs);
		REQUIRE(stats.build_time_ms >= 0.0);

		// Test update functionality
		std::vector<Vector3> moved_positions = positions;
		for (auto& pos : moved_positions) {
			pos += Vector3(0.01f, 0.01f, 0.01f); // Small movement
		}
		device_positions.copy_from_host(moved_positions.data(), num_particles);

		pairlist.update_pairlist(device_positions, num_particles);

		// Should still have pairs after small movement
		size_t new_num_pairs = pairlist.get_num_pairs();
		REQUIRE(new_num_pairs > 0);
	}

	SECTION("Needs Update Detection") {
		const size_t num_particles = 10;
		std::vector<Vector3> positions(num_particles);
		std::vector<Vector3> old_positions(num_particles);
		std::vector<Vector3> far_positions(num_particles);

		for (size_t i = 0; i < num_particles; ++i) {
			positions[i] = Vector3(static_cast<float>(i), 0.0f, 0.0f);
			old_positions[i] = positions[i];
			far_positions[i] = positions[i] + Vector3(10.0f, 0.0f, 0.0f); // Large movement
		}

		DeviceBuffer<Vector3> device_positions(num_particles, resource);
		DeviceBuffer<Vector3> device_old_positions(num_particles, resource);
		DeviceBuffer<Vector3> device_far_positions(num_particles, resource);

		device_positions.copy_from_host(positions.data(), num_particles);
		device_old_positions.copy_from_host(old_positions.data(), num_particles);
		device_far_positions.copy_from_host(far_positions.data(), num_particles);

		// Build initial pairlist
		pairlist.build_pairlist(device_old_positions, num_particles, cutoff);

		const float skin_distance = 1.0f;

		// Should not need update for same positions
		bool needs_update = pairlist.needs_update(device_positions,
												  device_old_positions,
												  num_particles,
												  skin_distance);
		REQUIRE_FALSE(needs_update);

		// Should need update for far positions
		bool needs_update_far = pairlist.needs_update(device_far_positions,
													  device_old_positions,
													  num_particles,
													  skin_distance);
		REQUIRE(needs_update_far);
	}

	SECTION("Configuration Options") {
		// Test displacement thresholds
		pairlist.set_displacement_thresholds(0.1f, 0.2f);

		// The simulation box is the Morton domain; set it directly rather than
		// deriving one from particle extents.
		PeriodicBox open_box(Vector3(10.0f, 10.0f, 10.0f), false, false, false);
		open_box.set_origin(Vector3(-5.0f, -5.0f, -5.0f));
		pairlist.set_periodic_box(open_box);

		PeriodicBox wrapped_box(Vector3(10.0f, 10.0f, 10.0f), true, true, true);
		wrapped_box.set_origin(Vector3(-5.0f, -5.0f, -5.0f));
		pairlist.set_periodic_box(wrapped_box);
	}

	SECTION("Resize Functionality") {
		const size_t new_max_particles = 400;
		const size_t new_max_pairs = 2000;

		pairlist.resize(new_max_particles, new_max_pairs);

		// Test with larger dataset after resize
		const size_t num_particles = 100;
		std::vector<Vector3> positions(num_particles);

		std::mt19937 rng(42);
		std::uniform_real_distribution<float> dist(0.0f, 10.0f);

		for (size_t i = 0; i < num_particles; ++i) {
			positions[i] = Vector3(dist(rng), dist(rng), dist(rng));
		}

		DeviceBuffer<Vector3> device_positions(num_particles, resource);
		device_positions.copy_from_host(positions.data(), num_particles);

		// Should work without issues after resize
		pairlist.build_pairlist(device_positions, num_particles, cutoff);

		auto stats = pairlist.get_statistics();
		REQUIRE(stats.num_particles == num_particles);
		REQUIRE(stats.num_pairs <= new_max_pairs);
	}

	SECTION("Access to Underlying Sorter") {
		const auto& sorter = pairlist.get_sorter();
		REQUIRE(sorter.get_optimization_mode() == ZOrderOptimizationMode::Pairlist);
	}

	SECTION("Sorted Position Access") {
		const size_t num_particles = 20;
		std::vector<Vector3> positions(num_particles);

		for (size_t i = 0; i < num_particles; ++i) {
			positions[i] = Vector3(static_cast<float>(i), 0.0f, 0.0f);
		}

		DeviceBuffer<Vector3> device_positions(num_particles, resource);
		device_positions.copy_from_host(positions.data(), num_particles);

		pairlist.build_pairlist(device_positions, num_particles, cutoff);

		const auto& sorted_positions = pairlist.get_sorted_positions();
		REQUIRE(sorted_positions.size() >= num_particles);
	}
}

namespace {

/// @brief Read the built pairlist back as a sorted list of (lo, hi) pairs.
std::vector<std::pair<int, int>> read_pairs(const Pairlist& pairlist) {
	const size_t n = pairlist.get_num_pairs();
	std::vector<std::pair<int, int>> out;
	if (n == 0) {
		return out;
	}
	std::vector<MARS::int2> raw(n);
	pairlist.get_neighbor_pairs().copy_to_host(raw.data(), n);
	out.reserve(n);
	for (const auto& p : raw) {
		out.emplace_back(p.x < p.y ? p.x : p.y, p.x < p.y ? p.y : p.x);
	}
	std::sort(out.begin(), out.end());
	return out;
}

} // namespace

TEST_CASE("ZOrderPairlist drops excluded pairs at build time", "[zorder][pairlist][exclusions]") {
	initialize_backend_once();
	const auto& resource = Resource(Global::single_resource_id);

	ZOrderPairlist pairlist(resource, 64, 256);
	PeriodicBox box(Vector3(60.0f, 60.0f, 60.0f), false, false, false);
	box.set_origin(Vector3(-30.0f, -30.0f, -30.0f));
	pairlist.set_periodic_box(box);

	// Four collinear particles one unit apart. At cutoff 2.5 every pair except
	// (0,3) is in range, so the unfiltered list is a known set of five.
	const size_t num_particles = 4;
	const std::vector<Vector3> positions{Vector3(0.0f, 0.0f, 0.0f),
										 Vector3(1.0f, 0.0f, 0.0f),
										 Vector3(2.0f, 0.0f, 0.0f),
										 Vector3(3.0f, 0.0f, 0.0f)};
	DeviceBuffer<Vector3> device_positions(num_particles, resource);
	device_positions.copy_from_host(positions.data(), num_particles);
	const float cutoff = 2.5f;

	const std::vector<std::pair<int, int>> all_pairs{{0, 1}, {0, 2}, {1, 2}, {1, 3}, {2, 3}};

	SECTION("Empty view excludes nothing") {
		pairlist.set_exclusions(ExclusionView{});
		pairlist.build_pairlist(device_positions, num_particles, cutoff);
		REQUIRE(read_pairs(pairlist) == all_pairs);
	}

	SECTION("Excluded pairs never reach the list") {
		// Symmetric CSR over all four particles excluding (0,1) and (1,3):
		//   p0 -> {1}   p1 -> {0, 3}   p2 -> {}   p3 -> {1}
		const std::vector<int> offsets{0, 1, 3, 3, 4};
		const std::vector<int> excluded{1, 0, 3, 1};

		DeviceBuffer<int> off_buf(offsets.size(), resource);
		off_buf.copy_from_host(offsets.data(), offsets.size());
		DeviceBuffer<int> excl_buf(excluded.size(), resource);
		excl_buf.copy_from_host(excluded.data(), excluded.size());

		pairlist.set_exclusions(
			ExclusionView{off_buf.data(), excl_buf.data(), static_cast<idx_t>(num_particles)});
		pairlist.build_pairlist(device_positions, num_particles, cutoff);

		const std::vector<std::pair<int, int>> expected{{0, 2}, {1, 2}, {2, 3}};
		REQUIRE(read_pairs(pairlist) == expected);
		REQUIRE(pairlist.get_num_pairs() == expected.size());
	}

	SECTION("Either endpoint's row alone is enough to drop the pair") {
		// The builder scans only the row of the endpoint whose thread emits the
		// pair, and Morton order decides which that is. A symmetric CSR covering
		// every particle named in an exclusion makes that choice irrelevant, and
		// DeviceBondedInteractions always builds one (num_excl_particles is the
		// largest excluded index plus one). Here only particle 3's row lists the
		// (1,3) exclusion, so the pair survives -- which is what makes storing
		// each exclusion under both partners load-bearing rather than redundant.
		const std::vector<int> offsets{0, 1, 2, 2, 3};
		const std::vector<int> excluded{1, 0, 1};

		DeviceBuffer<int> off_buf(offsets.size(), resource);
		off_buf.copy_from_host(offsets.data(), offsets.size());
		DeviceBuffer<int> excl_buf(excluded.size(), resource);
		excl_buf.copy_from_host(excluded.data(), excluded.size());

		pairlist.set_exclusions(
			ExclusionView{off_buf.data(), excl_buf.data(), static_cast<idx_t>(num_particles)});
		pairlist.build_pairlist(device_positions, num_particles, cutoff);

		// (0,1) is listed under both 0 and 1, so it goes. (1,3) is listed only
		// under 3, and particle 1 emits it, so it stays.
		const std::vector<std::pair<int, int>> expected{{0, 2}, {1, 2}, {1, 3}, {2, 3}};
		REQUIRE(read_pairs(pairlist) == expected);
	}

	SECTION("Excluding every in-range pair empties the list") {
		//   p0 -> {1, 2}   p1 -> {0, 2, 3}   p2 -> {0, 1, 3}   p3 -> {1, 2}
		const std::vector<int> offsets{0, 2, 5, 8, 10};
		const std::vector<int> excluded{1, 2, 0, 2, 3, 0, 1, 3, 1, 2};

		DeviceBuffer<int> off_buf(offsets.size(), resource);
		off_buf.copy_from_host(offsets.data(), offsets.size());
		DeviceBuffer<int> excl_buf(excluded.size(), resource);
		excl_buf.copy_from_host(excluded.data(), excluded.size());

		pairlist.set_exclusions(
			ExclusionView{off_buf.data(), excl_buf.data(), static_cast<idx_t>(num_particles)});
		pairlist.build_pairlist(device_positions, num_particles, cutoff);

		REQUIRE(pairlist.get_num_pairs() == 0);
	}

	SECTION("Particles sharing a rigid body never pair") {
		// 0 and 1 on body 7, 2 on body 9, 3 unattached. Only (0,1) shares a body.
		const std::vector<int> body{7, 7, 9, -1};
		DeviceBuffer<int> body_buf(body.size(), resource);
		body_buf.copy_from_host(body.data(), body.size());

		pairlist.set_exclusions(ExclusionView{nullptr, nullptr, 0, body_buf.data()});
		pairlist.build_pairlist(device_positions, num_particles, cutoff);

		const std::vector<std::pair<int, int>> expected{{0, 2}, {1, 2}, {1, 3}, {2, 3}};
		REQUIRE(read_pairs(pairlist) == expected);
	}

	SECTION("Unattached particles are never excluded by the body test") {
		// Every particle unattached: -1 must not match -1.
		const std::vector<int> body{-1, -1, -1, -1};
		DeviceBuffer<int> body_buf(body.size(), resource);
		body_buf.copy_from_host(body.data(), body.size());

		pairlist.set_exclusions(ExclusionView{nullptr, nullptr, 0, body_buf.data()});
		pairlist.build_pairlist(device_positions, num_particles, cutoff);

		REQUIRE(read_pairs(pairlist) == all_pairs);
	}

	SECTION("Body test and CSR compose") {
		// Body drops (0,1); the CSR drops (2,3). Both must apply.
		const std::vector<int> body{5, 5, -1, -1};
		DeviceBuffer<int> body_buf(body.size(), resource);
		body_buf.copy_from_host(body.data(), body.size());

		//   p0 -> {}   p1 -> {}   p2 -> {3}   p3 -> {2}
		const std::vector<int> offsets{0, 0, 0, 1, 2};
		const std::vector<int> excluded{3, 2};
		DeviceBuffer<int> off_buf(offsets.size(), resource);
		off_buf.copy_from_host(offsets.data(), offsets.size());
		DeviceBuffer<int> excl_buf(excluded.size(), resource);
		excl_buf.copy_from_host(excluded.data(), excluded.size());

		pairlist.set_exclusions(ExclusionView{off_buf.data(),
											  excl_buf.data(),
											  static_cast<idx_t>(num_particles),
											  body_buf.data()});
		pairlist.build_pairlist(device_positions, num_particles, cutoff);

		const std::vector<std::pair<int, int>> expected{{0, 2}, {1, 2}, {1, 3}};
		REQUIRE(read_pairs(pairlist) == expected);
	}
}
