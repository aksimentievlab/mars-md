#include "../catch_boiler.h"
#include "Compute/ZOrderPairlist.h"
#include <vector>
#include <random>

using namespace ARBD;

TEST_CASE("ZOrderPairlist Integration", "[zorder][pairlist][integration]") {
	Tests::TestBackendManager& manager = Tests::TestBackendManager::getInstance();
	REQUIRE(manager.isInitialized());

	const auto& resource = manager.get_resource();
	const size_t max_particles = 200;
	const size_t max_pairs = 1000;
	const float cutoff = 2.0f;

	ZOrderPairlist pairlist(resource, max_particles, max_pairs);

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
					positions[idx++] = Vector3(
						static_cast<float>(i) * 1.5f,
						static_cast<float>(j) * 1.5f,
						static_cast<float>(k) * 1.5f
					);
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
		bool needs_update = pairlist.needs_update(device_positions, device_old_positions, num_particles, skin_distance);
		REQUIRE_FALSE(needs_update);

		// Should need update for far positions
		bool needs_update_far = pairlist.needs_update(device_far_positions, device_old_positions, num_particles, skin_distance);
		REQUIRE(needs_update_far);
	}

	SECTION("Configuration Options") {
		// Test search range configuration
		pairlist.set_search_range(128);
		REQUIRE(pairlist.get_search_range() == 128);

		// Test displacement thresholds
		pairlist.set_displacement_thresholds(0.1f, 0.2f);

		// Test adaptive ranges
		pairlist.set_adaptive_ranges(true);
		REQUIRE(pairlist.is_using_adaptive_ranges());

		pairlist.set_adaptive_ranges(false);
		REQUIRE_FALSE(pairlist.is_using_adaptive_ranges());

		// Test hierarchical search
		pairlist.set_hierarchical_search(true);
		REQUIRE(pairlist.is_using_hierarchical_search());

		pairlist.set_hierarchical_search(false);
		REQUIRE_FALSE(pairlist.is_using_hierarchical_search());

		// Test bounding box mode
		Vector3 manual_min(-5.0f, -5.0f, -5.0f);
		Vector3 manual_max(5.0f, 5.0f, 5.0f);
		pairlist.set_bounding_box_mode(false, manual_min, manual_max);

		pairlist.set_bounding_box_mode(true); // Back to auto mode
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