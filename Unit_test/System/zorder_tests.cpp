#include "../catch_boiler.h"
#include "System/ZOrderKernels.h"
#include "System/ZOrderSort.h"
#include <random>
#include <vector>

using namespace ARBD;

TEST_CASE("ZOrderSort Basic Functionality", "[zorder][system]") {
	Tests::TestBackendManager& manager = Tests::TestBackendManager::getInstance();
	REQUIRE(manager.isInitialized());

	const auto& resource = manager.get_resource();
	const size_t max_particles = 1000;

	SECTION("System Mode Construction") {
		ZOrderSort sorter(resource, max_particles, ZOrderOptimizationMode::System);
		REQUIRE(sorter.get_optimization_mode() == ZOrderOptimizationMode::System);
		REQUIRE(sorter.get_num_particles() == 0);
	}

	SECTION("Pairlist Mode Construction") {
		ZOrderSort sorter(resource, max_particles, ZOrderOptimizationMode::Pairlist);
		REQUIRE(sorter.get_optimization_mode() == ZOrderOptimizationMode::Pairlist);
		REQUIRE(sorter.get_num_particles() == 0);
	}
}

TEST_CASE("ZOrderSort Particle Sorting", "[zorder][system]") {
	Tests::TestBackendManager& manager = Tests::TestBackendManager::getInstance();
	REQUIRE(manager.isInitialized());

	const auto& resource = manager.get_resource();
	const size_t num_particles = 100;
	ZOrderSort sorter(resource, num_particles, ZOrderOptimizationMode::System);

	// Create test particles in a cube
	std::vector<Vector3> positions(num_particles);
	std::mt19937 rng(42);
	std::uniform_real_distribution<float> dist(0.0f, 10.0f);

	for (size_t i = 0; i < num_particles; ++i) {
		positions[i] = Vector3(dist(rng), dist(rng), dist(rng));
	}

	DeviceBuffer<Vector3> device_positions(num_particles, resource);
	device_positions.copy_from_host(positions.data(), num_particles);

	Vector3 box_min(0.0f, 0.0f, 0.0f);
	Vector3 box_max(10.0f, 10.0f, 10.0f);

	SECTION("Basic Sorting") {
		sorter.sort_particles(device_positions, num_particles, box_min, box_max);

		REQUIRE(sorter.get_num_particles() == num_particles);

		// Verify we have Morton codes and indices
		const auto& morton_codes = sorter.get_morton_codes();
		const auto& sorted_indices = sorter.get_sorted_indices();
		const auto& inverse_indices = sorter.get_inverse_indices();

		REQUIRE(morton_codes.size() >= num_particles);
		REQUIRE(sorted_indices.size() >= num_particles);
		REQUIRE(inverse_indices.size() >= num_particles);
	}

	SECTION("Sort Validation") {
		sorter.sort_particles(device_positions, num_particles, box_min, box_max);

		uint32_t errors = sorter.validate_sorting();
		REQUIRE(errors == 0);
	}

	SECTION("Data Reordering") {
		// Create some test data to reorder
		std::vector<float> test_data(num_particles);
		for (size_t i = 0; i < num_particles; ++i) {
			test_data[i] = static_cast<float>(i);
		}

		DeviceBuffer<float> device_data(num_particles, resource);
		DeviceBuffer<float> reordered_data(num_particles, resource);
		device_data.copy_from_host(test_data.data(), num_particles);

		sorter.sort_particles(device_positions, num_particles, box_min, box_max);
		sorter.reorder_data(device_data, reordered_data, num_particles);

		// Copy back and verify the data was reordered
		std::vector<float> result(num_particles);
		reordered_data.copy_to_host(result.data(), num_particles, true);

		// Data should be different from original order (unless very unlikely)
		bool reordered = false;
		for (size_t i = 0; i < num_particles; ++i) {
			if (result[i] != test_data[i]) {
				reordered = true;
				break;
			}
		}
		REQUIRE(reordered);
	}
}

TEST_CASE("ZOrderSort Pairlist Mode Features", "[zorder][pairlist]") {
	Tests::TestBackendManager& manager = Tests::TestBackendManager::getInstance();
	REQUIRE(manager.isInitialized());

	const auto& resource = manager.get_resource();
	const size_t num_particles = 50;
	ZOrderSort sorter(resource, num_particles, ZOrderOptimizationMode::Pairlist);

	// Create test particles
	std::vector<Vector3> positions(num_particles);
	std::vector<Vector3> new_positions(num_particles);

	for (size_t i = 0; i < num_particles; ++i) {
		positions[i] = Vector3(static_cast<float>(i), 0.0f, 0.0f);
		new_positions[i] = positions[i] + Vector3(0.01f, 0.0f, 0.0f); // Small displacement
	}

	DeviceBuffer<Vector3> device_positions(num_particles, resource);
	DeviceBuffer<Vector3> device_new_positions(num_particles, resource);
	device_positions.copy_from_host(positions.data(), num_particles);
	device_new_positions.copy_from_host(new_positions.data(), num_particles);

	Vector3 box_min(-1.0f, -1.0f, -1.0f);
	Vector3 box_max(static_cast<float>(num_particles) + 1.0f, 1.0f, 1.0f);

	SECTION("Smart Updates Configuration") {
		sorter.enable_smart_updates(true);
		sorter.set_displacement_thresholds(0.05f, 0.1f);

		// Initial sort
		sorter.sort_particles(device_positions, num_particles, box_min, box_max);

		// Note: update_positions_incremental and compute_max_displacement are private methods
		// These would be tested through the public interface in real usage

		// For now, just verify the sorter was created with the right mode
		REQUIRE(sorter.get_optimization_mode() == ZOrderOptimizationMode::Pairlist);
	}

	SECTION("Morton Code Validation") {
		sorter.sort_particles(device_positions, num_particles, box_min, box_max);

		// Note: validate_morton_codes is also a private method
		// In real usage, this would be tested through the public API
		// For now, just verify the sort completed without error
		REQUIRE(sorter.get_num_particles() == num_particles);
	}
}

TEST_CASE("ZOrderSort Resize Functionality", "[zorder][system]") {
	Tests::TestBackendManager& manager = Tests::TestBackendManager::getInstance();
	REQUIRE(manager.isInitialized());

	const auto& resource = manager.get_resource();
	const size_t initial_particles = 100;
	const size_t new_particles = 200;

	ZOrderSort sorter(resource, initial_particles, ZOrderOptimizationMode::System);

	SECTION("Resize Buffers") {
		sorter.resize(new_particles);

		// Create larger dataset
		std::vector<Vector3> positions(new_particles);
		for (size_t i = 0; i < new_particles; ++i) {
			positions[i] = Vector3(static_cast<float>(i % 10), static_cast<float>(i / 10), 0.0f);
		}

		DeviceBuffer<Vector3> device_positions(new_particles, resource);
		device_positions.copy_from_host(positions.data(), new_particles);

		Vector3 box_min(0.0f, 0.0f, -1.0f);
		Vector3 box_max(10.0f, 20.0f, 1.0f);

		// Should work without issues after resize
		sorter.sort_particles(device_positions, new_particles, box_min, box_max);
		REQUIRE(sorter.get_num_particles() == new_particles);

		uint32_t errors = sorter.validate_sorting();
		REQUIRE(errors == 0);
	}
}

TEST_CASE("ZOrderSort Error Handling", "[zorder][error]") {
	Tests::TestBackendManager& manager = Tests::TestBackendManager::getInstance();
	REQUIRE(manager.isInitialized());

	const auto& resource = manager.get_resource();
	const size_t max_particles = 100;

	ZOrderSort sorter(resource, max_particles, ZOrderOptimizationMode::System);

	SECTION("Particle Count Validation") {
		std::vector<Vector3> positions(max_particles + 50); // Too many particles
		DeviceBuffer<Vector3> device_positions(max_particles + 50, resource);

		Vector3 box_min(0.0f);
		Vector3 box_max(1.0f);

		// Should throw exception for too many particles
		REQUIRE_THROWS_AS(
			sorter.sort_particles(device_positions, max_particles + 50, box_min, box_max),
			Exception);
	}

	SECTION("Mode Verification") {
		ZOrderSort system_sorter(resource, max_particles, ZOrderOptimizationMode::System);

		// Note: The private method exception tests have been removed since these methods
		// are private and not part of the public API. In real usage, mode-specific behavior
		// would be tested through the public interfaces.
		REQUIRE(system_sorter.get_optimization_mode() == ZOrderOptimizationMode::System);
	}
}
