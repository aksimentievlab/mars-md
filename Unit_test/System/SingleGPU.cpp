#include "../catch_boiler.h"
#include "System/PatchManager.h"
#include "System/SimSystem.h"
#include "System/SystemState.h"
#include <vector>
short single_resource_id = Global::single_resource_id;
using namespace ARBD;
using namespace Tests;
std::vector<Resource> device_resources = {Resource(0), Resource(1)};
std::vector<Resource> single_resource_list = {Resource(single_resource_id)};
/**
 * @brief Test suite for single resource optimization in SimSystem
 *
 * This test verifies that when only one computational resource is available,
 * SimSystem bypasses the decomposer and creates a simplified single-patch system.
 */
TEST_CASE("SimSystem Single Resource Optimization", "[SimSystem][SingleResource][System]") {

	// Initialize backend first
	initialize_backend_once();

	SECTION("Single resource system bypasses decomposer") {
		// Create SimSystem with single resource
		SimSystem sim_system(single_resource_list);

		// Configure basic simulation parameters
		sim_system.set_box_size(100.0f, 100.0f, 100.0f);
		sim_system.set_cutoff(5.0f);
		sim_system.set_pairlist_cutoff(10.0f);
		sim_system.set_timestep(0.001f);
		sim_system.set_num_steps(100);
		sim_system.set_estimated_particles(1000);

		// Note: We don't set a decomposer for single resource case
		// This should work without throwing an error

		// Create a simple system state for testing
		SystemState state(sim_system);
		// Add some test particles
		HostParticleData test_particles;
		test_particles.resize(100);
		for (size_t i = 0; i < test_particles.size(); ++i) {
			test_particles.global_id[i] = static_cast<int>(i);
			test_particles.type_id[i] = 0;
			test_particles.pos[i] = Vector3(static_cast<float>(i % 10) * 10.0f,
											static_cast<float>((i / 10) % 10) * 10.0f,
											static_cast<float>(i / 100) * 10.0f);
			test_particles.mom[i] = Vector3(0.0f, 0.0f, 0.0f);
		}
		state.set_from_host_particle_data(test_particles);

		// Test decomposition - should not throw and should create single patch
		REQUIRE_NOTHROW(sim_system.decompose_system(state));

		// Verify patch manager was created
		REQUIRE(sim_system.has_patch_manager());

		auto* patch_manager = sim_system.get_patch_manager();
		REQUIRE(patch_manager != nullptr);

		// Should have exactly one patch
		REQUIRE(patch_manager->get_num_patches() == 1);
	}

	SECTION("Single resource uses single patch covering entire system") {
		SimSystem sim_system(single_resource_list);

		// Set up system
		Vector3 box_size(50.0f, 60.0f, 70.0f);
		sim_system.set_box_size(box_size.x, box_size.y, box_size.z);
		sim_system.set_cutoff(5.0f);
		sim_system.set_estimated_particles(500);

		SystemState state(sim_system);
		HostParticleData test_particles;
		test_particles.resize(10); // Small number for testing
		for (size_t i = 0; i < test_particles.size(); ++i) {
			test_particles.global_id[i] = static_cast<int>(i);
			test_particles.type_id[i] = 0;
			test_particles.pos[i] = Vector3(static_cast<float>(i) * 5.0f,
											static_cast<float>(i) * 6.0f,
											static_cast<float>(i) * 7.0f);
		}
		state.set_from_host_particle_data(test_particles);

		// Decompose
		sim_system.decompose_system(state);

		auto* patch_manager = sim_system.get_patch_manager();
		REQUIRE(patch_manager != nullptr);

		// Single patch should cover entire system
		Vector3 global_min = patch_manager->get_global_min_bounds();
		Vector3 global_max = patch_manager->get_global_max_bounds();

		REQUIRE(global_min.x == Catch::Approx(0.0f));
		REQUIRE(global_min.y == Catch::Approx(0.0f));
		REQUIRE(global_min.z == Catch::Approx(0.0f));
		REQUIRE(global_max.x == Catch::Approx(box_size.x));
		REQUIRE(global_max.y == Catch::Approx(box_size.y));
		REQUIRE(global_max.z == Catch::Approx(box_size.z));
	}

	SECTION("Single resource avoids unnecessary decomposer setup") {
		SimSystem sim_system(single_resource_list);

		sim_system.set_box_size(100.0f, 100.0f, 100.0f);
		sim_system.set_cutoff(5.0f);
		sim_system.set_estimated_particles(1000);

		// Verify that no decomposer is required for single resource
		REQUIRE(sim_system.get_decomposer() == nullptr);

		SystemState state(sim_system);
		HostParticleData test_particles;
		test_particles.resize(50);
		for (size_t i = 0; i < test_particles.size(); ++i) {
			test_particles.global_id[i] = static_cast<int>(i);
			test_particles.type_id[i] = 0;
			test_particles.pos[i] = Vector3(static_cast<float>(i % 5) * 20.0f,
											static_cast<float>((i / 5) % 5) * 20.0f,
											static_cast<float>(i / 25) * 20.0f);
		}
		state.set_from_host_particle_data(test_particles);

		// Should work without decomposer
		REQUIRE_NOTHROW(sim_system.decompose_system(state));
		REQUIRE(sim_system.has_patch_manager());
	}

	SECTION("Compare single vs multi-resource for same system") {
		if (device_resources.size() < 2) {
			WARN("Need at least 2 resources for comparison test, skipping");
			return;
		}

		// Single resource system
		SimSystem single_system(single_resource_list);
		single_system.set_box_size(100.0f, 100.0f, 100.0f);
		single_system.set_cutoff(5.0f);
		single_system.set_estimated_particles(1000);

		// Multi-resource system
		std::vector<Resource> multi_resources = {device_resources[0], device_resources[1]};
		SimSystem multi_system(multi_resources);
		multi_system.set_box_size(100.0f, 100.0f, 100.0f);
		multi_system.set_cutoff(5.0f);
		multi_system.set_estimated_particles(1000);
		multi_system.set_decomposer_type(DecomposerType::Spatial);

		// Create identical test data
		SystemState state1(single_system), state2(multi_system);
		HostParticleData test_particles;
		test_particles.resize(100);
		for (size_t i = 0; i < test_particles.size(); ++i) {
			test_particles.global_id[i] = static_cast<int>(i);
			test_particles.type_id[i] = 0;
			test_particles.pos[i] = Vector3(static_cast<float>(i % 10) * 10.0f,
											static_cast<float>((i / 10) % 10) * 10.0f,
											static_cast<float>(i / 100) * 10.0f);
		}
		state1.set_from_host_particle_data(test_particles);
		state2.set_from_host_particle_data(test_particles);

		// Decompose both
		REQUIRE_NOTHROW(single_system.decompose_system(state1));
		REQUIRE_NOTHROW(multi_system.decompose_system(state2));

		// Single resource should have 1 patch, multi should have more
		REQUIRE(single_system.get_patch_manager()->get_num_patches() == 1);
		REQUIRE(multi_system.get_patch_manager()->get_num_patches() > 1);

		// Both should have valid patch managers
		REQUIRE(single_system.has_patch_manager());
		REQUIRE(multi_system.has_patch_manager());
	}
}

/**
 * @brief Test that single resource optimization maintains functionality
 */
TEST_CASE("Single Resource Functional Correctness", "[SimSystem][SingleResource][Functional]") {

	SECTION("Single resource system can be rebalanced") {
		SimSystem sim_system(single_resource_list);
		sim_system.set_box_size(100.0f, 100.0f, 100.0f);
		sim_system.set_cutoff(5.0f);
		sim_system.set_estimated_particles(1000);

		SystemState state(sim_system);
		HostParticleData test_particles;
		test_particles.resize(50);
		for (size_t i = 0; i < test_particles.size(); ++i) {
			test_particles.global_id[i] = static_cast<int>(i);
			test_particles.type_id[i] = 0;
			test_particles.pos[i] =
				Vector3(static_cast<float>(i) * 2.0f, static_cast<float>(i) * 2.0f, 0.0f);
		}
		state.set_from_host_particle_data(test_particles);

		// Initial decomposition
		sim_system.decompose_system(state);
		REQUIRE(sim_system.has_patch_manager());

		// Rebalancing should work (essentially re-creating single patch)
		REQUIRE_NOTHROW(sim_system.rebalance_system(state));
		REQUIRE(sim_system.has_patch_manager());
		REQUIRE(sim_system.get_patch_manager()->get_num_patches() == 1);
	}

	SECTION("Single resource handles empty particle system") {
		SimSystem sim_system(single_resource_list);
		sim_system.set_box_size(100.0f, 100.0f, 100.0f);
		sim_system.set_cutoff(5.0f);
		sim_system.set_estimated_particles(1000);

		SystemState empty_state(sim_system);
		HostParticleData empty_particles;
		empty_particles.resize(0); // No particles
		empty_state.set_from_host_particle_data(empty_particles);

		// Should handle empty system gracefully
		REQUIRE_NOTHROW(sim_system.decompose_system(empty_state));
		REQUIRE(sim_system.has_patch_manager());
		REQUIRE(sim_system.get_patch_manager()->get_num_patches() == 1);
	}
}
