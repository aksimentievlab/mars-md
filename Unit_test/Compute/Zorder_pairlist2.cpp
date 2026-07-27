/**
 * @file test_zorder_pairlist.cpp
 * @brief Test ZOrder pairlist building and inspection
 *
 * Focus:
 * - Build ZOrder pairlist (faster than CellList)
 * - Print pairs for verification
 * - Compare with expected neighbors
 * - Performance timing
 */

#include "../catch_boiler.h"

#include "Backend/Kernels.h"
#include "Backend/Resource.h"
#include "Objects/DeviceParticle.h"
#include "Objects/DeviceParticleManager.h"
#include "Objects/ParticleProperties.h"
#include "PatchOperation/PairListKernels/ZOrderPairlist.h"
#include "PatchOperation/Pairlist.h"
#include "Types/Types.h"
#include <chrono>
#include <numeric>

using namespace ARBD;

namespace {

/**
 * @brief Create particle grid for testing
 */
HostParticleData create_particle_grid(int nx, int ny, int nz, float spacing) {
	HostParticleData data;
	int total = nx * ny * nz;
	data.resize(total);

	int idx = 0;
	for (int x = 0; x < nx; x++) {
		for (int y = 0; y < ny; y++) {
			for (int z = 0; z < nz; z++) {
				data.global_id[idx] = idx;
				data.type_id[idx] = 0;
				data.pos[idx] = Vector3(x * spacing, y * spacing, z * spacing);
				data.mom[idx] = Vector3(0, 0, 0);
				data.force[idx] = Vector3(0, 0, 0);
				data.orient[idx] = Vector3(0, 0, 0);
				data.flags[idx] = 0;
				idx++;
			}
		}
	}

	return data;
}

/**
 * @brief Print pairlist to console with optional Morton code debug
 */
void print_pairlist(const Pairlist& pairlist,
					const HostParticleData& host_data,
					size_t max_pairs_to_print = 20,
					bool debug_morton = true) {
	size_t num_pairs = pairlist.get_num_pairs();
	size_t num_particles = host_data.size();

	// Copy pairs to host
	std::vector<ARBD::int2> pairs(num_pairs);
	pairlist.get_neighbor_pairs().copy_to_host_sync(pairs.data(), num_pairs);

	LOGINFO("=== ZOrder Pairlist ===");
	LOGINFO("Total pairs: {}", num_pairs);
	LOGINFO("Cutoff: {:.3f}", pairlist.get_cutoff());

	// Optional: Debug Morton codes
	if (debug_morton) {
		auto* zorder_pl = dynamic_cast<const ZOrderPairlist*>(&pairlist);
		if (zorder_pl) {
			const auto& sorter = zorder_pl->get_sorter();
			std::vector<morton_t> codes(num_particles);
			sorter.get_morton_codes().copy_to_host_sync(codes.data(), num_particles);

			LOGINFO("=== Morton Codes ===");
			for (size_t i = 0; i < num_particles; i++) {
				LOGINFO("  Particle {}: 0x{:08x}", i, codes[i]);
			}

			std::set<morton_t> unique(codes.begin(), codes.end());
			LOGINFO("Unique codes: {} / {}", unique.size(), num_particles);

			if (unique.size() < num_particles) {
				std::cout << "WARN:Duplicate Morton codes detected!" << std::endl;
				LOGWARN("⚠️  DUPLICATE MORTON CODES DETECTED!");
			}
		}
	}

	size_t print_count = std::min(num_pairs, max_pairs_to_print);
	LOGINFO("First {} pairs:", print_count);

	for (size_t i = 0; i < print_count; i++) {
		int pi = pairs[i].x;
		int pj = pairs[i].y;

		Vector3 pos_i = host_data.pos[pi];
		Vector3 pos_j = host_data.pos[pj];
		Vector3 diff = pos_j - pos_i;
		float dist = diff.length();

		LOGINFO("  Pair {}: ({}, {}) | pos_i=({:.2f},{:.2f},{:.2f}) pos_j=({:.2f},{:.2f},{:.2f}) | "
				"dist={:.3f}",
				i,
				pi,
				pj,
				pos_i.x,
				pos_i.y,
				pos_i.z,
				pos_j.x,
				pos_j.y,
				pos_j.z,
				dist);
	}

	if (num_pairs > max_pairs_to_print) {
		LOGINFO("  ... ({} more pairs)", num_pairs - max_pairs_to_print);
	}
}

/**
 * @brief Count neighbors per particle
 */
void print_neighbor_histogram(const Pairlist& pairlist, size_t num_particles) {
	size_t num_pairs = pairlist.get_num_pairs();
	std::vector<ARBD::int2> pairs(num_pairs);
	pairlist.get_neighbor_pairs().copy_to_host_sync(pairs.data(), num_pairs);

	// Count neighbors per particle
	std::vector<int> neighbor_count(num_particles, 0);
	for (const auto& pair : pairs) {
		neighbor_count[pair.x]++;
		neighbor_count[pair.y]++;
	}

	// Find min/max/avg
	int min_neighbors = *std::min_element(neighbor_count.begin(), neighbor_count.end());
	int max_neighbors = *std::max_element(neighbor_count.begin(), neighbor_count.end());
	float avg_neighbors =
		std::accumulate(neighbor_count.begin(), neighbor_count.end(), 0.0f) / num_particles;

	LOGINFO("=== Neighbor Statistics ===");
	LOGINFO("Min neighbors per particle: {}", min_neighbors);
	LOGINFO("Max neighbors per particle: {}", max_neighbors);
	LOGINFO("Avg neighbors per particle: {:.2f}", avg_neighbors);

	// Histogram
	std::map<int, int> histogram;
	for (int count : neighbor_count) {
		histogram[count]++;
	}

	LOGINFO("Histogram:");
	for (const auto& [count, freq] : histogram) {
		LOGINFO("  {} neighbors: {} particles", count, freq);
	}
}

} // anonymous namespace
// Add this test to your test file to debug what's happening

TEST_CASE("ZOrder Pairlist - DEBUG Bounding Box", "[pairlist][debug]") {
	initialize_backend_once();
	Resource res(Global::single_resource_id);

	// Create simple 2x2x2 grid (8 particles, spacing=2.0)
	HostParticleData host_data = create_particle_grid(2, 2, 2, 2.0f);

	std::cout << "\n=== INPUT PARTICLES ===" << std::endl;
	for (int i = 0; i < 8; i++) {
		auto& p = host_data.pos[i];
		std::cout << "Particle " << i << ": (" << p.x << ", " << p.y << ", " << p.z << ")"
				  << std::endl;
	}

	DeviceParticle particles(8, res);
	particles.copy_from_host(host_data, 8);

	// Create ZOrder pairlist
	auto pairlist_ptr = create_pairlist(PairlistBuilderType::ZOrder, res, 100, 1000);
	auto* zorder_pairlist = static_cast<ZOrderPairlist*>(pairlist_ptr.get());

	// Build pairlist
	float cutoff = 2.5f;
	pairlist_ptr->build_pairlist(particles.pos(), 8, cutoff);

	// Get access to sorter
	const auto& sorter = zorder_pairlist->get_sorter();

	// Copy Morton codes to host
	auto morton_codes = sorter.get_morton_codes();
	std::vector<morton_t> host_codes(8);
	morton_codes.copy_to_host_sync(host_codes.data(), 8);

	// Copy sorted indices
	auto sorted_indices = sorter.get_sorted_indices();
	std::vector<uint32_t> host_indices(8);
	sorted_indices.copy_to_host_sync(host_indices.data(), 8);

	// Copy sorted positions
	auto sorted_pos_buf = zorder_pairlist->get_sorted_positions();
	std::vector<Vector3> sorted_positions(8);
	sorted_pos_buf.copy_to_host_sync(sorted_positions.data(), 8);

	std::cout << "\n=== MORTON CODES (after sorting) ===" << std::endl;
	for (int i = 0; i < 8; i++) {
		std::cout << "sorted[" << i << "]: "
				  << "code=0x" << std::hex << std::setw(8) << std::setfill('0') << host_codes[i]
				  << std::dec << " orig_id=" << host_indices[i] << " pos=(" << sorted_positions[i].x
				  << "," << sorted_positions[i].y << "," << sorted_positions[i].z << ")"
				  << std::endl;
	}

	// Check for duplicate codes
	std::set<morton_t> unique_codes(host_codes.begin(), host_codes.end());
	std::cout << "\n=== ANALYSIS ===" << std::endl;
	std::cout << "Unique Morton codes: " << unique_codes.size() << " / 8" << std::endl;

	if (unique_codes.size() < 8) {
		std::cout << "WARNING: Duplicate Morton codes detected!" << std::endl;
		std::cout << "This means bounding box computation is still broken!" << std::endl;

		// Print histogram
		std::map<morton_t, int> code_count;
		for (auto code : host_codes) {
			code_count[code]++;
		}

		std::cout << "\nCode histogram:" << std::endl;
		for (auto& [code, count] : code_count) {
			std::cout << "  0x" << std::hex << code << std::dec << ": " << count << " particles"
					  << std::endl;
		}
	} else {
		std::cout << "✓ All Morton codes are unique (bounding box is working!)" << std::endl;
	}

	// Check search window
	std::cout << "\n=== SEARCH WINDOW CHECK ===" << std::endl;
	std::cout << "Cutoff: " << cutoff << std::endl;
	std::cout << "Cutoff²: " << (cutoff * cutoff) << std::endl;
	std::cout << "Pairs found: " << pairlist_ptr->get_num_pairs() << std::endl;

	// Manually check distances in sorted order
	int manual_pairs = 0;
	for (int i = 0; i < 8; i++) {
		for (int j = i + 1; j < 8; j++) {
			Vector3 dr = sorted_positions[j] - sorted_positions[i];
			float dist = dr.length();
			if (dist <= cutoff) {
				std::cout << "  sorted[" << i << "]→sorted[" << j << "]: "
						  << "orig(" << host_indices[i] << "," << host_indices[j] << ") "
						  << "dist=" << dist << std::endl;
				manual_pairs++;
			}
		}
	}
	std::cout << "Manual count (all pairs): " << manual_pairs << std::endl;

	// Check with search window
	const int search_range = 64;
	int window_pairs = 0;
	for (int i = 0; i < 8; i++) {
		uint32_t orig_i = host_indices[i];
		int start = std::max(0, i - search_range);
		int end = std::min(8, i + search_range);

		for (int j = start; j < end; j++) {
			if (i == j)
				continue;

			uint32_t orig_j = host_indices[j];
			if (orig_i >= orig_j)
				continue; // Skip duplicates

			Vector3 dr = sorted_positions[j] - sorted_positions[i];
			float dist = dr.length();
			if (dist <= cutoff) {
				window_pairs++;
			}
		}
	}
	std::cout << "Window search count (should match kernel): " << window_pairs << std::endl;
}
TEST_CASE("ZOrder Pairlist - Build 2x2x2 Grid", "[pairlist][zorder][print]") {
	initialize_backend_once();
	Resource res(Global::single_resource_id);

	// Create simple 2x2x2 grid (8 particles, spacing=2.0)
	HostParticleData host_data = create_particle_grid(2, 2, 2, 2.0f);

	DeviceParticle particles(host_data.size(), res);
	particles.copy_from_host(host_data, 8);

	// Create ZOrder pairlist
	auto pairlist = create_pairlist(PairlistBuilderType::ZOrder, res, 100, 1000);

	// Build with cutoff = 2.5 (should find nearest neighbors at distance 2.0)
	auto start = std::chrono::high_resolution_clock::now();
	pairlist->build_pairlist(particles.pos(), 8, 2.5f);
	auto end = std::chrono::high_resolution_clock::now();

	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	LOGINFO("ZOrder pairlist build time: {} μs", duration.count());

	// Print pairlist
	print_pairlist(*pairlist, host_data, 50); // Print all pairs

	// Print statistics
	print_neighbor_histogram(*pairlist, 8);

	// Verify
	REQUIRE(pairlist->get_num_pairs() > 0);
	REQUIRE(pairlist->get_num_pairs() <= 8 * 7 / 2); // Max possible pairs
}

TEST_CASE("ZOrder Pairlist - Build 3x3x3 Grid", "[pairlist][zorder][print]") {
	initialize_backend_once();
	Resource res(Global::single_resource_id);

	// Create 3x3x3 grid (27 particles, spacing=2.0)
	HostParticleData host_data = create_particle_grid(3, 3, 3, 2.0f);

	DeviceParticle particles(27, res);
	particles.copy_from_host(host_data, 27);

	auto pairlist = create_pairlist(PairlistBuilderType::ZOrder, res, 100, 1000);

	auto start = std::chrono::high_resolution_clock::now();
	pairlist->build_pairlist(particles.pos(), 27, 2.5f);
	auto end = std::chrono::high_resolution_clock::now();

	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	LOGINFO("ZOrder pairlist build time: {} μs", duration.count());

	// Print first 30 pairs
	print_pairlist(*pairlist, host_data, 30);
	print_neighbor_histogram(*pairlist, 27);

	auto stats = pairlist->get_statistics();
	LOGINFO("=== Statistics ===");
	LOGINFO("Particles: {}", stats.num_particles);
	LOGINFO("Pairs: {}", stats.num_pairs);
	LOGINFO("Avg neighbors: {:.2f}", stats.average_neighbors_per_particle);
}

TEST_CASE("ZOrder Pairlist - Different Cutoffs", "[pairlist][zorder][cutoff]") {
	initialize_backend_once();
	Resource res(Global::single_resource_id);

	// 3 particles in a line: 0---1---2 (spacing = 1.0)
	HostParticleData host_data(std::vector<int>{0, 1, 2});
	host_data.type_id = {0, 0, 0};
	host_data.pos =
		std::vector<Vector3>{Vector3(0, 0, 0), Vector3(1.0f, 0, 0), Vector3(2.0f, 0, 0)};
	host_data.mom = std::vector<Vector3>{Vector3(0, 0, 0), Vector3(0, 0, 0), Vector3(0, 0, 0)};
	host_data.force = std::vector<Vector3>{Vector3(0, 0, 0), Vector3(0, 0, 0), Vector3(0, 0, 0)};
	host_data.orient = std::vector<Vector3>{Vector3(0, 0, 0), Vector3(0, 0, 0), Vector3(0, 0, 0)};
	host_data.flags = std::vector<uint32_t>{0, 0, 0};

	DeviceParticle particles(3, res);
	particles.copy_from_host(host_data, 3);

	auto pairlist = create_pairlist(PairlistBuilderType::ZOrder, res, 10, 10);
	// RIGHT AFTER: pairlist->build_pairlist(particles.pos(), 8, 2.5f);
	int num_particles = 3;

	SECTION("Cutoff 1.1 - Only nearest neighbors") {
		pairlist->build_pairlist(particles.pos(), 3, 1.1f);
		LOGINFO("\n=== Cutoff 1.1 ===");
		print_pairlist(*pairlist, host_data, 10);

		// Should find (0,1) and (1,2) but NOT (0,2)
		REQUIRE(pairlist->get_num_pairs() == 2);
	}

	SECTION("Cutoff 2.5 - All pairs") {
		pairlist->build_pairlist(particles.pos(), 3, 2.5f);
		LOGINFO("\n=== Cutoff 2.5 ===");
		print_pairlist(*pairlist, host_data, 10);

		// Should find all 3 pairs: (0,1), (1,2), (0,2)
		REQUIRE(pairlist->get_num_pairs() == 3);
	}
}

TEST_CASE("ZOrder Pairlist - Performance Scaling", "[pairlist][zorder][performance]") {
	initialize_backend_once();
	Resource res(Global::single_resource_id);

	struct TestCase {
		int size;
		float spacing;
		float cutoff;
	};

	std::vector<TestCase> test_cases = {
		{2, 2.0f, 2.5f}, // 8 particles
		{3, 2.0f, 2.5f}, // 27 particles
		{4, 2.0f, 2.5f}, // 64 particles
		{5, 2.0f, 2.5f}, // 125 particles
	};

	LOGINFO("\n=== Performance Scaling ===");

	for (const auto& tc : test_cases) {
		int num_particles = tc.size * tc.size * tc.size;
		HostParticleData host_data = create_particle_grid(tc.size, tc.size, tc.size, tc.spacing);

		DeviceParticle particles(num_particles, res);
		particles.copy_from_host(host_data, num_particles);

		auto pairlist = create_pairlist(PairlistBuilderType::ZOrder, res, 1000, 10000);

		// Warmup
		pairlist->build_pairlist(particles.pos(), num_particles, tc.cutoff);

		// Timed run
		auto start = std::chrono::high_resolution_clock::now();
		pairlist->build_pairlist(particles.pos(), num_particles, tc.cutoff);
		auto end = std::chrono::high_resolution_clock::now();

		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
		size_t num_pairs = pairlist->get_num_pairs();

		LOGINFO("{}³ = {} particles | {} pairs | {} μs | {:.2f} pairs/μs",
				tc.size,
				num_particles,
				num_pairs,
				duration.count(),
				static_cast<float>(num_pairs) / duration.count());
	}
}

TEST_CASE("ZOrder Pairlist - Verify Pair Distances", "[pairlist][zorder][verify]") {
	initialize_backend_once();
	Resource res(Global::single_resource_id);

	// Create 3x3x3 grid
	HostParticleData host_data = create_particle_grid(3, 3, 3, 2.0f);

	DeviceParticle particles(27, res);
	particles.copy_from_host(host_data, 27);

	auto pairlist = create_pairlist(PairlistBuilderType::ZOrder, res, 100, 1000);
	float cutoff = 2.5f;
	pairlist->build_pairlist(particles.pos(), 27, cutoff);

	// Copy pairs and verify all are within cutoff
	size_t num_pairs = pairlist->get_num_pairs();
	std::vector<ARBD::int2> pairs(num_pairs);
	pairlist->get_neighbor_pairs().copy_to_host_sync(pairs.data(), num_pairs);

	size_t violations = 0;
	for (const auto& pair : pairs) {
		Vector3 pos_i = host_data.pos[pair.x];
		Vector3 pos_j = host_data.pos[pair.y];
		float dist = (pos_j - pos_i).length();

		if (dist > cutoff) {
			LOGWARN("Pair ({}, {}) exceeds cutoff: {:.3f} > {:.3f}", pair.x, pair.y, dist, cutoff);
			violations++;
		}
	}

	REQUIRE(violations == 0);
	LOGINFO("All {} pairs are within cutoff {:.3f}", num_pairs, cutoff);
}

TEST_CASE("ZOrder Pairlist - Compare Build vs Update", "[pairlist][zorder][update]") {
	initialize_backend_once();
	Resource res(Global::single_resource_id);

	HostParticleData host_data = create_particle_grid(3, 3, 3, 2.0f);
	DeviceParticle particles(27, res);
	particles.copy_from_host(host_data, 27);

	auto pairlist = create_pairlist(PairlistBuilderType::ZOrder, res, 100, 1000);

	// Initial build
	auto start_build = std::chrono::high_resolution_clock::now();
	pairlist->build_pairlist(particles.pos(), 27, 2.5f);
	auto end_build = std::chrono::high_resolution_clock::now();
	size_t pairs_build = pairlist->get_num_pairs();

	// Update (particles haven't moved)
	auto start_update = std::chrono::high_resolution_clock::now();
	pairlist->update_pairlist(particles.pos(), 27);
	auto end_update = std::chrono::high_resolution_clock::now();
	size_t pairs_update = pairlist->get_num_pairs();

	auto build_time =
		std::chrono::duration_cast<std::chrono::microseconds>(end_build - start_build);
	auto update_time =
		std::chrono::duration_cast<std::chrono::microseconds>(end_update - start_update);

	LOGINFO("Build:  {} pairs in {} μs", pairs_build, build_time.count());
	LOGINFO("Update: {} pairs in {} μs", pairs_update, update_time.count());
	LOGINFO("Speedup: {:.2f}x", static_cast<float>(build_time.count()) / update_time.count());

	REQUIRE(pairs_build == pairs_update);
}
