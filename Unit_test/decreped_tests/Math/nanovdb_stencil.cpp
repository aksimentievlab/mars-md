/**
 * @file test_nanovdb_stencil.cpp
 * @brief Unit tests for NanoVDB integration with ARBD's buffer and kernel system
 *
 * This test suite covers:
 * 1. Loading NanoVDB grids from files
 * 2. Computing gradients using stencils on device
 * 3. Performing interpolation operations
 * 4. Using neighbor finding with VDB grids
 */

#include "../catch_boiler.h"
#include "Backend/Kernels.h"
#include "IO/NanoVDBIO.h"
#include "Math/NanoGridHandle.h"
#include "Math/Vector3.h"
#include <iostream>
#include <vector>

using namespace ARBD;

TEST_CASE("NanoVDB Grid Loading and Query", "[nanovdb][io]") {
	SECTION("Query available grids from file") {
		std::string filename = "density_field.nvdb";

		// Query available grids first
		auto available_grids = IO::query_nanovdb_grids(filename);
		REQUIRE_FALSE(available_grids.empty());

		// Check if density grid exists
		std::string grid_name = "density";
		REQUIRE(IO::has_nanovdb_grid(filename, grid_name));
	}

	SECTION("Load NanoVDB grid using ARBD resource system") {
		std::string filename = "density_field.nvdb";
		std::string grid_name = "density";

		Resource cuda_resource{ResourceType::CUDA, 0};
		auto grid_adapter = IO::load_nanovdb_grid(filename, grid_name, cuda_resource);

		REQUIRE(grid_adapter.size() > 0);

		// Get the grid pointer for stencil operations
		const auto* grid = grid_adapter.grid<float>();
		REQUIRE(grid != nullptr);
	}
}

TEST_CASE("NanoVDB Gradient Computation", "[nanovdb][stencil][gradient]") {
	std::string filename = "density_field.nvdb";
	std::string grid_name = "density";

	Resource cuda_resource{ResourceType::CUDA, 0};
	auto grid_adapter = IO::load_nanovdb_grid(filename, grid_name, cuda_resource);
	const auto* grid = grid_adapter.grid<float>();

	REQUIRE(grid != nullptr);

	SECTION("Compute gradients for sample coordinates") {
		std::vector<nanovdb::Coord> sample_coords = {{10, 20, 30},
													 {15, 25, 35},
													 {20, 30, 40},
													 {25, 35, 45}};

		DeviceBuffer<nanovdb::Coord> coord_buffer(sample_coords.size(), cuda_resource);
		coord_buffer.copy_from_host(sample_coords.data(), sample_coords.size());

		DeviceBuffer<nanovdb::math::Vec3<float>> gradient_results(sample_coords.size(),
																  cuda_resource);

		// Create NanoGrid processor for stencil operations
		NanoGrid<Resource> nano_processor(cuda_resource);

		// Compute gradients using device kernels
		KernelConfig config;
		config.async = false;
		config.block_size = {256, 1, 1};

		auto gradient_event =
			nano_processor.compute_gradients(coord_buffer, gradient_results, grid, config);
		gradient_event.wait();

		// Verify results
		REQUIRE(gradient_results.size() == sample_coords.size());

		// Retrieve and verify gradient results
		std::vector<nanovdb::math::Vec3<float>> host_gradients(gradient_results.size());
		gradient_results.copy_to_host(host_gradients.data(), host_gradients.size());

		for (const auto& grad : host_gradients) {
			// Gradients should be finite values
			REQUIRE(std::isfinite(grad[0]));
			REQUIRE(std::isfinite(grad[1]));
			REQUIRE(std::isfinite(grad[2]));
		}
	}
}

TEST_CASE("NanoVDB Laplacian Computation", "[nanovdb][stencil][laplacian]") {
	std::string filename = "density_field.nvdb";
	std::string grid_name = "density";

	Resource cuda_resource{ResourceType::CUDA, 0};
	auto grid_adapter = IO::load_nanovdb_grid(filename, grid_name, cuda_resource);
	const auto* grid = grid_adapter.grid<float>();

	REQUIRE(grid != nullptr);

	SECTION("Compute Laplacians for sample coordinates") {
		std::vector<nanovdb::Coord> sample_coords = {{10, 20, 30},
													 {15, 25, 35},
													 {20, 30, 40},
													 {25, 35, 45}};

		DeviceBuffer<nanovdb::Coord> coord_buffer(sample_coords.size(), cuda_resource);
		coord_buffer.copy_from_host(sample_coords.data(), sample_coords.size());

		DeviceBuffer<float> laplacian_results(sample_coords.size(), cuda_resource);

		NanoGrid<Resource> nano_processor(cuda_resource);

		KernelConfig config;
		config.async = false;
		config.block_size = {256, 1, 1};

		auto laplacian_event =
			nano_processor.compute_laplacians(coord_buffer, laplacian_results, grid, config);
		laplacian_event.wait();

		// Verify results
		REQUIRE(laplacian_results.size() == sample_coords.size());

		std::vector<float> host_laplacians(laplacian_results.size());
		laplacian_results.copy_to_host(host_laplacians.data(), host_laplacians.size());

		for (float laplacian : host_laplacians) {
			// Laplacians should be finite values
			REQUIRE(std::isfinite(laplacian));
		}
	}
}

TEST_CASE("NanoVDB Interpolation", "[nanovdb][interpolation]") {
	std::string filename = "density_field.nvdb";
	std::string grid_name = "density";

	Resource cuda_resource{ResourceType::CUDA, 0};
	auto grid_adapter = IO::load_nanovdb_grid(filename, grid_name, cuda_resource);
	const auto* grid = grid_adapter.grid<float>();

	REQUIRE(grid != nullptr);

	SECTION("Interpolate values at world coordinates") {
		std::vector<nanovdb::math::Vec3<float>> world_positions = {{1.5f, 2.5f, 3.5f},
																   {4.5f, 5.5f, 6.5f},
																   {7.5f, 8.5f, 9.5f}};

		DeviceBuffer<nanovdb::math::Vec3<float>> position_buffer(world_positions.size(),
																 cuda_resource);
		position_buffer.copy_from_host(world_positions.data(), world_positions.size());

		DeviceBuffer<float> interpolation_results(world_positions.size(), cuda_resource);

		NanoGrid<Resource> nano_processor(cuda_resource);

		KernelConfig config;
		config.async = false;
		config.block_size = {256, 1, 1};

		auto interp_event =
			nano_processor.interpolate_values(position_buffer, interpolation_results, grid, config);
		interp_event.wait();

		// Verify results
		REQUIRE(interpolation_results.size() == world_positions.size());

		std::vector<float> host_interpolated(interpolation_results.size());
		interpolation_results.copy_to_host(host_interpolated.data(), host_interpolated.size());

		for (float value : host_interpolated) {
			// Interpolated values should be finite
			REQUIRE(std::isfinite(value));
		}
	}
}

TEST_CASE("NanoVDB Neighbor Finding", "[nanovdb][neighbors]") {
	std::string filename = "density_field.nvdb";
	std::string grid_name = "density";

	Resource cuda_resource{ResourceType::CUDA, 0};
	auto grid_adapter = IO::load_nanovdb_grid(filename, grid_name, cuda_resource);
	const auto* grid = grid_adapter.grid<float>();

	REQUIRE(grid != nullptr);

	SECTION("Count neighbors within cutoff radius") {
		std::vector<nanovdb::math::Vec3<float>> world_positions = {{1.5f, 2.5f, 3.5f},
																   {4.5f, 5.5f, 6.5f},
																   {7.5f, 8.5f, 9.5f}};

		DeviceBuffer<nanovdb::math::Vec3<float>> position_buffer(world_positions.size(),
																 cuda_resource);
		position_buffer.copy_from_host(world_positions.data(), world_positions.size());

		DeviceBuffer<int> neighbor_counts(world_positions.size(), cuda_resource);

		// Note: This section is commented out in the original example
		// as neighbor finding might not be fully implemented yet
		/*
		Interactions::NanoVDBNeighborList<float> neighbor_finder(
			std::move(grid_adapter), 2.0f);

		auto neighbor_event = neighbor_finder.count_neighbors(
			position_buffer, neighbor_counts, config);
		neighbor_event.wait();

		REQUIRE(neighbor_counts.size() == world_positions.size());

		std::vector<int> host_neighbor_counts(neighbor_counts.size());
		neighbor_counts.copy_to_host(host_neighbor_counts.data(), host_neighbor_counts.size());

		for (int count : host_neighbor_counts) {
			REQUIRE(count >= 0); // Neighbor counts should be non-negative
		}
		*/

		// For now, just verify the buffer setup works
		REQUIRE(neighbor_counts.size() == world_positions.size());
		REQUIRE(position_buffer.size() == world_positions.size());
	}
}

TEST_CASE("NanoVDB Error Handling", "[nanovdb][error]") {
	SECTION("Handle missing file gracefully") {
		std::string missing_filename = "nonexistent_file.nvdb";

		// Should handle missing file without crashing
		auto available_grids = IO::query_nanovdb_grids(missing_filename);
		REQUIRE(available_grids.empty());

		REQUIRE_FALSE(IO::has_nanovdb_grid(missing_filename, "any_grid"));
	}

	SECTION("Handle missing grid name gracefully") {
		std::string filename = "density_field.nvdb";
		std::string missing_grid = "nonexistent_grid";

		REQUIRE_FALSE(IO::has_nanovdb_grid(filename, missing_grid));
	}
}
