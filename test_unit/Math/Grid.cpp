/*********************************************************************
 * @file  test_BaseGrid.cpp
 *
 * @brief Catch2 test suite for BaseGrid V2 implementation
 *        Tests host functionality, device-safe functions, and I/O operations
 *
 * @author Test code for arbd2/cpp20 branch
 *********************************************************************/

#include "../catch_boiler.h"
#include "Backend/Buffer.h"
#include "Backend/Events.h"
#include "Backend/Resource.h"
#include "IO/FileHandle.h"
#include "Math/BaseGrid.h"
#include "Math/IndexList.h"

#include <cmath>
#include <filesystem>
#include <fstream>

using namespace ARBD;
using Catch::Approx;
using Catch::Matchers::WithinAbs;

// Include the test runner
DEF_RUN_TRIAL

// Test constants
constexpr float EPSILON = 1e-6f;
constexpr size_t TEST_NX = 10;
constexpr size_t TEST_NY = 8;
constexpr size_t TEST_NZ = 6;

/*===================*\
|  TEST FIXTURES      |
\*===================*/

// Helper function to create test grid with known pattern
BaseGrid<float> create_test_grid() {
	Matrix3 basis = Matrix3(1.0f, 1.0f, 1.0f);
	Vector3 origin(0.0f, 0.0f, 0.0f);
	BaseGrid<float> grid(basis, origin, 4, 3, 2);

	// Fill with pattern: value = ix + iy*10 + iz*100
	for (size_t ix = 0; ix < 4; ++ix) {
		for (size_t iy = 0; iy < 3; ++iy) {
			for (size_t iz = 0; iz < 2; ++iz) {
				float value = static_cast<float>(ix + iy * 10 + iz * 100);
				size_t idx = grid.index(ix, iy, iz);
				grid[idx] = value;
			}
		}
	}
	return grid;
}

/*===================*\
|  CONSTRUCTION TESTS |
\*===================*/

TEST_CASE("BaseGrid Construction", "[BaseGrid][construction]") {

	SECTION("Default constructor") {
		BaseGrid<float> grid;

		REQUIRE(grid.size() == 1);
		REQUIRE(grid.nx() == 1);
		REQUIRE(grid.ny() == 1);
		REQUIRE(grid.nz() == 1);
		REQUIRE(grid[0] == Approx(0.0f));
	}

	SECTION("Primary constructor") {
		Matrix3 basis = Matrix3(1.0f, 2.0f, 3.0f);
		Vector3 origin(-5.0f, -10.0f, -15.0f);

		BaseGrid<float> grid(basis, origin, TEST_NX, TEST_NY, TEST_NZ);

		REQUIRE(grid.nx() == TEST_NX);
		REQUIRE(grid.ny() == TEST_NY);
		REQUIRE(grid.nz() == TEST_NZ);
		REQUIRE(grid.size() == TEST_NX * TEST_NY * TEST_NZ);

		REQUIRE(grid.origin().x == Approx(-5.0f));
		REQUIRE(grid.origin().y == Approx(-10.0f));
		REQUIRE(grid.origin().z == Approx(-15.0f));

		REQUIRE(grid.basis().ex().x == Approx(1.0f));
		REQUIRE(grid.basis().ey().y == Approx(2.0f));
		REQUIRE(grid.basis().ez().z == Approx(3.0f));
	}

	SECTION("Orthogonal box constructor") {
		Vector3 box_size(10.0f, 20.0f, 30.0f);
		float dx = 1.0f;

		BaseGrid<float> grid(box_size, dx);

		REQUIRE(grid.nx() == 10);
		REQUIRE(grid.ny() == 20);
		REQUIRE(grid.nz() == 30);

		// Check that origin is at -box_size/2
		REQUIRE(grid.origin().x == Approx(-5.0f));
		REQUIRE(grid.origin().y == Approx(-10.0f));
		REQUIRE(grid.origin().z == Approx(-15.0f));
	}

	SECTION("Copy constructor") {
		auto grid1 = create_test_grid();
		grid1[5] = 42.0f;

		BaseGrid<float> grid2(grid1);

		REQUIRE(grid2.size() == grid1.size());
		REQUIRE(grid2.nx() == grid1.nx());
		REQUIRE(grid2.ny() == grid1.ny());
		REQUIRE(grid2.nz() == grid1.nz());
		REQUIRE(grid2[5] == Approx(42.0f));
	}

	SECTION("Move constructor") {
		auto grid1 = create_test_grid();
		grid1[7] = 123.0f;
		auto original_size = grid1.size();

		BaseGrid<float> grid2(std::move(grid1));

		REQUIRE(grid2.size() == original_size);
		REQUIRE(grid2[7] == Approx(123.0f));
	}

	SECTION("Invalid dimensions throw exception") {
		Matrix3 basis = Matrix3(1.0f, 1.0f, 1.0f);
		Vector3 origin(0.0f, 0.0f, 0.0f);

		REQUIRE_THROWS_AS(BaseGrid<float>(basis, origin, 0, 5, 5), Exception);
		REQUIRE_THROWS_AS(BaseGrid<float>(basis, origin, 5, 0, 5), Exception);
		REQUIRE_THROWS_AS(BaseGrid<float>(basis, origin, 5, 5, 0), Exception);
	}
}

/*===================*\
|  INDEXING TESTS     |
\*===================*/

TEST_CASE("BaseGrid Indexing", "[BaseGrid][indexing]") {
	auto grid = create_test_grid(); // 4x3x2 grid

	SECTION("Linear indexing") {
		// Test known index calculation: iz + iy*nz + ix*ny*nz
		size_t expected_idx = 1 + 0 * 2 + 2 * 3 * 2; // iz=1, iy=0, ix=2 => 13
		size_t actual_idx = grid.index(2, 0, 1);

		REQUIRE(actual_idx == expected_idx);
		REQUIRE(actual_idx == 13);
	}

	SECTION("Index to ijk conversion") {
		size_t linear_idx = 7; // Should be ix=1, iy=0, iz=1
		auto ijk = grid.index_to_ijk(linear_idx);

		REQUIRE(ijk.size() == 3);
		REQUIRE(ijk[0] == 1); // ix
		REQUIRE(ijk[1] == 0); // iy
		REQUIRE(ijk[2] == 1); // iz
	}

	SECTION("Round-trip indexing") {
		for (size_t ix = 0; ix < 4; ++ix) {
			for (size_t iy = 0; iy < 3; ++iy) {
				for (size_t iz = 0; iz < 2; ++iz) {
					size_t linear = grid.index(ix, iy, iz);
					auto ijk = grid.index_to_ijk(linear);

					REQUIRE(ijk[0] == ix);
					REQUIRE(ijk[1] == iy);
					REQUIRE(ijk[2] == iz);
				}
			}
		}
	}

	SECTION("Device-safe IndexList functionality") {
		IndexList<int, 5> list;

		REQUIRE(list.empty());
		REQUIRE(list.size() == 0);
		REQUIRE(list.capacity() == 5);

		list.add(10);
		list.add(20);
		list.add(30);

		REQUIRE(list.size() == 3);
		REQUIRE(list[0] == 10);
		REQUIRE(list[1] == 20);
		REQUIRE(list[2] == 30);

		REQUIRE(list.find(20) == 1);
		REQUIRE(list.find(99) == 5); // Not found returns capacity
		REQUIRE(list.contains(30));
		REQUIRE_FALSE(list.contains(99));
	}
}

/*===================*\
|  COORDINATE TESTS   |
\*===================*/

TEST_CASE("BaseGrid Coordinate Transforms", "[BaseGrid][coordinates]") {
	// Create grid with non-trivial basis
	Matrix3 basis(Vector3(2.0f, 0.0f, 0.0f), Vector3(0.0f, 1.5f, 0.0f), Vector3(0.0f, 0.0f, 0.5f));
	Vector3 origin(-1.0f, -2.0f, -3.0f);
	BaseGrid<float> grid(basis, origin, 5, 4, 6);

	SECTION("Transform to grid coordinates") {
		Vector3 world_pos(1.0f, 1.0f, -2.5f);
		Vector3 grid_pos = grid.transform_to_grid(world_pos);

		// Manual calculation: inv(basis) * (world_pos - origin)
		Vector3 expected(1.0f, 2.0f, 1.0f);

		REQUIRE(grid_pos.x == Approx(expected.x).margin(1e-5f));
		REQUIRE(grid_pos.y == Approx(expected.y).margin(1e-5f));
		REQUIRE(grid_pos.z == Approx(expected.z).margin(1e-5f));
	}

	SECTION("Transform to world coordinates") {
		Vector3 grid_pos(2.0f, 1.0f, 3.0f);
		Vector3 world_pos = grid.transform_to_world(grid_pos);

		// Manual calculation: basis * grid_pos + origin
		Vector3 expected(3.0f, -0.5f, -1.5f);

		REQUIRE(world_pos.x == Approx(expected.x).margin(1e-5f));
		REQUIRE(world_pos.y == Approx(expected.y).margin(1e-5f));
		REQUIRE(world_pos.z == Approx(expected.z).margin(1e-5f));
	}

	SECTION("Round-trip transformation") {
		Vector3 original_world(0.5f, 0.5f, -2.75f);
		Vector3 grid_pos = grid.transform_to_grid(original_world);
		Vector3 back_to_world = grid.transform_to_world(grid_pos);

		REQUIRE(back_to_world.x == Approx(original_world.x).margin(1e-5f));
		REQUIRE(back_to_world.y == Approx(original_world.y).margin(1e-5f));
		REQUIRE(back_to_world.z == Approx(original_world.z).margin(1e-5f));
	}

	SECTION("Bounds checking") {
		Vector3 inside_pos(0.0f, 0.0f, -2.5f);
		Vector3 outside_pos(10.0f, 10.0f, 10.0f);

		REQUIRE(grid.in_bounds(inside_pos));
		REQUIRE_FALSE(grid.in_bounds(outside_pos));
	}

	SECTION("Interpolation bounds checking") {
		Vector3 safe_pos(1.0f, 0.0f, -2.0f);
		Vector3 edge_pos(-0.9f, -1.9f, -2.9f); // Too close to edge

		REQUIRE(grid.in_interpolation_bounds(safe_pos));
		REQUIRE_FALSE(grid.in_interpolation_bounds(edge_pos));
	}
}

/*===================*\
|  DATA ACCESS TESTS  |
\*===================*/

TEST_CASE("BaseGrid Data Access", "[BaseGrid][data_access]") {
	auto grid = create_test_grid();

	SECTION("Basic data access") {
		grid[0] = 1.0f;
		grid[13] = 2.5f;

		REQUIRE(grid[0] == Approx(1.0f));
		REQUIRE(grid[13] == Approx(2.5f));
	}

	SECTION("Bounds checking with at()") {
		REQUIRE_NOTHROW(grid.at(5) = 3.0f);
		REQUIRE_THROWS_AS(grid.at(1000), Exception);

		REQUIRE(grid.at(5) == Approx(3.0f));
	}

	SECTION("Raw data pointer") {
		float* data_ptr = grid.data();
		data_ptr[5] = 7.5f;

		REQUIRE(grid[5] == Approx(7.5f));
		REQUIRE(data_ptr != nullptr);
	}

	SECTION("Span access") {
		auto span_view = grid.span();
		span_view[8] = 9.25f;

		REQUIRE(span_view.size() == 24); // 4*3*2
		REQUIRE(grid[8] == Approx(9.25f));
	}

	SECTION("Const access") {
		const auto& const_grid = grid;

		REQUIRE(const_grid[0] == Approx(0.0f));	 // ix=0, iy=0, iz=0 => 0+0*10+0*100
		REQUIRE(const_grid[7] == Approx(101.0f)); // ix=1, iy=0, iz=1 => 1+0*10+1*100

		const float* const_data = const_grid.data();
		REQUIRE(const_data != nullptr);

		auto const_span = const_grid.span();
		REQUIRE(const_span.size() == 24);
	}
}

/*===================*\
|  INTERPOLATION TESTS|
\*===================*/

TEST_CASE("BaseGrid Interpolation", "[BaseGrid][interpolation]") {
	// Create a simple 2x2x2 grid with known linear function
	Matrix3 basis = Matrix3(1.0f, 1.0f, 1.0f);
	Vector3 origin(0.0f, 0.0f, 0.0f);
	BaseGrid<float> grid(basis, origin, 2, 2, 2);

	// Set up linear function: f(x,y,z) = x + 2*y + 3*z
	for (size_t ix = 0; ix < 2; ++ix) {
		for (size_t iy = 0; iy < 2; ++iy) {
			for (size_t iz = 0; iz < 2; ++iz) {
				float value = static_cast<float>(ix) + 2.0f * static_cast<float>(iy) +
							  3.0f * static_cast<float>(iz);
				size_t idx = grid.index(ix, iy, iz);
				grid[idx] = value;
			}
		}
	}

	SECTION("Nearest neighbor sampling") {
		Vector3 pos(0.25f, 0.25f, 0.25f); // Should be closest to (0,0,0)
		float value = grid.get_value(pos);
		float expected = 0.0f; // Value at (0,0,0) = 0+2*0+3*0

		REQUIRE(value == Approx(expected));
	}

	SECTION("Trilinear interpolation at center") {
		Vector3 pos(0.5f, 0.5f, 0.5f); // Center of grid
		float value = grid.interpolate(pos);

		// For linear function, interpolation should give exact result
		float expected = 0.5f + 2.0f * 0.5f + 3.0f * 0.5f; // = 3.0

		REQUIRE(value == Approx(expected).margin(1e-4f));
	}

	SECTION("Interpolation at grid points") {
		Vector3 pos(1.0f, 1.0f, 1.0f); // At grid point (1,1,1)
		float value = grid.interpolate(pos);
		float expected = 1.0f + 2.0f * 1.0f + 3.0f * 1.0f; // = 6.0

		REQUIRE(value == Approx(expected).margin(1e-4f));
	}
}

/*===================*\
|  NEIGHBOR TESTS     |
\*===================*/

TEST_CASE("BaseGrid Neighbor Operations", "[BaseGrid][neighbors]") {
	auto grid = create_test_grid(); // 4x3x2 with pattern: ix + iy*10 + iz*100

	SECTION("Single neighbor access") {
		// Get neighbor of (2,2,2) in +x direction -> should be (3,2,2)
		float neighbor = grid.get_neighbor(2, 2, 2, 1, 0, 0);
		float expected = 3.0f + 2.0f * 10 + 2.0f * 100; // = 223

		REQUIRE(neighbor == Approx(expected));
	}

	SECTION("Neighbor list center access") {
		auto neighbors = grid.get_neighbor_list(1, 1, 1);
		float center_value = neighbors.center();
		float expected_center = 1.0f + 1.0f * 10 + 1.0f * 100; // = 111

		REQUIRE(center_value == Approx(expected_center));
	}

	SECTION("Neighbor list structure") {
		auto neighbors = grid.get_neighbor_list(1, 1, 0);

		// Check specific neighbors
		float left_neighbor = neighbors(-1, 0, 0); // (0,1,0) = 10
		float right_neighbor = neighbors(1, 0, 0); // (2,1,0) = 12
		float up_neighbor = neighbors(0, 1, 0);	   // (1,2,0) = 21
		float down_neighbor = neighbors(0, -1, 0); // (1,0,0) = 1

		REQUIRE(left_neighbor == Approx(10.0f));
		REQUIRE(right_neighbor == Approx(12.0f));
		REQUIRE(up_neighbor == Approx(21.0f));
		REQUIRE(down_neighbor == Approx(1.0f));
	}

	SECTION("NeighborList direct construction") {
		BaseGrid<float>::NeighborList<float> neighbors;

		// Test initialization
		REQUIRE(neighbors.center() == Approx(0.0f));

		// Test assignment
		neighbors(1, 1, 1) = 42.0f;
		REQUIRE(neighbors.center() == Approx(42.0f));

		neighbors(0, 1, 1) = 17.0f;
		REQUIRE(neighbors(0, 1, 1) == Approx(17.0f));
	}
}

/*===================*\
|  UTILITY TESTS      |
\*===================*/

TEST_CASE("BaseGrid Utility Operations", "[BaseGrid][utilities]") {
	auto grid = create_test_grid();

	SECTION("Zero operation") {
		grid[5] = 42.0f;
		grid.zero();

		REQUIRE(grid[5] == Approx(0.0f));

		// Check multiple elements
		for (size_t i = 0; i < grid.size(); ++i) {
			REQUIRE(grid[i] == Approx(0.0f));
		}
	}

	SECTION("Shift operation") {
		float original_value = grid[5];
		grid.shift(10.0f);

		REQUIRE(grid[5] == Approx(original_value + 10.0f));
	}

	SECTION("Scale operation") {
		float original_value = grid[7];
		grid.scale(2.5f);

		REQUIRE(grid[7] == Approx(original_value * 2.5f));
	}

	SECTION("Mean calculation") {
		// For our test pattern, calculate expected mean
		float sum = 0.0f;
		for (size_t ix = 0; ix < 4; ++ix) {
			for (size_t iy = 0; iy < 3; ++iy) {
				for (size_t iz = 0; iz < 2; ++iz) {
					sum += static_cast<float>(ix + iy * 10 + iz * 100);
				}
			}
		}
		float expected_mean = sum / 24.0f;

		float actual_mean = grid.mean();
		REQUIRE(actual_mean == Approx(expected_mean));
	}

	SECTION("Element-wise multiplication") {
		auto grid2 = create_test_grid();

		// Set grid2 to all 2.0
		for (size_t i = 0; i < grid2.size(); ++i) {
			grid2[i] = 2.0f;
		}

		float original_value = grid[10];
		grid.multiply(grid2);

		REQUIRE(grid[10] == Approx(original_value * 2.0f));
	}

	SECTION("Geometry calculations") {
		Matrix3 basis = Matrix3(2.0f, 3.0f, 4.0f);
		Vector3 origin(1.0f, 2.0f, 3.0f);
		BaseGrid<float> grid(basis, origin, 5, 4, 3);

		// Test cell volume
		float expected_cell_volume = 2.0f * 3.0f * 4.0f; // det(basis)
		REQUIRE(grid.get_cell_volume() == Approx(expected_cell_volume));

		// Test total volume
		float expected_total_volume = expected_cell_volume * 5 * 4 * 3;
		REQUIRE(grid.get_total_volume() == Approx(expected_total_volume));

		// Test center
		Vector3 expected_center = origin + basis.transform(Vector3(2.5f, 2.0f, 1.5f));
		Vector3 actual_center = grid.get_center();
		REQUIRE(actual_center.x == Approx(expected_center.x));
		REQUIRE(actual_center.y == Approx(expected_center.y));
		REQUIRE(actual_center.z == Approx(expected_center.z));
	}
}

/*===================*\
|  I/O TESTS          |
\*===================*/

TEST_CASE("BaseGrid I/O Operations", "[BaseGrid][io]") {
	auto grid = create_test_grid();
	const std::string test_file = "test_output/test_grid.dx";

	SECTION("Write and read DX format") {
		// Write grid
		REQUIRE_NOTHROW(grid.write(test_file, "Test grid data"));

		// Verify file exists
		REQUIRE(std::filesystem::exists(test_file));

		// Read grid back
		BaseGrid<float> loaded_grid;
		REQUIRE_NOTHROW(loaded_grid = BaseGrid<float>::read_from_file(test_file));

		// Verify dimensions match
		REQUIRE(loaded_grid.nx() == grid.nx());
		REQUIRE(loaded_grid.ny() == grid.ny());
		REQUIRE(loaded_grid.nz() == grid.nz());

		// Verify data matches
		for (size_t i = 0; i < grid.size(); ++i) {
			REQUIRE(loaded_grid[i] == Approx(grid[i]).margin(1e-6f));
		}

		// Verify geometry matches
		REQUIRE(loaded_grid.origin().x == Approx(grid.origin().x));
		REQUIRE(loaded_grid.origin().y == Approx(grid.origin().y));
		REQUIRE(loaded_grid.origin().z == Approx(grid.origin().z));
	}

	SECTION("Write without comments") {
		const std::string simple_file = "test_output/simple_grid.dx";

		REQUIRE_NOTHROW(grid.write(simple_file));
		REQUIRE(std::filesystem::exists(simple_file));
	}

	SECTION("Invalid file handling") {
		REQUIRE_THROWS_AS(BaseGrid<float>::read_from_file("nonexistent_file.dx"), Exception);
		REQUIRE_THROWS_AS(grid.write("/invalid/path/file.dx"), Exception);
	}
}

/*===================*\
|  DEVICE TESTS       |
\*===================*/

TEST_CASE("BaseGrid Device Integration", "[BaseGrid][device]") {
	auto grid = create_test_grid();

	SECTION("Device pointer management") {
		REQUIRE(grid.get_device_pointer() == nullptr);
		REQUIRE_FALSE(grid.is_device_dirty());

		// Simulate backend setting device pointer
		float* dummy_ptr = reinterpret_cast<float*>(0x12345678);
		grid.set_device_pointer(dummy_ptr);

		REQUIRE(grid.get_device_pointer() == dummy_ptr);
		REQUIRE_FALSE(grid.is_device_dirty());

		// Modify host data - should mark dirty
		grid[0] = 999.0f;
		REQUIRE(grid.is_device_dirty());

		// Mark clean
		grid.mark_device_clean();
		REQUIRE_FALSE(grid.is_device_dirty());
	}
}

/*===================*\
|  DEVICE FUNCTIONS   |
\*===================*/

TEST_CASE("Device-Safe Functions", "[BaseGrid][device_functions]") {
	// Create simple 3x3x3 grid
	Matrix3 basis = Matrix3(1.0f, 1.0f, 1.0f);
	Vector3 origin(0.0f, 0.0f, 0.0f);
	BaseGrid<float> grid(basis, origin, 3, 3, 3);

	// Fill with simple pattern
	for (size_t i = 0; i < grid.size(); ++i) {
		grid[i] = static_cast<float>(i);
	}

	SECTION("Device-safe interpolation function") {
		Vector3 pos(0.5f, 0.5f, 0.5f);
		float value = interpolate_grid_point(grid.data(),
											 pos,
											 grid.origin(),
											 grid.basis_inverse(),
											 grid.dimensions());

		// Should interpolate between surrounding values
		REQUIRE(value > 0.0f);
		REQUIRE(value < 27.0f); // Max value in 3x3x3 grid
	}

	SECTION("Device-safe nearest value function") {
		Vector3 pos(1.1f, 1.1f, 1.1f);
		float value = get_value_nearest(grid.data(),
										pos,
										grid.origin(),
										grid.basis_inverse(),
										grid.dimensions());

		// Should get value at (1,1,1) = index 13
		REQUIRE(value == Approx(13.0f));
	}

	SECTION("Device-safe neighbor functions") {
		auto neighbors = get_neighbor_list_from_grid(grid.data(), 1, 1, 1, grid.dimensions());

		REQUIRE(neighbors.center() == Approx(13.0f)); // Value at (1,1,1)

		float neighbor_value =
			get_neighbor_from_grid(grid.data(), 1, 1, 1, 1, 0, 0, grid.dimensions());
		REQUIRE(neighbor_value == Approx(22.0f)); // Value at (2,1,1)
	}

	SECTION("Wrap index function") {
		REQUIRE(wrap_index(-1, 5) == 4);
		REQUIRE(wrap_index(0, 5) == 0);
		REQUIRE(wrap_index(4, 5) == 4);
		REQUIRE(wrap_index(5, 5) == 0);
		REQUIRE(wrap_index(6, 5) == 1);
		REQUIRE(wrap_index(-7, 5) == 3);
	}
}

/*===================*\
|  TYPE TESTS         |
\*===================*/

TEST_CASE("BaseGrid Type Support", "[BaseGrid][types]") {

	SECTION("Float grid") {
		BaseGrid<float> grid_f;
		REQUIRE(grid_f.size() == 1);
		grid_f[0] = 3.14f;
		REQUIRE(grid_f[0] == Approx(3.14f));
	}


	SECTION("Integer grid (particle indices)") {
		BaseGrid<int> grid_i;
		grid_i[0] = 42;
		REQUIRE(grid_i[0] == 42);
	}
}

/*===================*\
|  PERFORMANCE TESTS  |
\*===================*/

TEST_CASE("BaseGrid Performance", "[BaseGrid][performance]") {

	SECTION("Large grid construction") {
		Matrix3 basis = Matrix3(1.0f, 1.0f, 1.0f);
		Vector3 origin(0.0f, 0.0f, 0.0f);

		auto start = std::chrono::high_resolution_clock::now();
		BaseGrid<float> large_grid(basis, origin, 100, 100, 100);
		auto end = std::chrono::high_resolution_clock::now();

		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

		REQUIRE(large_grid.size() == 1000000);
		// Construction should be reasonably fast (less than 100ms)
		REQUIRE(duration.count() < 100);
	}

	SECTION("Bulk data access") {
		auto grid = create_test_grid();

		auto start = std::chrono::high_resolution_clock::now();

		// Fill entire grid
		for (size_t i = 0; i < grid.size(); ++i) {
			grid[i] = static_cast<float>(i * 2);
		}

		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

		// Should be very fast for small grid
		REQUIRE(duration.count() < 1000);
		REQUIRE(grid[grid.size() - 1] == Approx(static_cast<float>((grid.size() - 1) * 2)));
	}

	SECTION("Interpolation performance") {
		Matrix3 basis = Matrix3(1.0f, 1.0f, 1.0f);
		Vector3 origin(0.0f, 0.0f, 0.0f);
		BaseGrid<float> grid(basis, origin, 50, 50, 50);

		// Fill with some data
		for (size_t i = 0; i < grid.size(); ++i) {
			grid[i] = std::sin(static_cast<float>(i) * 0.01f);
		}

		auto start = std::chrono::high_resolution_clock::now();

		float sum = 0.0f;
		for (int i = 0; i < 1000; ++i) {
			Vector3 pos(static_cast<float>(i % 49),
						static_cast<float>((i / 49) % 49),
						static_cast<float>((i / 2401) % 49));
			sum += grid.interpolate(pos);
		}

		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

		// 1000 interpolations should be fast
		REQUIRE(duration.count() < 10000); // Less than 10ms
		REQUIRE(sum != 0.0f);			   // Sanity check
	}
}

/*=======================*\
|  INTEGRATION TESTS      |
\*=======================*/

TEST_CASE("BaseGrid Integration", "[BaseGrid][integration]") {

	SECTION("Multi-grid operations") {
		Matrix3 basis = Matrix3(1.0f, 1.0f, 1.0f);
		Vector3 origin(0.0f, 0.0f, 0.0f);

		BaseGrid<float> grid1(basis, origin, 5, 5, 5);
		BaseGrid<float> grid2(basis, origin, 5, 5, 5);

		// Fill grids with different patterns
		for (size_t i = 0; i < grid1.size(); ++i) {
			grid1[i] = static_cast<float>(i);
			grid2[i] = static_cast<float>(i + 100);
		}

		// Test multiplication
		grid1.multiply(grid2);

		REQUIRE(grid1[0] == Approx(0.0f * 100.0f));
		REQUIRE(grid1[10] == Approx(10.0f * 110.0f));
	}

	SECTION("Grid with different basis") {
		// Test non-orthogonal grid
		Matrix3 basis(Vector3(1.0f, 0.5f, 0.0f),
					  Vector3(0.0f, 1.0f, 0.5f),
					  Vector3(0.0f, 0.0f, 1.0f));
		Vector3 origin(0.0f, 0.0f, 0.0f);

		BaseGrid<float> grid(basis, origin, 3, 3, 3);

		// Test coordinate transformations work correctly
		Vector3 grid_pos(1.0f, 1.0f, 1.0f);
		Vector3 world_pos = grid.transform_to_world(grid_pos);
		Vector3 back_to_grid = grid.transform_to_grid(world_pos);

		REQUIRE(back_to_grid.x == Approx(grid_pos.x).margin(1e-5f));
		REQUIRE(back_to_grid.y == Approx(grid_pos.y).margin(1e-5f));
		REQUIRE(back_to_grid.z == Approx(grid_pos.z).margin(1e-5f));
	}

	SECTION("Real-world scenario: potential field") {
		// Simulate a simple potential field around a point charge
		Matrix3 basis = Matrix3(0.1f, 0.1f, 0.1f); // 0.1 unit spacing
		Vector3 origin(-2.0f, -2.0f, -2.0f);				 // Center grid around origin
		BaseGrid<float> potential_grid(basis, origin, 40, 40, 40);

		Vector3 charge_pos(0.0f, 0.0f, 0.0f); // Charge at center
		float charge_strength = 1.0f;

		// Fill grid with 1/r potential
		for (size_t ix = 0; ix < 40; ++ix) {
			for (size_t iy = 0; iy < 40; ++iy) {
				for (size_t iz = 0; iz < 40; ++iz) {
					size_t idx = potential_grid.index(ix, iy, iz);
					Vector3 pos = potential_grid.get_position(idx);
					float r = (pos - charge_pos).length();

					if (r > 0.01f) { // Avoid singularity
						potential_grid[idx] = charge_strength / r;
					} else {
						potential_grid[idx] = 100.0f; // Large value near charge
					}
				}
			}
		}

		// Test interpolation at known positions
		Vector3 test_pos(1.0f, 0.0f, 0.0f); // 1 unit from charge
		float interpolated_potential = potential_grid.interpolate(test_pos);
		float expected_potential = charge_strength / 1.0f; // = 1.0

		REQUIRE(interpolated_potential == Approx(expected_potential).margin(0.1f));

		// Test gradient (should point toward charge)
		Vector3 gradient = potential_grid.compute_gradient(test_pos);
		REQUIRE(gradient.x < 0.0f);			  // Should point toward negative x (toward charge)
		REQUIRE(std::abs(gradient.y) < 0.1f); // Should be near zero
		REQUIRE(std::abs(gradient.z) < 0.1f); // Should be near zero
	}
}

/*=======================*\
|  EDGE CASE TESTS        |
\*=======================*/

TEST_CASE("BaseGrid Edge Cases", "[BaseGrid][edge_cases]") {

	SECTION("Minimal grid (1x1x1)") {
		Matrix3 basis = Matrix3(1.0f, 1.0f, 1.0f);
		Vector3 origin(0.0f, 0.0f, 0.0f);
		BaseGrid<float> mini_grid(basis, origin, 1, 1, 1);

		REQUIRE(mini_grid.size() == 1);
		mini_grid[0] = 42.0f;

		// All interpolation should return the same value
		Vector3 pos1(0.0f, 0.0f, 0.0f);
		Vector3 pos2(0.5f, 0.5f, 0.5f);

		REQUIRE(mini_grid.interpolate(pos1) == Approx(42.0f));
		REQUIRE(mini_grid.get_value(pos2) == Approx(42.0f));
	}

	SECTION("Very large spacing") {
		Matrix3 basis = Matrix3(1000.0f, 1000.0f, 1000.0f);
		Vector3 origin(-500.0f, -500.0f, -500.0f);
		BaseGrid<float> sparse_grid(basis, origin, 2, 2, 2);

		// Fill with pattern
		for (size_t i = 0; i < sparse_grid.size(); ++i) {
			sparse_grid[i] = static_cast<float>(i);
		}

		// Test coordinates work correctly with large spacing
		Vector3 center(0.0f, 0.0f, 0.0f);
		REQUIRE(sparse_grid.in_bounds(center));

		float value = sparse_grid.interpolate(center);
		REQUIRE(value >= 0.0f);
		REQUIRE(value <= 7.0f); // Max value in 2x2x2 grid
	}

	SECTION("Near-zero spacing") {
		Matrix3 basis = Matrix3(1e-6f, 1e-6f, 1e-6f);
		Vector3 origin(0.0f, 0.0f, 0.0f);
		BaseGrid<float> dense_grid(basis, origin, 3, 3, 3);

		// Should handle very small coordinates
		Vector3 tiny_pos(1e-6f, 1e-6f, 1e-6f);
		REQUIRE(dense_grid.in_bounds(tiny_pos));

		dense_grid[13] = 123.0f; // Center point
		REQUIRE(dense_grid[13] == Approx(123.0f));
	}

	SECTION("Boundary interpolation") {
		auto grid = create_test_grid(); // 4x3x2 grid

		// Test interpolation exactly at boundary (should not crash)
		Vector3 boundary_pos(3.99f, 2.99f, 1.99f); // Just inside bounds

		REQUIRE_NOTHROW(grid.interpolate(boundary_pos));

		// Test outside bounds (should return 0)
		Vector3 outside_pos(5.0f, 5.0f, 5.0f);
		REQUIRE(grid.interpolate(outside_pos) == Approx(0.0f));
	}
}

/*=======================*\
|  STRESS TESTS           |
\*=======================*/

TEST_CASE("BaseGrid Stress Tests", "[BaseGrid][stress]") {

	SECTION("Many operations") {
		auto grid = create_test_grid();

		// Perform many operations without crashing
		for (int i = 0; i < 1000; ++i) {
			grid.shift(0.1f);
			grid.scale(0.999f);

			Vector3 pos(static_cast<float>(i % 4),
						static_cast<float>((i / 4) % 3),
						static_cast<float>((i / 12) % 2));

			float value = grid.interpolate(pos);
			REQUIRE(std::isfinite(value));
		}
	}

	SECTION("Copy and move stress") {
		std::vector<BaseGrid<float>> grids;

		// Create many grids
		for (int i = 0; i < 100; ++i) {
			Matrix3 basis = Matrix3(1.0f + i * 0.01f, 1.0f, 1.0f);
			Vector3 origin(static_cast<float>(i), 0.0f, 0.0f);
			grids.emplace_back(basis, origin, 5, 5, 5);
			grids.back()[0] = static_cast<float>(i);
		}

		// Verify all grids are correct
		for (int i = 0; i < 100; ++i) {
			REQUIRE(grids[i][0] == Approx(static_cast<float>(i)));
			REQUIRE(grids[i].size() == 125);
		}
	}
}

/*=======================*\
|  BACKEND COMPATIBILITY  |
\*=======================*/

TEST_CASE("Backend Compatibility", "[BaseGrid][backend]") {

	SECTION("Grid config device accessibility") {
		auto grid = create_test_grid();

		// These should be accessible from device code
		auto config = grid.config();
		auto dimensions = grid.dimensions();
		auto origin = grid.origin();
		auto basis = grid.basis();
		auto basis_inv = grid.basis_inverse();

		REQUIRE(dimensions.x == 4);
		REQUIRE(dimensions.y == 3);
		REQUIRE(dimensions.z == 2);
		REQUIRE(config.total_size() == 24);
	}

	SECTION("Device function signatures") {
		auto grid = create_test_grid();

		// These functions should be callable with the right signatures
		float* data_ptr = grid.data();
		Vector3 pos(1.0f, 1.0f, 0.5f);

		REQUIRE_NOTHROW(interpolate_grid_point(data_ptr,
											   pos,
											   grid.origin(),
											   grid.basis_inverse(),
											   grid.dimensions()));

		REQUIRE_NOTHROW(get_value_nearest(data_ptr,
										  pos,
										  grid.origin(),
										  grid.basis_inverse(),
										  grid.dimensions()));

		REQUIRE_NOTHROW(get_neighbor_list_from_grid(data_ptr, 1, 1, 0, grid.dimensions()));

		REQUIRE_NOTHROW(get_neighbor_from_grid(data_ptr, 1, 1, 0, 1, 0, 0, grid.dimensions()));
	}
}
