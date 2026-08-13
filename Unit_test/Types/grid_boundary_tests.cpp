/**
 * @file grid_boundary_tests.cpp
 * @brief Boundary-condition handling in the shared grid samplers (host side).
 */

#include "../catch_boiler.h"
#include "Types/BaseGridDevice.h"
#include <vector>

using namespace ARBD;
using Catch::Approx;

namespace {

constexpr int kDirichlet = static_cast<int>(GridBoundaryCondition::Dirichlet);
constexpr int kNeumann = static_cast<int>(GridBoundaryCondition::Neumann);
constexpr int kPeriodic = static_cast<int>(GridBoundaryCondition::Periodic);

/// 4x4x4 grid whose value is its x index, so edge effects are easy to read off.
struct Ramp {
	Vector3_t<idx_t> dims{4, 4, 4};
	std::vector<arbd_real> v;

	Ramp() : v(64) {
		for (idx_t ix = 0; ix < 4; ++ix)
			for (idx_t iy = 0; iy < 4; ++iy)
				for (idx_t iz = 0; iz < 4; ++iz)
					v[iz + iy * 4 + ix * 16] = arbd_real(ix);
	}
	const arbd_real* data() const {
		return v.data();
	}
};

} // namespace

TEST_CASE("map_grid_index applies each boundary condition", "[grid][boundary]") {
	int j;

	SECTION("in range is untouched") {
		for (int bc : {kDirichlet, kNeumann, kPeriodic}) {
			j = 2;
			REQUIRE(map_grid_index(j, 4, bc));
			REQUIRE(j == 2);
		}
	}

	SECTION("Dirichlet rejects") {
		j = -1;
		REQUIRE_FALSE(map_grid_index(j, 4, kDirichlet));
		j = 4;
		REQUIRE_FALSE(map_grid_index(j, 4, kDirichlet));
	}

	SECTION("Neumann clamps to the edge") {
		j = -3;
		REQUIRE(map_grid_index(j, 4, kNeumann));
		REQUIRE(j == 0);
		j = 9;
		REQUIRE(map_grid_index(j, 4, kNeumann));
		REQUIRE(j == 3);
	}

	SECTION("Periodic wraps, including negatives") {
		j = -1;
		REQUIRE(map_grid_index(j, 4, kPeriodic));
		REQUIRE(j == 3);
		j = -5;
		REQUIRE(map_grid_index(j, 4, kPeriodic));
		REQUIRE(j == 3);
		j = 5;
		REQUIRE(map_grid_index(j, 4, kPeriodic));
		REQUIRE(j == 1);
	}

	SECTION("unrecognized values fall back to Dirichlet") {
		j = -1;
		REQUIRE_FALSE(map_grid_index(j, 4, -1)); // GridTerm's "use the grid's own"
		j = -1;
		REQUIRE_FALSE(map_grid_index(j, 4, 99));
	}
}

TEST_CASE("fetch_grid_value honors the boundary condition", "[grid][boundary]") {
	const Ramp g;

	// Interior is identical under every BC.
	for (int bc : {kDirichlet, kNeumann, kPeriodic}) {
		REQUIRE(fetch_grid_value(g.data(), 2, 1, 1, g.dims, bc) == Approx(2.0));
	}

	REQUIRE(fetch_grid_value(g.data(), -1, 1, 1, g.dims, kDirichlet) == Approx(0.0));
	REQUIRE(fetch_grid_value(g.data(), -1, 1, 1, g.dims, kNeumann) == Approx(0.0));  // clamps to ix=0
	REQUIRE(fetch_grid_value(g.data(), -1, 1, 1, g.dims, kPeriodic) == Approx(3.0)); // wraps to ix=3
	REQUIRE(fetch_grid_value(g.data(), 4, 1, 1, g.dims, kNeumann) == Approx(3.0));
	REQUIRE(fetch_grid_value(g.data(), 4, 1, 1, g.dims, kPeriodic) == Approx(0.0));
}

TEST_CASE("interpolation outside the grid follows the boundary condition",
		  "[grid][boundary][interpolate]") {
	const Ramp g;
	const Vector3 origin(0, 0, 0);
	const Matrix3 identity(1.0f);
	// Well outside on -x, still inside on y/z.
	const Vector3 outside(-2.5f, 1.0f, 1.0f);

	REQUIRE(interpolate_grid_point(g.data(), outside, origin, identity, g.dims, kDirichlet) ==
			Approx(0.0));
	// Neumann replicates ix=0, whose value is 0 - so it also reads 0, but for a
	// different reason. Probe +x instead, where the edge value is nonzero.
	const Vector3 outside_px(6.0f, 1.0f, 1.0f);
	REQUIRE(interpolate_grid_point(g.data(), outside_px, origin, identity, g.dims, kNeumann) ==
			Approx(3.0));
	REQUIRE(interpolate_grid_point(g.data(), outside_px, origin, identity, g.dims, kDirichlet) ==
			Approx(0.0));
}

TEST_CASE("interpolation inside the grid is unaffected by the boundary condition",
		  "[grid][boundary][interpolate]") {
	const Ramp g;
	const Vector3 origin(0, 0, 0);
	const Matrix3 identity(1.0f);
	const Vector3 inside(1.5f, 1.5f, 1.5f);

	const auto d = interpolate_grid_point(g.data(), inside, origin, identity, g.dims, kDirichlet);
	const auto n = interpolate_grid_point(g.data(), inside, origin, identity, g.dims, kNeumann);
	const auto p = interpolate_grid_point(g.data(), inside, origin, identity, g.dims, kPeriodic);

	REQUIRE(d == Approx(1.5)); // ramp in x
	REQUIRE(n == Approx(d));
	REQUIRE(p == Approx(d));
}

TEST_CASE("nearest lookup follows the boundary condition", "[grid][boundary][nearest]") {
	const Ramp g;
	const Vector3 origin(0, 0, 0);
	const Matrix3 identity(1.0f);
	const Vector3 outside_px(5.0f, 1.0f, 1.0f);

	REQUIRE(get_value_nearest(g.data(), outside_px, origin, identity, g.dims, kDirichlet) ==
			Approx(0.0));
	REQUIRE(get_value_nearest(g.data(), outside_px, origin, identity, g.dims, kNeumann) ==
			Approx(3.0));
	REQUIRE(get_value_nearest(g.data(), outside_px, origin, identity, g.dims, kPeriodic) ==
			Approx(1.0)); // 5 mod 4
}

TEST_CASE("gradient keeps the Dirichlet edge guard", "[grid][boundary][gradient]") {
	const Ramp g;
	const Vector3 origin(0, 0, 0);
	const Matrix3 identity(1.0f);

	// Interior: d/dx of the ramp is 1 under every BC.
	const Vector3 inside(2.0f, 2.0f, 2.0f);
	for (int bc : {kDirichlet, kNeumann, kPeriodic}) {
		REQUIRE(compute_gradient(g.data(), inside, origin, identity, identity, g.dims, bc).x ==
				Approx(1.0));
	}

	// At the edge Dirichlet returns zero rather than differencing against padding.
	const Vector3 edge(0.0f, 2.0f, 2.0f);
	const auto d = compute_gradient(g.data(), edge, origin, identity, identity, g.dims, kDirichlet);
	REQUIRE(d.x == Approx(0.0));
	REQUIRE(d.y == Approx(0.0));
	REQUIRE(d.z == Approx(0.0));

	// Neumann has a real neighbor to difference against, so it does not zero out.
	const auto n = compute_gradient(g.data(), edge, origin, identity, identity, g.dims, kNeumann);
	REQUIRE(n.x == Approx(0.5)); // (v[1] - v[0]) / 2 with the -1 tap clamped to ix=0
}
