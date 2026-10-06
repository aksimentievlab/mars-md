/**
 * @file grid_boundary_tests.cpp
 * @brief Boundary-condition handling in the shared grid samplers (host side).
 */

#include "../catch_boiler.h"
#include "Types/BaseGridDevice.h"
#include <vector>

using namespace MARS;
using Catch::Approx;

namespace {

constexpr int kDirichlet = static_cast<int>(GridBoundaryCondition::Dirichlet);
constexpr int kNeumann = static_cast<int>(GridBoundaryCondition::Neumann);
constexpr int kPeriodic = static_cast<int>(GridBoundaryCondition::Periodic);

/// 4x4x4 grid whose value is its x index, so edge effects are easy to read off.
struct Ramp {
	Vector3_t<idx_t> dims{4, 4, 4};
	std::vector<mars_real> v;

	Ramp() : v(64) {
		for (idx_t ix = 0; ix < 4; ++ix)
			for (idx_t iy = 0; iy < 4; ++iy)
				for (idx_t iz = 0; iz < 4; ++iz)
					v[iz + iy * 4 + ix * 16] = mars_real(ix);
	}
	const mars_real* data() const {
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

TEST_CASE("gradient is the derivative of the trilinear energy", "[grid][boundary][gradient]") {
	const Ramp g;
	const Vector3 origin(0, 0, 0);
	const Matrix3 identity(1.0f);

	// Interior: d/dx of the ramp is 1 under every BC.
	const Vector3 inside(2.0f, 2.0f, 2.0f);
	for (int bc : {kDirichlet, kNeumann, kPeriodic}) {
		REQUIRE(compute_gradient(g.data(), inside, origin, identity, identity, g.dims, bc).x ==
				Approx(1.0));
	}

	// Edge cell [0,1): both taps are in range, so every BC sees the ramp slope.
	const Vector3 edge(0.25f, 2.0f, 2.0f);
	for (int bc : {kDirichlet, kNeumann}) {
		const auto e = compute_gradient(g.data(), edge, origin, identity, identity, g.dims, bc);
		REQUIRE(e.x == Approx(1.0));
		REQUIRE(e.y == Approx(0.0));
		REQUIRE(e.z == Approx(0.0));
	}

	// Last cell [3,4): the +1 tap is padding (0), clamped (3) or wrapped (0).
	const Vector3 last(3.5f, 2.0f, 2.0f);
	REQUIRE(compute_gradient(g.data(), last, origin, identity, identity, g.dims, kDirichlet).x ==
			Approx(-3.0));
	REQUIRE(compute_gradient(g.data(), last, origin, identity, identity, g.dims, kNeumann).x ==
			Approx(0.0));
	REQUIRE(compute_gradient(g.data(), last, origin, identity, identity, g.dims, kPeriodic).x ==
			Approx(-3.0));
}

TEST_CASE("gradient matches finite difference of the energy off-grid",
		  "[grid][boundary][gradient]") {
	// Non-separable field so every cross term in the trilinear blend is exercised.
	const Vector3_t<idx_t> dims{6, 6, 6};
	std::vector<mars_real> v(dims.x * dims.y * dims.z);
	for (idx_t ix = 0; ix < dims.x; ++ix)
		for (idx_t iy = 0; iy < dims.y; ++iy)
			for (idx_t iz = 0; iz < dims.z; ++iz)
				v[iz + iy * dims.z + ix * dims.y * dims.z] =
					mars_real(ix * ix) - mars_real(2 * iy * iz) + mars_real(ix * iy * iz) * 0.5f;

	// Sheared basis checks the grid->world transform of the gradient too.
	const Matrix3 basis(Vector3(1.2f, 0.0f, 0.0f), Vector3(0.3f, 0.9f, 0.0f), Vector3(0.0f, 0.2f, 1.1f));
	const Matrix3 basis_inv = basis.inverse();
	const Vector3 origin(-0.5f, 0.25f, 1.0f);

	// Off-node points; h small enough to stay inside the cell.
	const Vector3 probes_grid[] = {Vector3(1.3f, 2.6f, 3.2f),
								   Vector3(2.7f, 1.4f, 1.8f),
								   Vector3(3.1f, 3.9f, 2.45f)};
	const mars_real h = 1e-2f;
	for (const auto& pg : probes_grid) {
		const Vector3 p = basis.transform(pg) + origin;
		const auto grad =
			compute_gradient(v.data(), p, origin, basis, basis_inv, dims, kDirichlet);
		const Vector3 axes[] = {Vector3(h, 0, 0), Vector3(0, h, 0), Vector3(0, 0, h)};
		mars_real fd[3];
		for (int a = 0; a < 3; ++a) {
			const mars_real ep =
				interpolate_grid_point(v.data(), p + axes[a], origin, basis_inv, dims, kDirichlet);
			const mars_real em =
				interpolate_grid_point(v.data(), p - axes[a], origin, basis_inv, dims, kDirichlet);
			fd[a] = (ep - em) / (2 * h);
		}
		REQUIRE(grad.x == Approx(fd[0]).epsilon(1e-3).margin(1e-3));
		REQUIRE(grad.y == Approx(fd[1]).epsilon(1e-3).margin(1e-3));
		REQUIRE(grad.z == Approx(fd[2]).epsilon(1e-3).margin(1e-3));
	}
}

TEST_CASE("gradient is zero inside a flat cell", "[grid][boundary][gradient]") {
	// Cell [1,2)^3 has all 8 corners equal; its outer neighbors do not.
	const Vector3_t<idx_t> dims{4, 4, 4};
	std::vector<mars_real> v(64);
	for (idx_t ix = 0; ix < 4; ++ix)
		for (idx_t iy = 0; iy < 4; ++iy)
			for (idx_t iz = 0; iz < 4; ++iz) {
				const bool corner = ix >= 1 && ix <= 2 && iy >= 1 && iy <= 2 && iz >= 1 && iz <= 2;
				v[iz + iy * 4 + ix * 16] = corner ? mars_real(5) : mars_real(ix + 3 * iy - iz);
			}

	const Vector3 origin(0, 0, 0);
	const Matrix3 identity(1.0f);
	const Vector3 p(1.4f, 1.6f, 1.3f);
	const auto grad = compute_gradient(v.data(), p, origin, identity, identity, dims, kDirichlet);
	REQUIRE(grad.x == Approx(0.0));
	REQUIRE(grad.y == Approx(0.0));
	REQUIRE(grad.z == Approx(0.0));
}
