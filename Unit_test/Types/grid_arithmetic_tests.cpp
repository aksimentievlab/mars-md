/**
 * @file grid_arithmetic_tests.cpp
 * @brief BaseGrid host-side arithmetic, reductions and reshaping.
 */

#include "../catch_boiler.h"
#include "Types/BaseGrid.h"
#include <cmath>

using namespace ARBD;
using Catch::Approx;

namespace {

/// BaseGrid::index() is private, so mirror its layout here.
idx_t lin(const BaseGrid<arbd_real>& g, idx_t ix, idx_t iy, idx_t iz) {
	return iz + iy * g.nz() + ix * g.ny() * g.nz();
}

/// nx*ny*nz grid with spacing dx, value = its x index.
BaseGrid<arbd_real> ramp(idx_t nx, idx_t ny, idx_t nz, arbd_real dx = arbd_real(1)) {
	BaseGrid<arbd_real> g(Matrix3(dx), Vector3(0, 0, 0), nx, ny, nz);
	for (idx_t ix = 0; ix < nx; ++ix)
		for (idx_t iy = 0; iy < ny; ++iy)
			for (idx_t iz = 0; iz < nz; ++iz)
				g[lin(g, ix, iy, iz)] = arbd_real(ix);
	return g;
}

BaseGrid<arbd_real> constant(idx_t n, arbd_real v) {
	BaseGrid<arbd_real> g(Matrix3(arbd_real(1)), Vector3(0, 0, 0), n, n, n);
	for (idx_t i = 0; i < g.size(); ++i)
		g[i] = v;
	return g;
}

} // namespace

TEST_CASE("Grid add and subtract are elementwise", "[grid][arithmetic]") {
	auto a = constant(3, 2.0f);
	const auto b = constant(3, 0.5f);

	a.add(b);
	REQUIRE(a[0] == Approx(2.5));
	REQUIRE(a[a.size() - 1] == Approx(2.5));

	a.subtract(b);
	REQUIRE(a[0] == Approx(2.0));
}

TEST_CASE("Grid arithmetic rejects size mismatch", "[grid][arithmetic]") {
	auto a = constant(3, 1.0f);
	const auto b = constant(4, 1.0f);

	REQUIRE_THROWS_AS(a.add(b), Exception);
	REQUIRE_THROWS_AS(a.subtract(b), Exception);
	REQUIRE_THROWS_AS(a.multiply(b), Exception);
}

TEST_CASE("integrate is sum times cell volume", "[grid][arithmetic]") {
	// 2 A spacing -> cell volume 8; 4^3 cells of value 0.5 -> 64*0.5*8 = 256
	BaseGrid<arbd_real> g(Matrix3(arbd_real(2)), Vector3(0, 0, 0), 4, 4, 4);
	for (idx_t i = 0; i < g.size(); ++i)
		g[i] = 0.5f;

	REQUIRE(g.get_cell_volume() == Approx(8.0));
	REQUIRE(g.integrate() == Approx(256.0));
}

TEST_CASE("integrate of a ramp matches the analytic sum", "[grid][arithmetic]") {
	const auto g = ramp(4, 4, 4);
	// sum over ix of ix, replicated 16 times per plane: (0+1+2+3)*16 = 96, dV = 1
	REQUIRE(g.integrate() == Approx(96.0));
}

TEST_CASE("min, max and mean", "[grid][arithmetic]") {
	const auto g = ramp(4, 2, 2);
	REQUIRE(g.min() == Approx(0.0));
	REQUIRE(g.max() == Approx(3.0));
	REQUIRE(g.mean() == Approx(1.5));
}

TEST_CASE("has_non_finite detects NaN and infinity", "[grid][arithmetic]") {
	auto g = constant(3, 1.0f);
	REQUIRE_FALSE(g.has_non_finite());

	g[5] = std::numeric_limits<arbd_real>::quiet_NaN();
	REQUIRE(g.has_non_finite());

	auto h = constant(3, 1.0f);
	h[2] = std::numeric_limits<arbd_real>::infinity();
	REQUIRE(h.has_non_finite());
}

TEST_CASE("resample preserves world extent", "[grid][arithmetic][resample]") {
	const auto g = ramp(4, 4, 4, arbd_real(2)); // extent 8 A per axis
	const auto r = g.resample(Vector3_t<idx_t>(8, 8, 8));

	REQUIRE(r.nx() == 8);
	REQUIRE(r.origin().x == Approx(g.origin().x));
	// Halving the spacing doubles the count, so total volume is unchanged.
	REQUIRE(r.get_total_volume() == Approx(g.get_total_volume()));
	REQUIRE(r.get_cell_volume() == Approx(g.get_cell_volume() / 8.0));
}

TEST_CASE("resample of a constant grid stays constant", "[grid][arithmetic][resample]") {
	const auto g = constant(4, 3.25f);
	const auto r = g.resample(Vector3_t<idx_t>(7, 5, 3));

	for (idx_t i = 0; i < r.size(); ++i) {
		REQUIRE(r[i] == Approx(3.25));
	}
}

TEST_CASE("resample handles odd target dimensions", "[grid][arithmetic][resample]") {
	// The reference C implementation silently skipped resampling for odd nz
	// (it tested ny twice); make sure odd dimensions are honored here.
	const auto g = ramp(4, 4, 4);
	const auto r = g.resample(Vector3_t<idx_t>(5, 5, 5));

	REQUIRE(r.nx() == 5);
	REQUIRE(r.ny() == 5);
	REQUIRE(r.nz() == 5);
}

TEST_CASE("resample rejects zero dimensions", "[grid][arithmetic][resample]") {
	const auto g = ramp(4, 4, 4);
	REQUIRE_THROWS_AS(g.resample(Vector3_t<idx_t>(0, 4, 4)), Exception);
}

TEST_CASE("pad surrounds the original with fill", "[grid][arithmetic][pad]") {
	const auto g = constant(2, 7.0f);
	const auto p = g.pad(Vector3_t<idx_t>(1, 1, 1), Vector3_t<idx_t>(2, 2, 2), -1.0f);

	REQUIRE(p.nx() == 5); // 1 + 2 + 2
	REQUIRE(p[lin(p, 0, 0, 0)] == Approx(-1.0));       // padded corner
	REQUIRE(p[lin(p, 1, 1, 1)] == Approx(7.0));        // original origin
	REQUIRE(p[lin(p, 2, 2, 2)] == Approx(7.0));        // original far corner
	REQUIRE(p[lin(p, 3, 3, 3)] == Approx(-1.0));       // padded again
	// Origin shifts down by one cell on each axis.
	REQUIRE(p.origin().x == Approx(g.origin().x - 1.0));
}

TEST_CASE("crop extracts a sub-box and shifts the origin", "[grid][arithmetic][crop]") {
	const auto g = ramp(4, 4, 4);
	const auto c = g.crop(Vector3_t<idx_t>(1, 0, 0), Vector3_t<idx_t>(2, 4, 4));

	REQUIRE(c.nx() == 2);
	REQUIRE(c.ny() == 4);
	REQUIRE(c.origin().x == Approx(1.0));
	// Values carry over: the ramp starts at ix=1 now.
	REQUIRE(c[lin(c, 0, 0, 0)] == Approx(1.0));
	REQUIRE(c[lin(c, 1, 0, 0)] == Approx(2.0));
}

TEST_CASE("crop rejects out-of-range requests", "[grid][arithmetic][crop]") {
	const auto g = ramp(4, 4, 4);
	REQUIRE_THROWS_AS(g.crop(Vector3_t<idx_t>(2, 0, 0), Vector3_t<idx_t>(3, 4, 4)), Exception);
	REQUIRE_THROWS_AS(g.crop(Vector3_t<idx_t>(0, 0, 0), Vector3_t<idx_t>(0, 4, 4)), Exception);
}

TEST_CASE("pad then crop round-trips", "[grid][arithmetic][pad][crop]") {
	const auto g = ramp(3, 3, 3);
	const auto back = g.pad(Vector3_t<idx_t>(2, 1, 3), Vector3_t<idx_t>(1, 2, 1), 0.0f)
						  .crop(Vector3_t<idx_t>(2, 1, 3), Vector3_t<idx_t>(3, 3, 3));

	REQUIRE(back.nx() == g.nx());
	for (idx_t i = 0; i < g.size(); ++i) {
		REQUIRE(back[i] == Approx(g[i]));
	}
	REQUIRE(back.origin().x == Approx(g.origin().x));
}
