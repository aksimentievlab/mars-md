/**
 * @file Convolution.cpp
 * @brief Real-space grid convolution (ConvolveGridKernel / convolve_grids).
 */

#include "../catch_boiler.h"
#include "Interactions/NonBondedInteraction.h"
#include "Types/BaseGrid.h"

using namespace MARS;
using Catch::Approx;

namespace {

idx_t lin(const BaseGrid<mars_real>& g, idx_t ix, idx_t iy, idx_t iz) {
	return iz + iy * g.nz() + ix * g.ny() * g.nz();
}

BaseGrid<mars_real> make_grid(idx_t n, mars_real dx = mars_real(1)) {
	BaseGrid<mars_real> g(Matrix3(dx), Vector3(0, 0, 0), n, n, n);
	g.zero();
	return g;
}

/// Density that is 1 at one voxel and 0 elsewhere.
BaseGrid<mars_real> delta(idx_t n, idx_t ix, idx_t iy, idx_t iz) {
	auto g = make_grid(n);
	g[lin(g, ix, iy, iz)] = mars_real(1);
	return g;
}

} // namespace

TEST_CASE("Convolution with a delta density reproduces the kernel", "[grid][convolution]") {
	initialize_backend_once();
	Resource res(Global::single_resource_id);

	// 3^3 kernel with distinct values, centered on (1,1,1).
	auto kern = make_grid(3);
	for (idx_t i = 0; i < kern.size(); ++i)
		kern[i] = mars_real(i + 1);

	// Delta at the middle of a 7^3 box, far from any edge.
	const auto dens = delta(7, 3, 3, 3);
	const auto out = convolve_grids(dens, kern, res);

	// out(3+a-1, 3+b-1, 3+c-1) == kern(a,b,c)
	for (idx_t a = 0; a < 3; ++a) {
		for (idx_t b = 0; b < 3; ++b) {
			for (idx_t c = 0; c < 3; ++c) {
				REQUIRE(out[lin(out, 3 + a - 1, 3 + b - 1, 3 + c - 1)] ==
						Approx(kern[lin(kern, a, b, c)]));
			}
		}
	}
}

TEST_CASE("Convolution leaves everything outside the kernel support at zero",
		  "[grid][convolution]") {
	initialize_backend_once();
	Resource res(Global::single_resource_id);

	auto kern = make_grid(3);
	for (idx_t i = 0; i < kern.size(); ++i)
		kern[i] = mars_real(1);

	const auto dens = delta(7, 3, 3, 3);
	const auto out = convolve_grids(dens, kern, res);

	// Only the 3^3 block around (3,3,3) is nonzero.
	idx_t nonzero = 0;
	for (idx_t i = 0; i < out.size(); ++i) {
		if (out[i] != mars_real(0))
			++nonzero;
	}
	REQUIRE(nonzero == 27);
}

TEST_CASE("Convolution is linear in the density", "[grid][convolution]") {
	initialize_backend_once();
	Resource res(Global::single_resource_id);

	auto kern = make_grid(3);
	for (idx_t i = 0; i < kern.size(); ++i)
		kern[i] = mars_real(0.5);

	auto a = delta(7, 2, 3, 3);
	const auto b = delta(7, 4, 3, 3);
	auto sum = a;
	sum.add(b);

	const auto ca = convolve_grids(a, kern, res);
	const auto cb = convolve_grids(b, kern, res);
	const auto csum = convolve_grids(sum, kern, res);

	for (idx_t i = 0; i < csum.size(); ++i) {
		REQUIRE(csum[i] == Approx(ca[i] + cb[i]));
	}
}

TEST_CASE("Convolution sum rule: integral of output equals product of integrals",
		  "[grid][convolution]") {
	initialize_backend_once();
	Resource res(Global::single_resource_id);

	// Periodic density so no support is lost off the edges.
	auto dens = make_grid(8);
	for (idx_t i = 0; i < dens.size(); ++i)
		dens[i] = mars_real(0.25);
	dens.set_boundary(GridBoundaryCondition::Periodic);

	auto kern = make_grid(3);
	for (idx_t i = 0; i < kern.size(); ++i)
		kern[i] = mars_real(2);

	const auto out = convolve_grids(dens, kern, res);

	// VoxelSum: sum(out) == sum(dens) * sum(kern)
	const mars_real sum_dens = dens.integrate() / dens.get_cell_volume();
	const mars_real sum_kern = kern.integrate() / kern.get_cell_volume();
	const mars_real sum_out = out.integrate() / out.get_cell_volume();
	REQUIRE(sum_out == Approx(sum_dens * sum_kern).epsilon(1e-4));
}

TEST_CASE("Periodic density wraps, Dirichlet does not", "[grid][convolution][boundary]") {
	initialize_backend_once();
	Resource res(Global::single_resource_id);

	auto kern = make_grid(3);
	for (idx_t i = 0; i < kern.size(); ++i)
		kern[i] = mars_real(1);

	// Delta on the x=0 face, so the kernel reaches past the low edge.
	auto dirichlet = delta(5, 0, 2, 2);
	auto periodic = dirichlet;
	periodic.set_boundary(GridBoundaryCondition::Periodic);

	const auto d_out = convolve_grids(dirichlet, kern, res);
	const auto p_out = convolve_grids(periodic, kern, res);

	// Dirichlet: the wrapped column stays empty.
	REQUIRE(d_out[lin(d_out, 4, 2, 2)] == Approx(0.0));
	// Periodic: the delta's support wraps around to ix = 4.
	REQUIRE(p_out[lin(p_out, 4, 2, 2)] == Approx(1.0));

	// Total mass is conserved under wrapping but clipped under Dirichlet.
	mars_real d_sum = 0, p_sum = 0;
	for (idx_t i = 0; i < d_out.size(); ++i) {
		d_sum += d_out[i];
		p_sum += p_out[i];
	}
	REQUIRE(p_sum == Approx(27.0)); // full 3^3 stencil
	REQUIRE(d_sum < p_sum);
}

TEST_CASE("Volume normalization scales by the density cell volume",
		  "[grid][convolution][normalization]") {
	initialize_backend_once();
	Resource res(Global::single_resource_id);

	// 2 A spacing -> cell volume 8
	BaseGrid<mars_real> dens(Matrix3(mars_real(2)), Vector3(0, 0, 0), 5, 5, 5);
	dens.zero();
	dens[lin(dens, 2, 2, 2)] = mars_real(1);

	auto kern = make_grid(3);
	for (idx_t i = 0; i < kern.size(); ++i)
		kern[i] = mars_real(1);

	const auto voxel = convolve_grids(dens, kern, res, ConvolutionNormalization::VoxelSum);
	const auto vol = convolve_grids(dens, kern, res, ConvolutionNormalization::Volume);

	REQUIRE(dens.get_cell_volume() == Approx(8.0));
	for (idx_t i = 0; i < voxel.size(); ++i) {
		REQUIRE(vol[i] == Approx(voxel[i] * 8.0));
	}
}

TEST_CASE("Convolution rejects a kernel larger than the density", "[grid][convolution]") {
	initialize_backend_once();
	Resource res(Global::single_resource_id);

	const auto dens = make_grid(3);
	const auto kern = make_grid(5);
	REQUIRE_THROWS_AS(convolve_grids(dens, kern, res), Exception);
}

TEST_CASE("Convolution preserves the density's geometry", "[grid][convolution]") {
	initialize_backend_once();
	Resource res(Global::single_resource_id);

	BaseGrid<mars_real> dens(Matrix3(mars_real(1.5)), Vector3(2, 3, 4), 6, 6, 6);
	dens.zero();
	const auto kern = make_grid(3);
	const auto out = convolve_grids(dens, kern, res);

	REQUIRE(out.nx() == dens.nx());
	REQUIRE(out.origin().x == Approx(dens.origin().x));
	REQUIRE(out.get_cell_volume() == Approx(dens.get_cell_volume()));
}
