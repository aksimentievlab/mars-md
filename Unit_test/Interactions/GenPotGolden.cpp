/**
 * @file GenPotGolden.cpp
 * @brief Compares convolve_grids against the reference gen_pot tool.
 */

#include "../catch_boiler.h"
#include "IO/DxIO.h"
#include "Interactions/NonBondedInteraction.h"
#include "Types/BaseGrid.h"
#include <cmath>
#include <filesystem>
#include <string>

using namespace ARBD;
using Catch::Approx;

namespace {

/// gen_pot's lenard_jones_repulsion_compute, verbatim (Kernel/*.c).
double wca(double r2, double c6, double c12) {
	const double sigma2 = std::pow(c12 / c6, 1.0 / 3.0);
	const double epsilon = c6 * c6 / (4 * c12);
	if (r2 <= std::pow(2.0, 1.0 / 3.0) * sigma2) {
		const double d6 = r2 * r2 * r2;
		if (d6 > 0.0) {
			const double e = (-c6 / d6 + c12 / (d6 * d6)) + epsilon;
			return e > 100.0 ? 100.0 : e;
		}
		return 100.0;
	}
	return 0.0;
}

/**
 * @brief Kernel grid sampled about its own center, offsets (m - (n-1)/2)*dx.
 * @param n Width per axis; even n gives gen_pot's half-integer sampling, odd n
 *        samples r=0 (where WCA saturates at the cap).
 */
BaseGrid<arbd_real> build_wca_kernel(arbd_real dx, idx_t n, double c6, double c12) {
	BaseGrid<arbd_real> k(Matrix3(dx), Vector3(0, 0, 0), n, n, n);
	const double center = (double(n) - 1.0) / 2.0;
	for (idx_t a = 0; a < n; ++a) {
		for (idx_t b = 0; b < n; ++b) {
			for (idx_t c = 0; c < n; ++c) {
				const double x = (double(a) - center) * double(dx);
				const double y = (double(b) - center) * double(dx);
				const double z = (double(c) - center) * double(dx);
				k[c + b * n + a * n * n] =
					static_cast<arbd_real>(wca(x * x + y * y + z * z, c6, c12));
			}
		}
	}
	return k;
}

double sum_of(const BaseGrid<arbd_real>& g) {
	double s = 0;
	for (idx_t i = 0; i < g.size(); ++i)
		s += double(g[i]);
	return s;
}

} // namespace

TEST_CASE("convolve_grids reproduces gen_pot's WCA potential", "[grid][convolution]") {

	initialize_backend_once();
	Resource res(Global::single_resource_id);

	const std::string stem =
		(std::filesystem::path(__FILE__).parent_path() / "1ema.protein").string();
	const double c6 = 1228.8;
	const double c12 = 2516582.4;

	const auto density = DXReader::read_from_file<arbd_real>(stem + ".Density.dx");
	const auto reference = DXReader::read_from_file<arbd_real>(stem + ".pot.dx");

	REQUIRE(density.nx() == reference.nx());
	REQUIRE(density.ny() == reference.ny());
	REQUIRE(density.nz() == reference.nz());

	// WCA vanishes past 2^(1/6) sigma. An even width reproduces gen_pot's
	// half-integer sampling: this grid's origin puts world-0 half a voxel off a
	// lattice point, so gen_pot never samples r=0. Sampling on integers instead
	// changes sum(K) by 8.5% here - the kernel is steep and capped near the
	// core - which would show up as a bogus normalization error.
	const double sigma = std::sqrt(std::pow(c12 / c6, 1.0 / 3.0));
	const double cutoff = std::pow(2.0, 1.0 / 6.0) * sigma;
	const arbd_real dx = density.basis().ex().x;
	const idx_t n = 2 * static_cast<idx_t>(std::ceil(cutoff / double(dx)));

	const auto kernel = build_wca_kernel(dx, n, c6, c12);
	const auto out = convolve_grids(density, kernel, res, ConvolutionNormalization::VoxelSum);

	// Total mass is invariant under any displacement, so it isolates the
	// normalization and kernel amplitude from the centering convention.
	const double sum_out = sum_of(out);
	const double sum_ref = sum_of(reference);
	INFO("sum(ours)=" << sum_out << " sum(gen_pot)=" << sum_ref << " ratio=" << sum_out / sum_ref);
	REQUIRE(sum_out == Approx(sum_ref).epsilon(1e-4));

	// Extremes should also match to within interpolation of the half-voxel offset.
	INFO("max ours=" << out.max() << " ref=" << reference.max());
	INFO("min ours=" << out.min() << " ref=" << reference.min());
	REQUIRE(out.max() == Approx(reference.max()).epsilon(5e-2));
	REQUIRE_FALSE(out.has_non_finite());
}
