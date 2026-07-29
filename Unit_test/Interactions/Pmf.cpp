/**
 * @file Pmf.cpp
 * @brief Tests for the flattened per-type PMF grid table consumed by
 * compute_position_dependent_force (Interactions/Nonbonded/PmfKernels.h).
 *
 * A particle type may reference any number of PMF grids - legacy ARBD's
 * `gridFile` takes a whitespace-separated list, each entry with its own scale -
 * so ParticleTypeView carries an offset+count range per type into one flat
 * GridTerm table. These tests pin down that wiring: that a type sums every one
 * of its terms, that each term's own scale applies, that one type's terms don't
 * leak into another's, and that a type with no terms is unaffected.
 *
 * Host-side only: compute_position_dependent_force is HOST DEVICE, so no GPU
 * or kernel launch is needed. The interpolation math itself is covered by
 * Types/basegrid_device_tests.cpp; here the grids carry exactly linear fields
 * (V = x and V = y in grid units), which nearest-index central differences
 * reproduce exactly, so expected forces are exact rather than approximate.
 */

#include "../catch_boiler.h"
#include "Interactions/Nonbonded/PmfKernels.h"
#include "Objects/DeviceParticle.h"
#include "Types/BaseGrid.h"

using Catch::Approx;
using namespace ARBD;

namespace {

constexpr idx_t N = 8;
constexpr float DX = 1.0f;
// Sampled well inside the grid: compute_gradient zeroes out within one cell of
// the boundary, and interpolate_grid_point's i0+1 tap needs room too.
constexpr float SAMPLE = 3.5f;

/// V(ix,iy,iz) = ix in grid units, so grad V = (1/DX, 0, 0) in world units.
BaseGrid<float> make_ramp_x() {
	BaseGrid<float> g(Matrix3(DX), Vector3(0.0f), N, N, N);
	for (idx_t ix = 0; ix < N; ++ix)
		for (idx_t iy = 0; iy < N; ++iy)
			for (idx_t iz = 0; iz < N; ++iz)
				g[iz + iy * N + ix * N * N] = static_cast<float>(ix);
	return g;
}

/// V(ix,iy,iz) = iy, so grad V = (0, 1/DX, 0).
BaseGrid<float> make_ramp_y() {
	BaseGrid<float> g(Matrix3(DX), Vector3(0.0f), N, N, N);
	for (idx_t ix = 0; ix < N; ++ix)
		for (idx_t iy = 0; iy < N; ++iy)
			for (idx_t iz = 0; iz < N; ++iz)
				g[iz + iy * N + ix * N * N] = static_cast<float>(iy);
	return g;
}

BaseGridView<float> host_view(const BaseGrid<float>& g, int grid_id) {
	BaseGridView<float> v{};
	v.data = g.data();
	v.origin = Vector3(0.0f);
	v.basis = Matrix3(DX);
	v.basis_inv = Matrix3(1.0f / DX);
	v.dimensions = g.dimensions();
	v.grid_id = grid_id;
	v.boundary_condition = static_cast<int>(GridBoundaryCondition::Dirichlet);
	return v;
}

/**
 * @brief Host stand-in for DeviceParticleTypes, holding the flat term table
 * @details Mirrors what DeviceParticleTypes::copy_from_host builds, so the
 *          offset/count layout under test is the same one the device path uses.
 */
struct HostTypes {
	std::vector<float> charge;
	std::vector<uint32_t> smd_freq;
	std::vector<int> pmf_offset;
	std::vector<int> pmf_count;
	std::vector<GridTerm> pmf_terms;
	std::vector<int> diffusion_grid_id;
	std::vector<ARBD::int3> force_grid_id;
	std::vector<Vector3> force_grid_scale;

	/// Append a type with the given charge and PMF terms; returns its type_id.
	int add(float q, const std::vector<GridTerm>& terms) {
		charge.push_back(q);
		smd_freq.push_back(0);
		pmf_offset.push_back(static_cast<int>(pmf_terms.size()));
		pmf_count.push_back(static_cast<int>(terms.size()));
		pmf_terms.insert(pmf_terms.end(), terms.begin(), terms.end());
		diffusion_grid_id.push_back(-1);
		force_grid_id.push_back(ARBD::int3(-1, -1, -1));
		force_grid_scale.push_back(Vector3(1.0f));
		return static_cast<int>(charge.size()) - 1;
	}

	ParticleTypeView view() const {
		ParticleTypeView v{};
		v.charge = charge.data();
		v.pmf_smd_freq = smd_freq.data();
		v.pmf_grid_offset = pmf_offset.data();
		v.pmf_grid_count = pmf_count.data();
		v.pmf_grid_terms = pmf_terms.data();
		v.diffusion_grid_id = diffusion_grid_id.data();
		v.force_grid_id = force_grid_id.data();
		v.force_grid_scale = force_grid_scale.data();
		return v;
	}
};

GridTerm term(int grid_id, float scale) {
	GridTerm t;
	t.grid_id = grid_id;
	t.scale = scale;
	return t;
}

constexpr int SCHEME_LINEAR = 0;

} // namespace

TEST_CASE("PMF grid table: per-type offset/count ranges", "[pmf][grids]") {
	const BaseGrid<float> ramp_x = make_ramp_x();
	const BaseGrid<float> ramp_y = make_ramp_y();
	const std::vector<BaseGridView<float>> grids{host_view(ramp_x, 0), host_view(ramp_y, 1)};

	const Vector3 pos(SAMPLE, SAMPLE, SAMPLE);

	HostTypes types;
	const int type_none = types.add(0.0f, {});					   // no gridFile
	const int type_one = types.add(0.0f, {term(0, 2.0f)});		   // one grid, scaled
	const int type_two = types.add(0.0f, {term(0, 2.0f), term(1, -3.0f)}); // two grids
	const ParticleTypeView view = types.view();

	SECTION("a type with no terms gets no grid force") {
		const Vector3 f = compute_position_dependent_force(
			pos, type_none, view, grids.data(), 0.0f, SCHEME_LINEAR);
		CHECK(f.x == Approx(0.0f));
		CHECK(f.y == Approx(0.0f));
		CHECK(f.z == Approx(0.0f));
	}

	SECTION("a single term applies its own scale, negating the gradient") {
		// F = -scale * grad V = -2 * (1,0,0)
		const Vector3 f = compute_position_dependent_force(
			pos, type_one, view, grids.data(), 0.0f, SCHEME_LINEAR);
		CHECK(f.x == Approx(-2.0f));
		CHECK(f.y == Approx(0.0f));
		CHECK(f.z == Approx(0.0f));
	}

	SECTION("both terms of a two-grid type contribute, each with its own scale") {
		// F = -(2 * (1,0,0) + (-3) * (0,1,0)) = (-2, +3, 0). The y component is
		// the whole point: with a single pmf_grid_id it would have been 0.
		const Vector3 f = compute_position_dependent_force(
			pos, type_two, view, grids.data(), 0.0f, SCHEME_LINEAR);
		CHECK(f.x == Approx(-2.0f));
		CHECK(f.y == Approx(3.0f));
		CHECK(f.z == Approx(0.0f));
	}

	SECTION("energy sums scale * V over the type's terms") {
		// V_x = V_y = 3.5 at the sample point, so 2*3.5 + (-3)*3.5 = -3.5.
		const Vector3 f = compute_position_dependent_force(
			pos, type_two, view, grids.data(), 0.0f, SCHEME_LINEAR, /*get_energy=*/true);
		CHECK(f.t == Approx(-3.5f));
	}

	SECTION("terms are not shared between types") {
		// type_one's range must stop before type_two's terms in the flat table.
		const Vector3 f_one = compute_position_dependent_force(
			pos, type_one, view, grids.data(), 0.0f, SCHEME_LINEAR);
		const Vector3 f_two = compute_position_dependent_force(
			pos, type_two, view, grids.data(), 0.0f, SCHEME_LINEAR);
		CHECK(f_one.y == Approx(0.0f));
		CHECK(f_two.y == Approx(3.0f));
		CHECK(types.pmf_offset[type_two] == types.pmf_offset[type_one] + 1);
	}

	SECTION("the electric field term is independent of the grid terms") {
		HostTypes charged;
		const int t = charged.add(1.5f, {term(0, 2.0f)});
		const Vector3 f = compute_position_dependent_force(
			pos, t, charged.view(), grids.data(), 4.0f, SCHEME_LINEAR);
		CHECK(f.x == Approx(-2.0f));
		CHECK(f.z == Approx(1.5f * 4.0f));
	}

	SECTION("an invalid term is skipped, not sampled") {
		HostTypes with_hole;
		const int t = with_hole.add(0.0f, {term(-1, 2.0f), term(1, -3.0f)});
		const Vector3 f = compute_position_dependent_force(
			pos, t, with_hole.view(), grids.data(), 0.0f, SCHEME_LINEAR);
		CHECK(f.x == Approx(0.0f));
		CHECK(f.y == Approx(3.0f));
	}
}

TEST_CASE("PMF grid table: cubic scheme walks the same term range", "[pmf][grids]") {
	// Catmull-Rom reproduces a linear field exactly too, so the same expected
	// values hold - this checks the scheme flag doesn't bypass the loop.
	const BaseGrid<float> ramp_x = make_ramp_x();
	const BaseGrid<float> ramp_y = make_ramp_y();
	const std::vector<BaseGridView<float>> grids{host_view(ramp_x, 0), host_view(ramp_y, 1)};

	HostTypes types;
	const int type_two = types.add(0.0f, {term(0, 2.0f), term(1, -3.0f)});

	const Vector3 f = compute_position_dependent_force(Vector3(SAMPLE, SAMPLE, SAMPLE),
													  type_two,
													  types.view(),
													  grids.data(),
													  0.0f,
													  /*scheme=*/1);
	CHECK(f.x == Approx(-2.0f));
	CHECK(f.y == Approx(3.0f));
}
