/**
 * @file Pmf.cpp
 * @brief Tests for the flattened per-type PMF grid table consumed by
 * compute_position_dependent_force (Interactions/Nonbonded/Pmf.h).
 */

#include "Interactions/Nonbonded/Pmf.h"
#include "../catch_boiler.h"
#include "Objects/DeviceParticle.h"
#include "Types/BaseGrid.h"

using Catch::Approx;
using namespace MARS;

namespace {

constexpr idx_t N = 8;
constexpr float DX = 1.0f;
// Sampled well inside the grid so the i0+1 tap is in range.
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
	std::vector<MARS::int3> force_grid_id;
	std::vector<Vector3> force_grid_scale;

	/// Append a type with the given charge and PMF terms; returns its type_id.
	int add(float q, const std::vector<GridTerm>& terms) {
		charge.push_back(q);
		smd_freq.push_back(0);
		pmf_offset.push_back(static_cast<int>(pmf_terms.size()));
		pmf_count.push_back(static_cast<int>(terms.size()));
		pmf_terms.insert(pmf_terms.end(), terms.begin(), terms.end());
		diffusion_grid_id.push_back(-1);
		force_grid_id.push_back(MARS::int3(-1, -1, -1));
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

constexpr InterpolationOrder SCHEME_LINEAR = InterpolationOrder::Linear;

} // namespace

TEST_CASE("PMF grid table: per-type offset/count ranges", "[pmf][grids]") {
	const BaseGrid<float> ramp_x = make_ramp_x();
	const BaseGrid<float> ramp_y = make_ramp_y();
	const std::vector<BaseGridView<float>> grids{host_view(ramp_x, 0), host_view(ramp_y, 1)};

	const Vector3 pos(SAMPLE, SAMPLE, SAMPLE);

	HostTypes types;
	const int type_none = types.add(0.0f, {});							   // no gridFile
	const int type_one = types.add(0.0f, {term(0, 2.0f)});				   // one grid, scaled
	const int type_two = types.add(0.0f, {term(0, 2.0f), term(1, -3.0f)}); // two grids
	const ParticleTypeView view = types.view();

	SECTION("a type with no terms gets no grid force") {
		const Vector3 f = compute_position_dependent_force(pos,
														   type_none,
														   view,
														   grids.data(),
														   Vector3{0.0f, 0.0f, 0.0f},
														   SCHEME_LINEAR);
		CHECK(f.x == Approx(0.0f));
		CHECK(f.y == Approx(0.0f));
		CHECK(f.z == Approx(0.0f));
	}

	SECTION("a single term applies its own scale, negating the gradient") {
		// F = -scale * grad V = -2 * (1,0,0)
		const Vector3 f = compute_position_dependent_force(pos,
														   type_one,
														   view,
														   grids.data(),
														   Vector3{0.0f, 0.0f, 0.0f},
														   SCHEME_LINEAR);
		CHECK(f.x == Approx(-2.0f));
		CHECK(f.y == Approx(0.0f));
		CHECK(f.z == Approx(0.0f));
	}

	SECTION("both terms of a two-grid type contribute, each with its own scale") {
		// F = -(2 * (1,0,0) + (-3) * (0,1,0)) = (-2, +3, 0). The y component is
		// the whole point: with a single pmf_grid_id it would have been 0.
		const Vector3 f = compute_position_dependent_force(pos,
														   type_two,
														   view,
														   grids.data(),
														   Vector3{0.0f, 0.0f, 0.0f},
														   SCHEME_LINEAR);
		CHECK(f.x == Approx(-2.0f));
		CHECK(f.y == Approx(3.0f));
		CHECK(f.z == Approx(0.0f));
	}

	SECTION("energy sums scale * V over the type's terms") {
		// V_x = V_y = 3.5 at the sample point, so 2*3.5 + (-3)*3.5 = -3.5.
		const Vector3 f = compute_position_dependent_force(pos,
														   type_two,
														   view,
														   grids.data(),
														   Vector3{0.0f, 0.0f, 0.0f},
														   SCHEME_LINEAR,
														   /*get_energy=*/true);
		CHECK(f.t == Approx(-3.5f));
	}

	SECTION("terms are not shared between types") {
		// type_one's range must stop before type_two's terms in the flat table.
		const Vector3 f_one = compute_position_dependent_force(pos,
															   type_one,
															   view,
															   grids.data(),
															   Vector3{0.0f, 0.0f, 0.0f},
															   SCHEME_LINEAR);
		const Vector3 f_two = compute_position_dependent_force(pos,
															   type_two,
															   view,
															   grids.data(),
															   Vector3{0.0f, 0.0f, 0.0f},
															   SCHEME_LINEAR);
		CHECK(f_one.y == Approx(0.0f));
		CHECK(f_two.y == Approx(3.0f));
		CHECK(types.pmf_offset[type_two] == types.pmf_offset[type_one] + 1);
	}

	SECTION("the electric field term is independent of the grid terms") {
		HostTypes charged;
		const int t = charged.add(1.5f, {term(0, 2.0f)});
		const Vector3 f = compute_position_dependent_force(pos,
														   t,
														   charged.view(),
														   grids.data(),
														   Vector3{0.0f, 0.0f, 4.0f},
														   SCHEME_LINEAR);
		CHECK(f.x == Approx(-2.0f));
		CHECK(f.z == Approx(1.5f * 4.0f));
	}

	SECTION("an invalid term is skipped, not sampled") {
		HostTypes with_hole;
		const int t = with_hole.add(0.0f, {term(-1, 2.0f), term(1, -3.0f)});
		const Vector3 f = compute_position_dependent_force(pos,
														   t,
														   with_hole.view(),
														   grids.data(),
														   Vector3{0.0f, 0.0f, 0.0f},
														   SCHEME_LINEAR);
		CHECK(f.x == Approx(0.0f));
		CHECK(f.y == Approx(3.0f));
	}
}

TEST_CASE("PMF linear force is -dE/dx of the sampled energy off-grid", "[pmf][grids][gradient]") {
	// Curved, non-separable field: any stencil other than the trilinear derivative
	// disagrees with the energy slope here (a ramp would hide it).
	BaseGrid<float> g(Matrix3(DX), Vector3(0.0f), N, N, N);
	for (idx_t ix = 0; ix < N; ++ix)
		for (idx_t iy = 0; iy < N; ++iy)
			for (idx_t iz = 0; iz < N; ++iz) {
				const float x = static_cast<float>(ix) - 3.0f;
				g[iz + iy * N + ix * N * N] = x * x - 0.5f * float(iy * iz) + 0.25f * float(ix * iy);
			}
	const std::vector<BaseGridView<float>> grids{host_view(g, 0)};

	// Interior, first cell, and last cell (whose +1 tap is resolved by the BC).
	const Vector3 probes[] = {Vector3(3.3f, 2.6f, 4.2f),
							  Vector3(0.3f, 3.4f, 3.6f),
							  Vector3(7.4f, 3.4f, 3.6f)};
	const float h = 1e-2f; // keeps every probe +-h inside its cell
	const float scale = 2.0f;

	for (int bc : {0, 1, 2}) {
		GridTerm t = term(0, scale);
		t.boundary_condition = bc;
		HostTypes types;
		const int type_id = types.add(0.0f, {t});
		const ParticleTypeView view = types.view();

		auto energy = [&](const Vector3& p) {
			return compute_position_dependent_force(p,
													type_id,
													view,
													grids.data(),
													Vector3{0.0f, 0.0f, 0.0f},
													SCHEME_LINEAR,
													/*get_energy=*/true)
				.t;
		};

		for (const Vector3& p : probes) {
			const Vector3 f = compute_position_dependent_force(p,
															   type_id,
															   view,
															   grids.data(),
															   Vector3{0.0f, 0.0f, 0.0f},
															   SCHEME_LINEAR);
			const Vector3 ex(h, 0.0f, 0.0f), ey(0.0f, h, 0.0f), ez(0.0f, 0.0f, h);
			const float fd_x = -(energy(p + ex) - energy(p - ex)) / (2.0f * h);
			const float fd_y = -(energy(p + ey) - energy(p - ey)) / (2.0f * h);
			const float fd_z = -(energy(p + ez) - energy(p - ez)) / (2.0f * h);

			INFO("bc=" << bc << " probe=(" << p.x << "," << p.y << "," << p.z << ")");
			CHECK(f.x == Approx(fd_x).epsilon(1e-3f).margin(2e-3f));
			CHECK(f.y == Approx(fd_y).epsilon(1e-3f).margin(2e-3f));
			CHECK(f.z == Approx(fd_z).epsilon(1e-3f).margin(2e-3f));
		}
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
													   Vector3{0.0f, 0.0f, 0.0f},
													   InterpolationOrder::Cubic);
	CHECK(f.x == Approx(-2.0f));
	CHECK(f.y == Approx(3.0f));
}
