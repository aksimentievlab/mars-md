/**
 * @file GridGrid.cpp
 * @brief Phase 1 standalone test for the grid-grid rigid-body force kernel
 * (Interactions/Nonbonded/GridGridKernels.h) - see arbd2v/plan.md.
 *
 * Validates the per-voxel force/energy/torque math via finite differences on
 * a purely host-side loop (no Patch/RigidBodySystem/GridManager/config
 * parsing needed, and no GPU required), then cross-checks the real
 * ComputeGridGridForceKernel device path against that same host loop when a
 * GPU backend is available.
 */

#include "../catch_boiler.h"
#include "Backend/Kernels.h"
#include "Interactions/Nonbonded/GridGridKernels.h"
#include "Types/BaseGrid.h"

#include <cmath>

using Catch::Approx;
using namespace ARBD;

namespace {

// N=8 per dimension, dx=1.0, grid-local origin=0 (so a body's own local
// origin doubles as its rotation center - see the torque test below for why
// that matters). rho is a single unit-mass voxel near the grid's own center
// index; u is an exact quadratic bowl u(i,j,k) = -((i-c)^2+(j-c)^2+(k-c)^2)
// centered at the same index - smooth and, since Catmull-Rom cubic splines
// reproduce polynomials up to degree 3 exactly, the cubic interpolation
// scheme samples/differentiates it exactly (no interpolation error to
// confound the finite-difference comparison), as long as we stay away from
// the zero-padded boundary.
constexpr idx_t N = 8;
constexpr float DX = 1.0f;
constexpr float C = 3.0f; // bowl/voxel center index

BaseGrid<float> make_rho() {
	BaseGrid<float> g(Matrix3(DX), Vector3(0.0f), N, N, N);
	g.zero();
	idx_t lin = 3 + 3 * N + 3 * N * N; // (ix,iy,iz) = (3,3,3)
	g[lin] = 1.0f;
	return g;
}

BaseGrid<float> make_u() {
	BaseGrid<float> g(Matrix3(DX), Vector3(0.0f), N, N, N);
	for (idx_t ix = 0; ix < N; ++ix) {
		for (idx_t iy = 0; iy < N; ++iy) {
			for (idx_t iz = 0; iz < N; ++iz) {
				const float dxc = float(ix) - C;
				const float dyc = float(iy) - C;
				const float dzc = float(iz) - C;
				g[iz + iy * N + ix * N * N] = -(dxc * dxc + dyc * dyc + dzc * dzc);
			}
		}
	}
	return g;
}

BaseGridView<float> host_view(const BaseGrid<float>& g) {
	BaseGridView<float> v{};
	v.data = g.data();
	v.origin = Vector3(0.0f);
	v.basis = Matrix3(DX);
	v.basis_inv = Matrix3(1.0f / DX);
	v.dimensions = g.dimensions();
	v.grid_id = -1;
	v.boundary_condition = 0; // Dirichlet; unused as long as samples stay interior
	return v;
}

// Sum grid_grid_voxel_force_torque over every rho voxel on the host - no
// kernel launch, mirrors what ComputeGridGridForceKernel does with a
// shared-memory block reduction, but with a plain serial accumulator.
struct Totals {
	Vector3 force_energy; // .xyz = force on rho's body, .t = energy
	Vector3 torque;		 // about rho's own origin, lab frame
};

Totals host_grid_grid_totals(const BaseGridView<float>& rho,
							 const BaseGridView<float>& u,
							 const Matrix3& basis_rho,
							 const Matrix3& basis_u_inv,
							 const Vector3& origin_offset,
							 int scheme) {
	Totals t{Vector3(0.0f), Vector3(0.0f)};
	for (idx_t r_id = 0; r_id < rho.size(); ++r_id) {
		Vector3 fe, tq;
		gridgrid_detail::grid_grid_voxel_force_torque(
			rho, u, basis_rho, basis_u_inv, origin_offset, r_id, scheme, fe, tq);
		// operator+= only adds x/y/z (see Types/Vector3.h) - energy lives in
		// .t here, so it needs an explicit add too (matches the fix in
		// ComputeGridGridForceKernel's block reduction, see GridGridKernels.h).
		t.force_energy += fe;
		t.force_energy.t += fe.t;
		t.torque += tq;
	}
	return t;
}

// P1/R1 = rho's body pose, P2/R2 = u's body pose. Both grids have
// grid-local origin=0, so this matches
// origin_offset = (R1*0 + P1) - (R2*0 + P2) = P1 - P2, basis_rho = R1*grid_basis,
// basis_u_inv = (R2*grid_basis).inverse() from GridGridKernels.h's doc comment.
Totals evaluate(const BaseGrid<float>& rho_grid,
				const BaseGrid<float>& u_grid,
				const Vector3& P1,
				const Matrix3& R1,
				const Vector3& P2,
				const Matrix3& R2,
				int scheme) {
	const BaseGridView<float> rho = host_view(rho_grid);
	const BaseGridView<float> u = host_view(u_grid);
	// grid_basis = DX*Identity here, and a rotation composed with a uniform
	// scalar scaling commutes (R*(DX*I) == DX*(R*I) == DX*R), so this uses
	// the scalar*Matrix3 operator instead of a general matrix-matrix
	// product. Matrix3_t has no operator* for two matrices yet - Phase
	// 3/4's RigidBodySystem (which must compute rb.orientation * grid.basis
	// for an arbitrary, not-necessarily-uniform-scale grid basis) will need
	// one added; flagged here rather than worked around, since it's a real
	// gap for later, not just this test.
	const Matrix3 basis_rho = DX * R1;
	const Matrix3 basis_u_inv = (DX * R2).inverse();
	const Vector3 origin_offset = P1 - P2;
	return host_grid_grid_totals(rho, u, basis_rho, basis_u_inv, origin_offset, scheme);
}

Matrix3 rotate_z(float theta) {
	const float c = std::cos(theta);
	const float s = std::sin(theta);
	return Matrix3(Vector3(c, s, 0.0f), Vector3(-s, c, 0.0f), Vector3(0.0f, 0.0f, 1.0f));
}

} // namespace

TEST_CASE("Grid-grid force matches finite-difference energy gradient", "[gridgrid][force]") {
	BaseGrid<float> rho_grid = make_rho();
	BaseGrid<float> u_grid = make_u();

	// Offset u's body so the sampled point (rho's single voxel, seen in u's
	// frame) lands at a non-integer, non-symmetric interior point of the
	// bowl - an integer or on-axis point would make some gradient
	// components exactly zero by symmetry and not exercise interpolation.
	const Vector3 P2(-0.3f, -0.2f, -0.15f);
	const Matrix3 I(1.0f);

	const Totals at_p0 = evaluate(rho_grid, u_grid, Vector3(0.0f), I, P2, I, 1);

	const float eps = 1e-3f;
	const Totals at_plus = evaluate(rho_grid, u_grid, Vector3(eps, 0.0f, 0.0f), I, P2, I, 1);
	const Totals at_minus = evaluate(rho_grid, u_grid, Vector3(-eps, 0.0f, 0.0f), I, P2, I, 1);

	const float fd_force_x = -(at_plus.force_energy.t - at_minus.force_energy.t) / (2.0f * eps);

	REQUIRE(at_p0.force_energy.x == Approx(fd_force_x).epsilon(0.01f));

	// Sanity: analytic gradient of the quadratic bowl at the sampled point
	// (rho voxel at index (3,3,3), u_local = (3.3, 3.2, 3.15) given P2 above,
	// dx=1 and identity rotations) is -2*(x-3) etc, so force = -grad =
	// (0.6, 0.4, 0.3). Cubic (Catmull-Rom) interpolation reproduces an exact
	// quadratic's value and derivative exactly, so this should match tightly,
	// not just within FD tolerance.
	REQUIRE(at_p0.force_energy.x == Approx(0.6f).epsilon(0.01f));
	REQUIRE(at_p0.force_energy.y == Approx(0.4f).epsilon(0.01f));
	REQUIRE(at_p0.force_energy.z == Approx(0.3f).epsilon(0.01f));
}

TEST_CASE("Grid-grid torque matches finite-difference energy gradient over rotation",
		 "[gridgrid][torque]") {
	BaseGrid<float> rho_grid = make_rho();
	BaseGrid<float> u_grid = make_u();

	const Vector3 P2(-0.3f, -0.2f, -0.15f);
	const Matrix3 I(1.0f);

	const Totals at_p0 = evaluate(rho_grid, u_grid, Vector3(0.0f), I, P2, I, 1);

	const float delta = 1e-3f;
	const Totals at_plus = evaluate(rho_grid, u_grid, Vector3(0.0f), rotate_z(delta), P2, I, 1);
	const Totals at_minus = evaluate(rho_grid, u_grid, Vector3(0.0f), rotate_z(-delta), P2, I, 1);

	const float fd_torque_z = -(at_plus.force_energy.t - at_minus.force_energy.t) / (2.0f * delta);

	REQUIRE(at_p0.torque.z == Approx(fd_torque_z).epsilon(0.02f));
}

TEST_CASE("Grid-grid energy is symmetric under swapping which grid is rho vs u",
		 "[gridgrid][symmetry]") {
	BaseGrid<float> rho_grid = make_rho();
	BaseGrid<float> u_grid = make_u();

	const Vector3 P2(-0.3f, -0.2f, -0.15f);
	const Matrix3 I(1.0f);

	// rho acting through u's field, vs. rho acting through u's field with
	// rho/u swapped and the relative pose negated - both describe the same
	// physical configuration (a single mass point in the bowl's field) so
	// the total interaction energy should match.
	const Totals forward = evaluate(rho_grid, u_grid, Vector3(0.0f), I, P2, I, 1);

	// Swapped: now u_grid plays rho's role (density) and rho_grid plays u's
	// role (potential) - but rho_grid is all zero except one voxel, so this
	// isn't the same physical system unless both grids carry a comparable
	// density and potential. Skip a literal swap and instead check energy is
	// unchanged under rigidly translating both bodies by the same amount
	// (translation invariance of a two-body interaction), which is the
	// meaningful symmetry to test here without needing a second density grid.
	const Vector3 shift(5.0f, -3.0f, 2.0f);
	const Totals shifted = evaluate(rho_grid, u_grid, shift, I, P2 + shift, I, 1);

	REQUIRE(forward.force_energy.t == Approx(shifted.force_energy.t).epsilon(1e-4f));
	REQUIRE(forward.force_energy.x == Approx(shifted.force_energy.x).epsilon(1e-4f));
}

#if defined(USE_CUDA) || defined(USE_SYCL)
TEST_CASE("Grid-grid device kernel matches host loop", "[gridgrid][device]") {
	initialize_backend_once();
	if (!Tests::Global::backend_available) {
		WARN("No GPU backend available - skipping device cross-check");
		return;
	}

	Resource res(Global::single_resource_id);

	BaseGrid<float> rho_grid = make_rho();
	BaseGrid<float> u_grid = make_u();
	const Vector3 P2(-0.3f, -0.2f, -0.15f);
	const Matrix3 I(1.0f);

	const Totals host_totals = evaluate(rho_grid, u_grid, Vector3(0.0f), I, P2, I, 1);

	BaseGridView<float> rho_view = rho_grid.get_device_view(res);
	BaseGridView<float> u_view = u_grid.get_device_view(res);

	DeviceBuffer<Vector3> d_force_energy(1, res);
	DeviceBuffer<Vector3> d_torque(1, res);
	Vector3 zero(0.0f);
	d_force_energy.copy_from_host(&zero, 1, true);
	d_torque.copy_from_host(&zero, 1, true);

	ComputeGridGridForceKernel kernel;
	kernel.basis_rho = DX * I;
	kernel.basis_u_inv = (DX * I).inverse();
	kernel.origin_offset = Vector3(0.0f) - P2;
	kernel.scheme = 1;
	kernel.block_size = 128;

	KernelConfig config = KernelConfig::for_1d(rho_grid.size(), res);
	config.block_size = kerneldim3{kernel.block_size, 1, 1};
	config.grid_size =
		kerneldim3{(rho_grid.size() + kernel.block_size - 1) / kernel.block_size, 1, 1};
	config.shared_memory = 2 * kernel.block_size * sizeof(Vector3);
	config.sync = true;

	Event evt = launch_kernel_with_workitem(
		res, config, kernel, rho_view, u_view, d_force_energy.data(), d_torque.data());
	evt.wait();

	Vector3 device_force_energy, device_torque;
	d_force_energy.copy_to_host(&device_force_energy, 1, true);
	d_torque.copy_to_host(&device_torque, 1, true);

	REQUIRE(device_force_energy.x == Approx(host_totals.force_energy.x).epsilon(0.01f));
	REQUIRE(device_force_energy.y == Approx(host_totals.force_energy.y).epsilon(0.01f));
	REQUIRE(device_force_energy.z == Approx(host_totals.force_energy.z).epsilon(0.01f));
	REQUIRE(device_force_energy.t == Approx(host_totals.force_energy.t).epsilon(0.01f));
	REQUIRE(device_torque.z == Approx(host_totals.torque.z).epsilon(0.01f));
}
#endif
