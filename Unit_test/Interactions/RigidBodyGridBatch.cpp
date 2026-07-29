/**
 * @file RigidBodyGridBatch.cpp
 * @brief Phase 4.1 test for batched RB grid-grid dispatch
 * (Interactions/Nonbonded/RigidBodyGridBatch.h): cull -> prefix sum ->
 * batched force, cross-checked against Phase 1's per-pair device kernel
 * (GridGridKernels.h) on the identical two-body configuration used by
 * Unit_test/Interactions/GridGrid.cpp, plus a deliberately undersized
 * worklist to exercise the overflow flag.
 */

#include "Interactions/Nonbonded/RigidBodyGridBatch.h"
#include "../catch_boiler.h"
#include "Backend/Kernels.h"
#include "Objects/DeviceRigidBodyManager.h"
#include "Types/BaseGrid.h"

using Catch::Approx;
using namespace ARBD;

#if defined(USE_CUDA) || defined(USE_SYCL)

namespace {

// Same construction as GridGrid.cpp's make_rho/make_u: rho is a single unit
// voxel at index (3,3,3); u is a quadratic bowl centered at the same index -
// see that file for why (cubic interpolation reproduces it exactly).
constexpr idx_t N = 8;
constexpr float DX = 1.0f;
constexpr float C = 3.0f;

BaseGrid<float> make_rho() {
	BaseGrid<float> g(Matrix3(DX), Vector3(0.0f), N, N, N);
	g.zero();
	g[3 + 3 * N + 3 * N * N] = 1.0f;
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

// rb0 owns the density grid (rho), rb1 owns the potential grid (u) - a true
// RB-RB pair (RigidBodyGridPair::is_pmf == false), so rb1 receives the
// Newton's-third-law reaction.
HostRigidBodyData two_body_config(const Vector3& p0, const Vector3& p1) {
	HostRigidBodyData host;
	for (int i = 0; i < 2; ++i) {
		RigidBodyIO rb{};
		rb.id = i;
		rb.type_id = i;
		rb.position = (i == 0) ? p0 : p1;
		rb.orientation = Matrix3(1.0f);
		host.push_back(rb);
	}
	return host;
}

} // namespace

TEST_CASE("RBGridBatch: batched kernel matches Phase 1's per-pair kernel for the action body",
		  "[rigidbody][gridgrid][batch]") {
	initialize_backend_once();
	if (!Tests::Global::backend_available) {
		WARN("No GPU backend available - skipping device cross-check");
		return;
	}
	Resource res(Global::single_resource_id);

	BaseGrid<float> rho_grid = make_rho();
	BaseGrid<float> u_grid = make_u();
	const Vector3 P0(0.0f, 0.0f, 0.0f);
	const Vector3 P1(-0.3f, -0.2f, -0.15f);

	// --- Phase 1 reference: single-pair device kernel ---
	BaseGridView<float> rho_view = rho_grid.get_device_view(res);
	BaseGridView<float> u_view = u_grid.get_device_view(res);
	rho_view.grid_id = 0;
	u_view.grid_id = 1;

	DeviceBuffer<Vector3> ref_force_energy(1, res);
	DeviceBuffer<Vector3> ref_torque(1, res);
	Vector3 zero3(0.0f);
	ref_force_energy.copy_from_host(&zero3, 1, true);
	ref_torque.copy_from_host(&zero3, 1, true);

	ComputeGridGridForceKernel ref_kernel;
	ref_kernel.basis_rho = DX * Matrix3(1.0f);
	ref_kernel.basis_u_inv = (DX * Matrix3(1.0f)).inverse();
	ref_kernel.origin_offset = P0 - P1;
	ref_kernel.scheme = 1;
	ref_kernel.block_size = 128;

	KernelConfig ref_config = KernelConfig::for_1d(rho_grid.size(), res);
	ref_config.block_size = kerneldim3{ref_kernel.block_size, 1, 1};
	ref_config.grid_size =
		kerneldim3{(rho_grid.size() + ref_kernel.block_size - 1) / ref_kernel.block_size, 1, 1};
	ref_config.shared_memory = 2 * ref_kernel.block_size * sizeof(Vector3);
	ref_config.sync = true;
	launch_kernel_with_workitem(res,
								ref_config,
								ref_kernel,
								rho_view,
								u_view,
								ref_force_energy.data(),
								ref_torque.data())
		.wait();

	Vector3 ref_fe, ref_tq;
	ref_force_energy.copy_to_host(&ref_fe, 1, true);
	ref_torque.copy_to_host(&ref_tq, 1, true);

	// --- Batched pipeline: two RB instances, one grid-grid pair ---
	DeviceRigidBody bodies(2, res);
	HostRigidBodyData host = two_body_config(P0, P1);
	bodies.copy_from_host(host, 2);

	DeviceBuffer<BaseGridView<float>> grid_views(2, res);
	BaseGridView<float> views[2] = {rho_view, u_view};
	grid_views.copy_from_host(views, 2, true);

	RigidBodyGridPair pair;
	pair.type_i = 0;
	pair.type_j = 1;
	pair.grid_id_rho = 0;
	pair.grid_id_u = 1;
	pair.is_pmf = false;
	pair.update_period = 1;
	DeviceBuffer<RigidBodyGridPair> grid_pairs(1, res);
	grid_pairs.copy_from_host(&pair, 1, true);

	ARBD::int2 candidate(0, 1);
	int candidate_idx = 0;
	DeviceBuffer<ARBD::int2> candidates(1, res);
	DeviceBuffer<int> candidates_pair_idx(1, res);
	candidates.copy_from_host(&candidate, 1, true);
	candidates_pair_idx.copy_from_host(&candidate_idx, 1, true);

	const idx_t threads_per_block = 128;
	const idx_t num_blocks = (rho_grid.size() + threads_per_block - 1) / threads_per_block;

	DeviceBuffer<RBGridWork> work(1, res);
	DeviceBuffer<unsigned int> work_count(1, res);
	DeviceBuffer<unsigned int> total_blocks(1, res);
	DeviceBuffer<unsigned int> overflow(1, res);
	const unsigned int zero_u = 0;
	work_count.copy_from_host(&zero_u, 1, true);
	overflow.copy_from_host(&zero_u, 1, true);

	const DeviceRigidBody& cbodies = bodies;
	RBGridCullKernel cull{cbodies.view(),
						  candidates.data(),
						  candidates_pair_idx.data(),
						  grid_pairs.data(),
						  grid_views.data(),
						  1,
						  1000.0f * 1000.0f, // cutoff: generous, both bodies must pass
						  /*step=*/0,
						  threads_per_block,
						  /*scheme=*/1,
						  work.data(),
						  work_count.data(),
						  /*capacity=*/1,
						  overflow.data()};
	KernelConfig cull_config = KernelConfig::for_1d(1, res);
	cull_config.sync = true;
	launch_kernel(res, cull_config, cull).wait();

	{
		unsigned int wc = 0;
		work_count.copy_to_host(&wc, 1, true);
		INFO("work_count after cull = " << wc);
		REQUIRE(wc == 1);
		RBGridWork w{};
		work.copy_to_host(&w, 1, true);
		INFO("work[0]: rho_grid_id=" << w.rho_grid_id << " u_grid_id=" << w.u_grid_id
									 << " rb_i=" << w.rb_i << " rb_j=" << w.rb_j
									 << " num_blocks=" << w.num_blocks);
		REQUIRE(w.rho_grid_id == 0);
		REQUIRE(w.u_grid_id == 1);
		REQUIRE(w.rb_i == 0);
		REQUIRE(w.rb_j == 1);
		REQUIRE(w.num_blocks == num_blocks);
	}

	RBGridPrefixSumKernel scan{work.data(), work_count.data(), total_blocks.data()};
	KernelConfig scan_config = KernelConfig::for_1d(1, res);
	scan_config.sync = true;
	launch_kernel(res, scan_config, scan).wait();

	{
		unsigned int tb = 0;
		total_blocks.copy_to_host(&tb, 1, true);
		INFO("total_blocks after prefix sum = " << tb);
		REQUIRE(tb == static_cast<unsigned int>(num_blocks));
	}

	RBGridBatchedForceKernel force{bodies.view(),
								   work.data(),
								   work_count.data(),
								   total_blocks.data(),
								   grid_views.data(),
								   threads_per_block};
	KernelConfig force_config;
	force_config.dim = 1;
	force_config.block_size = kerneldim3{threads_per_block, 1, 1};
	force_config.grid_size = kerneldim3{num_blocks, 1, 1};
	force_config.problem_size = kerneldim3{num_blocks * threads_per_block, 1, 1};
	force_config.shared_memory = 2 * threads_per_block * sizeof(Vector3);
	force_config.sync = true;
	launch_kernel_with_workitem(res, force_config, force).wait();

	unsigned int overflow_flag = 0;
	overflow.copy_to_host(&overflow_flag, 1, true);
	REQUIRE(overflow_flag == 0);

	HostRigidBodyData result;
	bodies.copy_to_host(result, 2);

	// rb0 (action body): must match Phase 1's per-pair kernel tightly (same
	// per-voxel formula, just reached via the batched worklist).
	REQUIRE(result.force[0].x == Approx(ref_fe.x).epsilon(1e-4f));
	REQUIRE(result.force[0].y == Approx(ref_fe.y).epsilon(1e-4f));
	REQUIRE(result.force[0].z == Approx(ref_fe.z).epsilon(1e-4f));
	REQUIRE(result.torque[0].x == Approx(ref_tq.x).epsilon(1e-4f));
	REQUIRE(result.torque[0].y == Approx(ref_tq.y).epsilon(1e-4f));
	REQUIRE(result.torque[0].z == Approx(ref_tq.z).epsilon(1e-4f));

	// rb1 (reaction body): Newton's third law force, and the standard
	// rigid-body torque transfer to a different reference point (see
	// RigidBodyGridBatch.h's RBGridBatchedForceKernel doc).
	const Vector3 origin_offset = P0 - P1;
	const Vector3 expected_torque1 = -(result.torque[0] + origin_offset.cross(result.force[0]));
	REQUIRE(result.force[1].x == Approx(-result.force[0].x).epsilon(1e-4f));
	REQUIRE(result.force[1].y == Approx(-result.force[0].y).epsilon(1e-4f));
	REQUIRE(result.force[1].z == Approx(-result.force[0].z).epsilon(1e-4f));
	REQUIRE(result.torque[1].x == Approx(expected_torque1.x).epsilon(1e-4f));
	REQUIRE(result.torque[1].y == Approx(expected_torque1.y).epsilon(1e-4f));
	REQUIRE(result.torque[1].z == Approx(expected_torque1.z).epsilon(1e-4f));
}

TEST_CASE("RBGridBatch: cull kernel sets the overflow flag on an undersized worklist",
		  "[rigidbody][gridgrid][batch][overflow]") {
	initialize_backend_once();
	if (!Tests::Global::backend_available) {
		WARN("No GPU backend available - skipping device cross-check");
		return;
	}
	Resource res(Global::single_resource_id);

	BaseGrid<float> rho_grid = make_rho();
	BaseGrid<float> u_grid = make_u();
	BaseGridView<float> rho_view = rho_grid.get_device_view(res);
	BaseGridView<float> u_view = u_grid.get_device_view(res);

	DeviceRigidBody bodies(2, res);
	HostRigidBodyData host = two_body_config(Vector3(0.0f), Vector3(-0.3f, -0.2f, -0.15f));
	bodies.copy_from_host(host, 2);

	DeviceBuffer<BaseGridView<float>> grid_views(2, res);
	BaseGridView<float> views[2] = {rho_view, u_view};
	grid_views.copy_from_host(views, 2, true);

	RigidBodyGridPair pair;
	pair.grid_id_rho = 0;
	pair.grid_id_u = 1;
	pair.is_pmf = false;
	pair.update_period = 1;
	DeviceBuffer<RigidBodyGridPair> grid_pairs(1, res);
	grid_pairs.copy_from_host(&pair, 1, true);

	ARBD::int2 candidate(0, 1);
	int candidate_idx = 0;
	DeviceBuffer<ARBD::int2> candidates(1, res);
	DeviceBuffer<int> candidates_pair_idx(1, res);
	candidates.copy_from_host(&candidate, 1, true);
	candidates_pair_idx.copy_from_host(&candidate_idx, 1, true);

	// Capacity 0 while one real candidate survives cull - must overflow, not
	// silently drop the interaction.
	DeviceBuffer<RBGridWork> work(1, res);
	DeviceBuffer<unsigned int> work_count(1, res);
	DeviceBuffer<unsigned int> overflow(1, res);
	const unsigned int zero_u = 0;
	work_count.copy_from_host(&zero_u, 1, true);
	overflow.copy_from_host(&zero_u, 1, true);

	const DeviceRigidBody& cbodies = bodies;
	RBGridCullKernel cull{cbodies.view(),
						  candidates.data(),
						  candidates_pair_idx.data(),
						  grid_pairs.data(),
						  grid_views.data(),
						  1,
						  1000.0f * 1000.0f,
						  /*step=*/0,
						  /*threads_per_block=*/128,
						  /*scheme=*/1,
						  work.data(),
						  work_count.data(),
						  /*capacity=*/0,
						  overflow.data()};
	KernelConfig cull_config = KernelConfig::for_1d(1, res);
	cull_config.sync = true;
	launch_kernel(res, cull_config, cull).wait();

	unsigned int overflow_flag = 0;
	overflow.copy_to_host(&overflow_flag, 1, true);
	REQUIRE(overflow_flag > 0);
}

#endif // USE_CUDA || USE_SYCL
