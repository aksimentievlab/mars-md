/**
 * @file RigidBodyParticleGridBatch.cpp
 * @brief Phase 4.3 test for batched particle-RB grid dispatch
 * (Interactions/Nonbonded/RigidBodyParticleGridBatch.h): a single particle
 * sampling a single RB's potential grid, checked against closed-form
 * analytic values (the same quadratic-bowl construction used by
 * GridGrid.cpp/RigidBodyGridBatch.cpp, so cubic interpolation reproduces it
 * exactly).
 */

#include "../catch_boiler.h"
#include "Backend/Kernels.h"
#include "Interactions/Nonbonded/RigidBodyParticleGridBatch.h"
#include "Objects/DeviceParticleManager.h"
#include "Objects/DeviceRigidBodyManager.h"
#include "Backend/Events.h"
#include "Types/BaseGrid.h"
#include <utility>

using Catch::Approx;
using namespace ARBD;

#if defined(USE_CUDA) || defined(USE_SYCL)

namespace {

// Quadratic bowl u(i,j,k) = -((i-c)^2+(j-c)^2+(k-c)^2), centered at index
// (3,3,3), dx=1, grid-local origin=0 - same construction as
// GridGrid.cpp/RigidBodyGridBatch.cpp.
constexpr idx_t N = 8;
constexpr float DX = 1.0f;
constexpr float C = 3.0f;

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

} // namespace

TEST_CASE("RBParticleGridBatch: single particle sampling a single RB's potential "
		  "grid matches the analytic bowl gradient",
		  "[rigidbody][particlegrid][batch]") {
	initialize_backend_once();
	if (!Tests::Global::backend_available) {
		WARN("No GPU backend available - skipping device cross-check");
		return;
	}
	Resource res(Global::single_resource_id);

	BaseGrid<float> u_grid = make_u();
	BaseGridView<float> u_view = u_grid.get_device_view(res);

	// RB at the world origin, identity orientation, so world position ==
	// grid-local position - the particle sampled at (3.3, 3.2, 3.15) sits at
	// a fractional offset (0.3, 0.2, 0.15) from the bowl's center index.
	DeviceRigidBody bodies(1, res);
	HostRigidBodyData rb_host;
	RigidBodyIO rb{};
	rb.id = 0;
	rb.type_id = 0;
	rb.position = Vector3(0.0f, 0.0f, 0.0f);
	rb.orientation = Matrix3(1.0f);
	rb_host.push_back(rb);
	bodies.copy_from_host(rb_host, 1);

	DeviceParticle particles(1, res);
	HostParticleData particle_host;
	particle_host.resize(1);
	particle_host.global_id[0] = 0;
	particle_host.type_id[0] = 0;
	particle_host.pos[0] = Vector3(3.3f, 3.2f, 3.15f);
	particle_host.mom[0] = Vector3(0.0f, 0.0f, 0.0f);
	particles.copy_from_host(particle_host, 1);
	particles.clear_forces();

	DeviceBuffer<BaseGridView<float>> grid_views(1, res);
	grid_views.copy_from_host(&u_view, 1, true);

	int candidate_rb_id = 0;
	int candidate_grid_id = 0;
	float candidate_scale = 1.0f;
	DeviceBuffer<int> rb_ids(1, res);
	DeviceBuffer<int> grid_ids(1, res);
	DeviceBuffer<float> scales(1, res);
	rb_ids.copy_from_host(&candidate_rb_id, 1, true);
	grid_ids.copy_from_host(&candidate_grid_id, 1, true);
	scales.copy_from_host(&candidate_scale, 1, true);

	DeviceBuffer<RBParticleGridWork> work(1, res);

	RBParticleGridBuildKernel build{std::as_const(bodies).view(),
									rb_ids.data(),
									grid_ids.data(),
									scales.data(),
									/*num_candidates=*/1,
									grid_views.data(),
									/*scheme=*/1,
									work.data()};
	KernelConfig build_config = KernelConfig::for_1d(1, res);
	build_config.sync = true;
	launch_kernel(res, build_config, build).wait();

	const idx_t threads_per_block = 128;
	const idx_t blocks_per_candidate = 1; // ceil(1 particle / 128)
	RBParticleGridForceKernel force{bodies.view(),
									particles.view(),
									work.data(),
									grid_views.data(),
									/*num_particles=*/1,
									blocks_per_candidate,
									threads_per_block};
	KernelConfig force_config;
	force_config.dim = 1;
	force_config.block_size = kerneldim3{threads_per_block, 1, 1};
	force_config.grid_size = kerneldim3{blocks_per_candidate, 1, 1};
	force_config.problem_size = kerneldim3{blocks_per_candidate * threads_per_block, 1, 1};
	force_config.shared_memory = 2 * threads_per_block * sizeof(Vector3);
	force_config.sync = true;
	Event evt=launch_kernel_with_workitem(res, force_config, force);
	evt.wait();

	HostParticleData particle_result;
	particles.copy_to_host(particle_result, 1, /*need_energy=*/true);
	HostRigidBodyData rb_result;
	bodies.copy_to_host(rb_result, 1);

	// Analytic: force = -grad = 2*(0.3, 0.2, 0.15) = (0.6, 0.4, 0.3);
	// energy = -(0.3^2+0.2^2+0.15^2) = -0.1525.
	REQUIRE(particle_result.force[0].x == Approx(0.6f).epsilon(0.01f));
	REQUIRE(particle_result.force[0].y == Approx(0.4f).epsilon(0.01f));
	REQUIRE(particle_result.force[0].z == Approx(0.3f).epsilon(0.01f));
	REQUIRE(particle_result.energy[0] == Approx(-0.1525f).epsilon(0.01f));

	// RB reaction: force = -(0.6,0.4,0.3); torque = p x (-force) about the
	// grid's lab-frame origin (== rb.position here, since grid.origin==0).
	REQUIRE(rb_result.force[0].x == Approx(-0.6f).epsilon(0.01f));
	REQUIRE(rb_result.force[0].y == Approx(-0.4f).epsilon(0.01f));
	REQUIRE(rb_result.force[0].z == Approx(-0.3f).epsilon(0.01f));
	REQUIRE(rb_result.torque[0].x == Approx(0.30f).epsilon(0.02f));
	REQUIRE(rb_result.torque[0].y == Approx(-0.90f).epsilon(0.02f));
	REQUIRE(rb_result.torque[0].z == Approx(0.60f).epsilon(0.02f));
}

#endif // USE_CUDA || USE_SYCL
