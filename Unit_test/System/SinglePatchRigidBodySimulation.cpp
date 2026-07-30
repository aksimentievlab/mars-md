#include "../catch_boiler.h"
#include "IO/DxIO.h"
#include "Objects/ParticleProperties.h"
#include "Objects/RigidBodyProperties.h"
#include "SimManager.h"
#include "System/PatchManager.h"
#include "System/SimSystem.h"
#include <filesystem>
#include <vector>

using namespace ARBD;
using namespace Tests;

namespace {

// Same quadratic-bowl construction as the Phase 4 tests (GridGrid.cpp,
// RigidBodyGridBatch.cpp): u(i,j,k) = -((i-c)^2+(j-c)^2+(k-c)^2), centered at
// index (3,3,3), dx=1, grid-local origin=0.
BaseGrid<float> make_bowl_grid() {
	constexpr idx_t N = 8;
	constexpr float DX = 1.0f;
	constexpr float C = 3.0f;
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

/**
 * @brief Phase 5 end-to-end wiring smoke test: SimSystem (with a rigid body
 * type + instance) -> SimManager::init() (constructs RigidBodyManager) ->
 * SimManager::run() (RB grid-grid/particle-RB/Langevin forces + DLM
 * integration alongside the particle path).
 *
 * This checks that the pipeline runs without crashing and that the rigid
 * body's state actually changes - it is not a physics-accuracy test (that's
 * Phase 6's job per todo.md).
 */
TEST_CASE("Single patch simulation with a rigid body runs end-to-end",
		  "[SimManager][SingleResource][integration][rigidbody]") {
	initialize_backend_once();
	if (!Tests::Global::backend_available) {
		WARN("No GPU backend available - skipping device cross-check");
		return;
	}

	std::vector<Resource> resources = {Resource(::Global::single_resource_id)};
	SimSystem sys(resources);

	const float box = 400.0f;
	sys.set_box_size(box, box, box);
	sys.set_periodicity(true, true, true);
	sys.set_temperature(300.0f);
	sys.set_cutoff(10.0f);
	sys.set_pairlist_cutoff(20.0f);
	sys.set_timestep(2e-5f);
	const int num_steps = 20;
	sys.set_num_steps(num_steps);
	// Larger than num_steps - no trajectory/energy output during this short
	// run, keeping the test focused on the force/integration wiring.
	sys.set_output_period(500.0f);
	sys.set_energy_output_period(500.0f);
	sys.set_output_name("single_patch_rb_smoke_test");
	sys.set_estimated_particles(16);
	sys.set_particle_integrator_type(IntegratorType::Brownian);
	sys.set_rigid_body_integrator_type(IntegratorType::Langevin);

	// One particle type, one particle placed inside the rigid body's
	// potential grid's support.
	ParticleType ptype("Ar");
	ptype.mass = 39.948f;
	ptype.diffusion = Vector3(1.0f, 1.0f, 1.0f);
	sys.add_particle_type(ptype);

	const Vector3 rb_position(0.0f, 0.0f, 0.0f);
	const Vector3 particle_position(3.3f, 3.2f, 3.15f);

	std::vector<ParticleIO> particles;
	ParticleIO p;
	p.id = 0;
	p.type_name = "Ar";
	p.position = particle_position;
	particles.push_back(p);

	// One rigid body type carrying a potential grid, loaded directly (no
	// ConfigParser - mirrors how SinglePatchSimulation.cpp builds SimSystem
	// via API calls, see IO/RigidBodyConfigParsing.cpp for the parser path).
	const auto tmp_dir = std::filesystem::temp_directory_path();
	const std::string grid_path = (tmp_dir / "arbd_test_rb_sim_bowl.dx").string();
	DXReader::write_grid(make_bowl_grid(), grid_path);
	GridKey grid_key = sys.get_grid_manager().add_dense_grid(grid_path);
	REQUIRE(grid_key.is_valid());

	// Mass/inertia/damping scaled for stability at a realistic MD timestep
	// under the legacy unit-conversion constants baked into
	// RBAddLangevinKernel/RBIntegrateDLMKernel (Constants.h:
	// impulse_to_momentum ~4.19e4, velocity_scale/langevin_damp_scale 1e4) -
	// mass=inertia=trans_damping=1.0 (fine for the Phase 4 unit tests, which
	// only ever made one add_langevin_forces() call at dt=1.0) is wildly
	// unstable once chained through many steps of the real DLM integrator at
	// dt=2e-5 ns: the damping term scales with trans_damping*langevin_damp_scale,
	// so trans_damping=1.0 against mass=1.0 is an enormous effective damping
	// coefficient relative to the timestep.
	RigidBodyType rbtype;
	rbtype.name = "A";
	rbtype.mass = 10000.0f;
	rbtype.inertia = Vector3(1.0e6f, 1.0e6f, 1.0e6f);
	rbtype.trans_damping = Vector3(1.0e-3f, 1.0e-3f, 1.0e-3f);
	rbtype.rot_damping = Vector3(1.0e-3f, 1.0e-3f, 1.0e-3f);
	rbtype.potential_grids = {GridTerm{grid_key.grid_id}};
	sys.add_rigid_body_type(rbtype);

	RigidBodyIO rb_io{};
	rb_io.id = 0;
	rb_io.type_id = 0;
	rb_io.position = rb_position;
	rb_io.orientation = Matrix3(1.0f);

	SimManager manager(sys);
	manager.set_initial_particles(particles);
	manager.set_initial_rigid_bodies({rb_io});

	REQUIRE_NOTHROW(manager.init());
	std::filesystem::remove(grid_path);

	REQUIRE(manager.get_rigid_body_manager() != nullptr);

	REQUIRE_NOTHROW(manager.run());

	HostRigidBodyData rb_result;
	manager.get_rigid_body_manager()->bodies().copy_to_host(rb_result, 1);
	REQUIRE(rb_result.size() == 1);

	REQUIRE(std::isfinite(rb_result.position[0].x));
	REQUIRE(std::isfinite(rb_result.position[0].y));
	REQUIRE(std::isfinite(rb_result.position[0].z));
	REQUIRE(std::isfinite(rb_result.momentum[0].x));
	REQUIRE(std::isfinite(rb_result.angular_momentum[0].x));

	// After num_steps of Langevin kicks + the potential-grid force, the rigid
	// body should have actually moved from its initial position.
	const Vector3 disp = rb_result.position[0] - rb_position;
	REQUIRE(disp.length2() > 0.0f);

	// The particle should also have moved under the rigid body's potential
	// force (in addition to Brownian diffusion).
	auto* patch_manager = sys.get_patch_manager();
	REQUIRE(patch_manager != nullptr);
	HostParticleData particle_result;
	patch_manager->get_patch(0).copy_particles_to_host(particle_result, 0, 1);
	REQUIRE(particle_result.size() == 1);
	REQUIRE(std::isfinite(particle_result.pos[0].x));
	REQUIRE(std::isfinite(particle_result.pos[0].y));
	REQUIRE(std::isfinite(particle_result.pos[0].z));
}
