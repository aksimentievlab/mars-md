/**
 * @file RigidBodyManager.cpp
 * @brief Tests for Phase 4's orchestration skeleton (Objects/RigidBodyManager.h):
 *        construction/initialize(), Langevin forces, and DLM integration,
 *        batched across RB instances via the SoA views from Phase 2.
 */

#include "../catch_boiler.h"
#include "Constants.h"
#include "Backend/Events.h"
#include "IO/DxIO.h"
#include "Objects/DeviceParticleManager.h"
#include "Objects/Grid.h"
#include "Objects/RigidBodyManager.h"
#include "Objects/RigidBodyProperties.h"
#include <filesystem>

using namespace ARBD;

namespace {

std::vector<RigidBodyType> single_type(float mass, Vector3 inertia) {
	std::vector<RigidBodyType> types(1);
	types[0].name = "A";
	types[0].id = 0;
	types[0].mass = mass;
	types[0].inertia = inertia;
	types[0].trans_damping = Vector3(1.0f, 1.0f, 1.0f);
	types[0].rot_damping = Vector3(1.0f, 1.0f, 1.0f);
	return types;
}

HostRigidBodyData single_body_at_rest(Vector3 position) {
	HostRigidBodyData host;
	RigidBodyIO rb{};
	rb.id = 0;
	rb.type_id = 0;
	rb.position = position;
	rb.orientation = Matrix3(1.0f);
	rb.momentum = Vector3(0.0f, 0.0f, 0.0f);
	rb.angular_momentum = Vector3(0.0f, 0.0f, 0.0f);
	rb.force = Vector3(0.0f, 0.0f, 0.0f);
	rb.torque = Vector3(0.0f, 0.0f, 0.0f);
	host.push_back(rb);
	return host;
}

// No grids configured in these tests, so the format lookup should never be
// called; fail loudly if Phase 3's pair matching somehow invokes it.
GridFormat unreachable_grid_format(int) {
	FAIL("grid_format lookup should not be called with no grids configured");
	return GridFormat::Dense;
}

} // namespace

TEST_CASE("RigidBodyManager requires initialize() before use", "[device][rigidbody][manager]") {
	initialize_backend_once();
	Resource res(Global::single_resource_id);
	RigidBodyManager mgr({res});

	REQUIRE(mgr.size() == 0);
	REQUIRE_THROWS_AS(mgr.add_langevin_forces(1.0f, 1.0f, 42, 0), Exception);
}

TEST_CASE("RigidBodyManager rejects an out-of-range compute_resource_idx",
		  "[device][rigidbody][manager]") {
	initialize_backend_once();
	Resource res(Global::single_resource_id);
	REQUIRE_THROWS_AS(RigidBodyManager({res}, 1), Exception);
}

TEST_CASE("RigidBodyManager::initialize builds device state and an empty pair list",
		  "[device][rigidbody][manager]") {
	initialize_backend_once();
	Resource res(Global::single_resource_id);
	RigidBodyManager mgr({res});

	auto types = single_type(10.0f, Vector3(1.0f, 2.0f, 3.0f));
	auto host = single_body_at_rest(Vector3(1.0f, 2.0f, 3.0f));

	mgr.initialize(types, host, unreachable_grid_format);

	REQUIRE(mgr.size() == 1);
	REQUIRE(mgr.force_pairs().size() == 0);
}

TEST_CASE("RigidBodyManager::integrate_motion free-drifts a moving body",
		  "[device][rigidbody][manager]") {
	initialize_backend_once();
	Resource res(Global::single_resource_id);
	RigidBodyManager mgr({res});

	auto types = single_type(2.0f, Vector3(1.0f, 1.0f, 1.0f));
	HostRigidBodyData host;
	RigidBodyIO rb{};
	rb.id = 0;
	rb.type_id = 0;
	rb.position = Vector3(0.0f, 0.0f, 0.0f);
	rb.orientation = Matrix3(1.0f);
	rb.momentum = Vector3(2.0f, 0.0f, 0.0f); // mass=2 -> velocity 1.0 along x
	rb.angular_momentum = Vector3(0.0f, 0.0f, 0.0f);
	rb.force = Vector3(0.0f, 0.0f, 0.0f);
	rb.torque = Vector3(0.0f, 0.0f, 0.0f);
	host.push_back(rb);

	mgr.initialize(types, host, unreachable_grid_format);

	const float dt = 0.5f;
	mgr.integrate_motion(dt, PeriodicBox());

	HostRigidBodyData result;
	mgr.bodies().copy_to_host(result, 1);

	// No force/torque applied, so momentum is unchanged and the position
	// drifts by (momentum/mass) * dt * constants::velocity_scale, matching
	// legacy integrateDLM's substep-1 drift.
	const float expected_x = dt * (2.0f / 2.0f) * constants::velocity_scale;
	REQUIRE(result.momentum[0].x == Catch::Approx(2.0f));
	REQUIRE(result.position[0].x == Catch::Approx(expected_x));
	REQUIRE(result.position[0].y == Catch::Approx(0.0f));
	REQUIRE(result.position[0].z == Catch::Approx(0.0f));
}

TEST_CASE("RigidBodyManager::add_langevin_forces perturbs force/torque",
		  "[device][rigidbody][manager]") {
	initialize_backend_once();
	Resource res(Global::single_resource_id);
	RigidBodyManager mgr({res});

	auto types = single_type(1.0f, Vector3(1.0f, 1.0f, 1.0f));
	auto host = single_body_at_rest(Vector3(0.0f, 0.0f, 0.0f));
	mgr.initialize(types, host, unreachable_grid_format);

	Event evt = mgr.add_langevin_forces(/*dt=*/1.0f, /*kT=*/1.0f, /*base_seed=*/7, /*step=*/0);
	evt.wait();
	HostRigidBodyData result;
	mgr.bodies().copy_to_host(result, 1);

	// A body at rest with zero prior force/torque should pick up a nonzero
	// random kick from the Langevin thermostat (RBAddLangevinKernel).
	REQUIRE((result.force[0].x != 0.0f || result.force[0].y != 0.0f ||
			 result.force[0].z != 0.0f));
}

namespace {

// Same rho/u construction as Unit_test/Interactions/RigidBodyGridBatch.cpp
// and GridGrid.cpp: rho is a single unit voxel at index (3,3,3); u is a
// quadratic bowl centered at the same index (cubic interpolation reproduces
// it exactly). GridManager only loads grids from a file, so these are
// round-tripped through a temp .dx file rather than injected directly.
constexpr idx_t kGridN = 8;
constexpr float kGridDX = 1.0f;
constexpr float kGridC = 3.0f;

BaseGrid<float> make_rho_grid() {
	BaseGrid<float> g(Matrix3(kGridDX), Vector3(0.0f), kGridN, kGridN, kGridN);
	g.zero();
	g[3 + 3 * kGridN + 3 * kGridN * kGridN] = 1.0f;
	return g;
}

BaseGrid<float> make_u_grid() {
	BaseGrid<float> g(Matrix3(kGridDX), Vector3(0.0f), kGridN, kGridN, kGridN);
	for (idx_t ix = 0; ix < kGridN; ++ix) {
		for (idx_t iy = 0; iy < kGridN; ++iy) {
			for (idx_t iz = 0; iz < kGridN; ++iz) {
				const float dxc = float(ix) - kGridC;
				const float dyc = float(iy) - kGridC;
				const float dzc = float(iz) - kGridC;
				g[iz + iy * kGridN + ix * kGridN * kGridN] = -(dxc * dxc + dyc * dyc + dzc * dzc);
			}
		}
	}
	return g;
}

} // namespace

TEST_CASE("RigidBodyManager: prepare_grid_grid_dispatch + compute_grid_grid_forces "
		  "reproduces the known analytic RB-RB force/torque",
		  "[device][rigidbody][manager][gridgrid][batch]") {
	initialize_backend_once();
	if (!Tests::Global::backend_available) {
		WARN("No GPU backend available - skipping device cross-check");
		return;
	}
	Resource res(Global::single_resource_id);
	std::vector<Resource> resources{res};

	const auto tmp_dir = std::filesystem::temp_directory_path();
	const std::string rho_path = (tmp_dir / "arbd_test_rbmanager_rho.dx").string();
	const std::string u_path = (tmp_dir / "arbd_test_rbmanager_u.dx").string();
	DXReader::write_grid(make_rho_grid(), rho_path);
	DXReader::write_grid(make_u_grid(), u_path);

	GridManager grid_manager;
	grid_manager.set_resources(&resources);
	const GridKey rho_key = grid_manager.add_dense_grid(rho_path);
	const GridKey u_key = grid_manager.add_dense_grid(u_path);
	grid_manager.build_device_arrays();

	std::filesystem::remove(rho_path);
	std::filesystem::remove(u_path);

	// type0 ("A") owns the density grid; type1 ("B") owns the potential grid
	// - a true RB-RB pair (RigidBodyGridPair::is_pmf == false), matched by
	// the shared key name "Elec" (Phase 3's RigidBodyForcePairList).
	std::vector<RigidBodyType> types(2);
	types[0].name = "A";
	types[0].id = 0;
	types[0].mass = 1.0f;
	types[0].inertia = Vector3(1.0f, 1.0f, 1.0f);
	types[0].density_grid_keys = {"Elec"};
	types[0].density_grids = {GridTerm{rho_key.grid_id}};
	types[1].name = "B";
	types[1].id = 1;
	types[1].mass = 1.0f;
	types[1].inertia = Vector3(1.0f, 1.0f, 1.0f);
	types[1].potential_grid_keys = {"Elec"};
	types[1].potential_grids = {GridTerm{u_key.grid_id}};

	const Vector3 P0(0.0f, 0.0f, 0.0f);
	const Vector3 P1(-0.3f, -0.2f, -0.15f);
	HostRigidBodyData host;
	RigidBodyIO rb0{};
	rb0.id = 0;
	rb0.type_id = 0;
	rb0.position = P0;
	rb0.orientation = Matrix3(1.0f);
	host.push_back(rb0);
	RigidBodyIO rb1{};
	rb1.id = 1;
	rb1.type_id = 1;
	rb1.position = P1;
	rb1.orientation = Matrix3(1.0f);
	host.push_back(rb1);

	RigidBodyManager mgr(resources);
	mgr.initialize(types, host, [&](int id) { return grid_manager.get_grid_format(id); });
	REQUIRE(mgr.force_pairs().size() == 1);

	mgr.prepare_grid_grid_dispatch(grid_manager, /*grid_resource_idx=*/0);
	mgr.compute_grid_grid_forces(grid_manager, /*grid_resource_idx=*/0, /*step=*/0, /*cutoff=*/1000.0f)
		.wait();

	REQUIRE_FALSE(mgr.grid_grid_worklist_overflowed());

	HostRigidBodyData result;
	mgr.bodies().copy_to_host(result, 2);

	// Known analytic values for this exact configuration (see
	// GridGrid.cpp's "Grid-grid force matches finite-difference energy
	// gradient" test): the bowl's gradient at the sampled point gives
	// force = (0.6, 0.4, 0.3) on rb0.
	REQUIRE(result.force[0].x == Catch::Approx(0.6f).epsilon(0.01f));
	REQUIRE(result.force[0].y == Catch::Approx(0.4f).epsilon(0.01f));
	REQUIRE(result.force[0].z == Catch::Approx(0.3f).epsilon(0.01f));

	// rb1 receives the exact Newton's-third-law reaction.
	const Vector3 origin_offset = P0 - P1;
	const Vector3 expected_torque1 = -(result.torque[0] + origin_offset.cross(result.force[0]));
	REQUIRE(result.force[1].x == Catch::Approx(-result.force[0].x).epsilon(1e-4f));
	REQUIRE(result.force[1].y == Catch::Approx(-result.force[0].y).epsilon(1e-4f));
	REQUIRE(result.force[1].z == Catch::Approx(-result.force[0].z).epsilon(1e-4f));
	REQUIRE(result.torque[1].x == Catch::Approx(expected_torque1.x).epsilon(1e-4f));
	REQUIRE(result.torque[1].y == Catch::Approx(expected_torque1.y).epsilon(1e-4f));
	REQUIRE(result.torque[1].z == Catch::Approx(expected_torque1.z).epsilon(1e-4f));
}

TEST_CASE("RigidBodyManager: prepare_particle_grid_dispatch + compute_particle_rb_forces "
		  "reproduces the known analytic particle-RB force/torque",
		  "[device][rigidbody][manager][particlegrid][batch]") {
	initialize_backend_once();
	if (!Tests::Global::backend_available) {
		WARN("No GPU backend available - skipping device cross-check");
		return;
	}
	Resource res(Global::single_resource_id);
	std::vector<Resource> resources{res};

	// Same bowl construction as RigidBodyGridBatch.cpp/make_u_grid above,
	// round-tripped through a temp .dx file (GridManager only loads from file).
	const auto tmp_dir = std::filesystem::temp_directory_path();
	const std::string u_path = (tmp_dir / "arbd_test_rbmanager_particle_u.dx").string();
	DXReader::write_grid(make_u_grid(), u_path);

	GridManager grid_manager;
	grid_manager.set_resources(&resources);
	const GridKey u_key = grid_manager.add_dense_grid(u_path);
	grid_manager.build_device_arrays();
	std::filesystem::remove(u_path);

	std::vector<RigidBodyType> types(1);
	types[0].name = "A";
	types[0].id = 0;
	types[0].mass = 1.0f;
	types[0].inertia = Vector3(1.0f, 1.0f, 1.0f);
	types[0].potential_grids = {GridTerm{u_key.grid_id}};

	// RB at the world origin, identity orientation, so world position ==
	// grid-local position - the particle at (3.3, 3.2, 3.15) sits at a
	// fractional offset (0.3, 0.2, 0.15) from the bowl's center index.
	HostRigidBodyData rb_host;
	RigidBodyIO rb{};
	rb.id = 0;
	rb.type_id = 0;
	rb.position = Vector3(0.0f, 0.0f, 0.0f);
	rb.orientation = Matrix3(1.0f);
	rb_host.push_back(rb);

	RigidBodyManager mgr(resources);
	mgr.initialize(types, rb_host, [&](int id) { return grid_manager.get_grid_format(id); });

	DeviceParticle particles(1, res);
	HostParticleData particle_host;
	particle_host.resize(1);
	particle_host.global_id[0] = 0;
	particle_host.type_id[0] = 0;
	particle_host.pos[0] = Vector3(3.3f, 3.2f, 3.15f);
	particles.copy_from_host(particle_host, 1);
	particles.clear_forces();

	mgr.prepare_particle_grid_dispatch(types, /*num_particles=*/1);
	mgr.compute_particle_rb_forces(grid_manager, /*grid_resource_idx=*/0, particles.view()).wait();

	HostParticleData particle_result;
	particles.copy_to_host(particle_result, 1, /*need_energy=*/true);
	HostRigidBodyData rb_result;
	mgr.bodies().copy_to_host(rb_result, 1);

	// Analytic: force = -grad = 2*(0.3, 0.2, 0.15) = (0.6, 0.4, 0.3);
	// energy = -(0.3^2+0.2^2+0.15^2) = -0.1525.
	REQUIRE(particle_result.force[0].x == Catch::Approx(0.6f).epsilon(0.01f));
	REQUIRE(particle_result.force[0].y == Catch::Approx(0.4f).epsilon(0.01f));
	REQUIRE(particle_result.force[0].z == Catch::Approx(0.3f).epsilon(0.01f));
	REQUIRE(particle_result.energy[0] == Catch::Approx(-0.1525f).epsilon(0.01f));

	// RB reaction: force = -(0.6,0.4,0.3); torque = p x (-force) about the
	// grid's lab-frame origin (== rb.position here, since grid.origin==0).
	REQUIRE(rb_result.force[0].x == Catch::Approx(-0.6f).epsilon(0.01f));
	REQUIRE(rb_result.force[0].y == Catch::Approx(-0.4f).epsilon(0.01f));
	REQUIRE(rb_result.force[0].z == Catch::Approx(-0.3f).epsilon(0.01f));
	REQUIRE(rb_result.torque[0].x == Catch::Approx(0.30f).epsilon(0.02f));
	REQUIRE(rb_result.torque[0].y == Catch::Approx(-0.90f).epsilon(0.02f));
	REQUIRE(rb_result.torque[0].z == Catch::Approx(0.60f).epsilon(0.02f));
}
