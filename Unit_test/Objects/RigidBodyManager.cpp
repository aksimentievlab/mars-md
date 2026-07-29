/**
 * @file RigidBodyManager.cpp
 * @brief Tests for Phase 4's orchestration skeleton (Objects/RigidBodyManager.h):
 *        construction/initialize(), Langevin forces, and DLM integration,
 *        batched across RB instances via the SoA views from Phase 2.
 */

#include "../catch_boiler.h"
#include "Constants.h"
#include "Objects/RigidBodyManager.h"
#include "Objects/RigidBodyProperties.h"

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

	mgr.add_langevin_forces(/*dt=*/1.0f, /*kT=*/1.0f, /*base_seed=*/7, /*step=*/0);

	HostRigidBodyData result;
	mgr.bodies().copy_to_host(result, 1);

	// A body at rest with zero prior force/torque should pick up a nonzero
	// random kick from the Langevin thermostat (RBAddLangevinKernel).
	REQUIRE((result.force[0].x != 0.0f || result.force[0].y != 0.0f ||
			 result.force[0].z != 0.0f));
}
