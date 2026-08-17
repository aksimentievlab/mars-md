/**
 * @file RigidBodyExternalForces.cpp
 * @brief Tests for the host-set external force/torque path and the Brownian
 *        rigid-body integrator: RigidBodyType::check_damping, RBHostFTManager
 *        baseline/push composition, and RBIntegrateBDKernel.
 *
 * Every dynamic assertion runs at kT = 0 so the integrators are deterministic
 * and can be checked against closed form rather than a tolerance band.
 */

#include "System/RigidBodyManager.h"
#include "../catch_boiler.h"
#include "Constants.h"
#include "Objects/RigidBodyProperties.h"
#include "RBOperation/RBHostFTManager.h"

using namespace ARBD;

namespace {

std::vector<RigidBodyType> damped_type(Vector3 trans_damping, Vector3 rot_damping) {
	std::vector<RigidBodyType> types(1);
	types[0].name = "A";
	types[0].id = 0;
	types[0].mass = 2.0f;
	types[0].inertia = Vector3(1.0f, 1.0f, 1.0f);
	types[0].trans_damping = trans_damping;
	types[0].rot_damping = rot_damping;
	return types;
}

HostRigidBodyData bodies_at_origin(int count) {
	HostRigidBodyData host;
	for (int i = 0; i < count; ++i) {
		RigidBodyIO rb{};
		rb.id = i;
		rb.type_id = 0;
		rb.position = Vector3(0.0f, 0.0f, 0.0f);
		rb.orientation = Matrix3(1.0f);
		rb.momentum = Vector3(0.0f, 0.0f, 0.0f);
		rb.angular_momentum = Vector3(0.0f, 0.0f, 0.0f);
		rb.force = Vector3(0.0f, 0.0f, 0.0f);
		rb.torque = Vector3(0.0f, 0.0f, 0.0f);
		host.push_back(rb);
	}
	return host;
}

GridFormat unreachable_grid_format(int) {
	FAIL("grid_format lookup should not be called with no grids configured");
	return GridFormat::Dense;
}

} // namespace

// =============================================================================
// check_damping - pure host, no device needed
// =============================================================================

TEST_CASE("check_damping accepts a fully specified type", "[rigidbody][damping]") {
	RigidBodyType t = damped_type(Vector3(1.0f, 1.0f, 1.0f), Vector3(1.0f, 1.0f, 1.0f))[0];
	REQUIRE_NOTHROW(t.check_damping(IntegratorType::Brownian));
	REQUIRE_NOTHROW(t.check_damping(IntegratorType::Langevin));
}

TEST_CASE("check_damping rejects the default zero damping under Brownian",
		  "[rigidbody][damping]") {
	// trans_damping/rot_damping default to zero while mass/inertia do not, so a
	// config that simply omits transDamping reaches the kernel with an infinite
	// mobility and NaNs the whole trajectory on step 1.
	RigidBodyType t = damped_type(Vector3(0.0f, 0.0f, 0.0f), Vector3(1.0f, 1.0f, 1.0f))[0];
	REQUIRE_THROWS_AS(t.check_damping(IntegratorType::Brownian), Exception);

	// DLM never divides by damping, so the same type is legal there.
	REQUIRE_NOTHROW(t.check_damping(IntegratorType::Langevin));
}

TEST_CASE("check_damping rejects a single zero damping component", "[rigidbody][damping]") {
	RigidBodyType t = damped_type(Vector3(1.0f, 0.0f, 1.0f), Vector3(1.0f, 1.0f, 1.0f))[0];
	REQUIRE_THROWS_AS(t.check_damping(IntegratorType::Brownian), Exception);

	RigidBodyType r = damped_type(Vector3(1.0f, 1.0f, 1.0f), Vector3(1.0f, 1.0f, 0.0f))[0];
	REQUIRE_THROWS_AS(r.check_damping(IntegratorType::Brownian), Exception);
}

TEST_CASE("check_damping rejects non-positive mass or inertia for every integrator",
		  "[rigidbody][damping]") {
	RigidBodyType t = damped_type(Vector3(1.0f, 1.0f, 1.0f), Vector3(1.0f, 1.0f, 1.0f))[0];
	t.mass = 0.0f;
	REQUIRE_THROWS_AS(t.check_damping(IntegratorType::Brownian), Exception);
	REQUIRE_THROWS_AS(t.check_damping(IntegratorType::Langevin), Exception);

	RigidBodyType i = damped_type(Vector3(1.0f, 1.0f, 1.0f), Vector3(1.0f, 1.0f, 1.0f))[0];
	i.inertia = Vector3(1.0f, 0.0f, 1.0f);
	REQUIRE_THROWS_AS(i.check_damping(IntegratorType::Langevin), Exception);
}

// =============================================================================
// RBHostFTManager - host-set force/torque reaching the device
// =============================================================================

TEST_CASE("set_external_loads lands constantForce in the device external buffers",
		  "[device][rigidbody][external]") {
	initialize_backend_once();
	Resource res(Global::single_resource_id);
	RigidBodyManager mgr({res});

	auto types = damped_type(Vector3(1.0f, 1.0f, 1.0f), Vector3(1.0f, 1.0f, 1.0f));
	mgr.initialize(types, bodies_at_origin(3), unreachable_grid_format);

	const std::vector<Vector3> force{Vector3(1.0f, 0.0f, 0.0f),
									 Vector3(0.0f, 2.0f, 0.0f),
									 Vector3(0.0f, 0.0f, 3.0f)};
	const std::vector<Vector3> torque{Vector3(0.0f, 0.0f, 4.0f),
									  Vector3(0.0f, 5.0f, 0.0f),
									  Vector3(6.0f, 0.0f, 0.0f)};
	mgr.set_external_loads(force, torque);

	std::vector<Vector3> device_force;
	std::vector<Vector3> device_torque;
	mgr.bodies().external_force().copy_to_host(device_force);
	mgr.bodies().external_torque().copy_to_host(device_torque);

	for (size_t i = 0; i < force.size(); ++i) {
		REQUIRE(device_force[i].x == Catch::Approx(force[i].x));
		REQUIRE(device_force[i].y == Catch::Approx(force[i].y));
		REQUIRE(device_force[i].z == Catch::Approx(force[i].z));
		REQUIRE(device_torque[i].x == Catch::Approx(torque[i].x));
		REQUIRE(device_torque[i].y == Catch::Approx(torque[i].y));
		REQUIRE(device_torque[i].z == Catch::Approx(torque[i].z));
	}
}

TEST_CASE("push_with_baseline sums the constant and the dynamic contribution",
		  "[device][rigidbody][external]") {
	initialize_backend_once();
	Resource res(Global::single_resource_id);
	RigidBodyManager mgr({res});

	auto types = damped_type(Vector3(1.0f, 1.0f, 1.0f), Vector3(1.0f, 1.0f, 1.0f));
	mgr.initialize(types, bodies_at_origin(3), unreachable_grid_format);

	const std::vector<Vector3> constant(3, Vector3(1.0f, 0.0f, 0.0f));
	mgr.set_external_loads(constant, constant);

	// Body 1 only, as a dynamic producer (scuff-em) would push it.
	const std::vector<int> idx{1};
	const std::vector<Vector3> dynamic_force{Vector3(0.0f, 10.0f, 0.0f)};
	const std::vector<Vector3> dynamic_torque{Vector3(0.0f, 0.0f, 20.0f)};
	mgr.host_forces()
		.push_with_baseline(idx, dynamic_force, dynamic_torque, mgr.bodies().view())
		.wait();

	std::vector<Vector3> device_force;
	std::vector<Vector3> device_torque;
	mgr.bodies().external_force().copy_to_host(device_force);
	mgr.bodies().external_torque().copy_to_host(device_torque);

	// ApplyExternalForcesKernel assigns rather than accumulates, so without the
	// baseline this body's constantForce would have been erased outright.
	REQUIRE(device_force[1].x == Catch::Approx(1.0f));
	REQUIRE(device_force[1].y == Catch::Approx(10.0f));
	REQUIRE(device_torque[1].z == Catch::Approx(20.0f));

	// Untouched bodies keep the baseline alone.
	REQUIRE(device_force[0].x == Catch::Approx(1.0f));
	REQUIRE(device_force[0].y == Catch::Approx(0.0f));
	REQUIRE(device_force[2].x == Catch::Approx(1.0f));
}

TEST_CASE("RBHostFTManager rejects mismatched spans and over-capacity pushes",
		  "[device][rigidbody][external]") {
	initialize_backend_once();
	Resource res(Global::single_resource_id);
	RBHostFTManager host_ft(2, res);
	RigidBodyManager mgr({res});
	auto types = damped_type(Vector3(1.0f, 1.0f, 1.0f), Vector3(1.0f, 1.0f, 1.0f));
	mgr.initialize(types, bodies_at_origin(2), unreachable_grid_format);

	const std::vector<int> two_idx{0, 1};
	const std::vector<Vector3> one_vec{Vector3(0.0f)};
	REQUIRE_THROWS_AS(host_ft.push(two_idx, one_vec, one_vec, mgr.bodies().view()), Exception);

	const std::vector<int> three_idx{0, 1, 0};
	const std::vector<Vector3> three_vec(3, Vector3(0.0f));
	REQUIRE_THROWS_AS(host_ft.push(three_idx, three_vec, three_vec, mgr.bodies().view()),
					  Exception);
}

// =============================================================================
// Brownian integrator - deterministic at kT = 0
// =============================================================================

TEST_CASE("integrate_brownian drifts by mobility * force * dt at zero temperature",
		  "[device][rigidbody][brownian]") {
	initialize_backend_once();
	Resource res(Global::single_resource_id);
	RigidBodyManager mgr({res});

	const float damping = 4.0f;
	const float mass = 2.0f;
	auto types = damped_type(Vector3(damping, damping, damping), Vector3(1.0f, 1.0f, 1.0f));
	mgr.initialize(types, bodies_at_origin(1), unreachable_grid_format);

	// Applied purely as an external load, which is what constantForce becomes.
	const std::vector<Vector3> force{Vector3(3.0f, 0.0f, 0.0f)};
	const std::vector<Vector3> torque{Vector3(0.0f, 0.0f, 0.0f)};
	mgr.set_external_loads(force, torque);

	const float dt = 0.5f;
	mgr.integrate_brownian(dt, /*kT=*/0.0f, /*base_seed=*/7, /*step=*/0, PeriodicBox()).wait();

	HostRigidBodyData result;
	mgr.bodies().copy_to_host(result, 1);

	// kT = 0 kills the noise term, leaving offset = (F / (damping*mass)) * dt
	// with the orientation identity. damping is scaled at use by
	// langevin_damping_unit, exactly as the kernel does.
	const float scaled_damping = damping * constants::langevin_damping_unit;
	const float expected_x = 3.0f / (scaled_damping * mass) * dt;

	REQUIRE(result.position[0].x == Catch::Approx(expected_x).epsilon(1e-4));
	REQUIRE(result.position[0].y == Catch::Approx(0.0f).margin(1e-5));
	REQUIRE(result.position[0].z == Catch::Approx(0.0f).margin(1e-5));
	REQUIRE(std::isfinite(result.position[0].x));
}

TEST_CASE("integrate_brownian leaves a force-free body put at zero temperature",
		  "[device][rigidbody][brownian]") {
	initialize_backend_once();
	Resource res(Global::single_resource_id);
	RigidBodyManager mgr({res});

	auto types = damped_type(Vector3(4.0f, 4.0f, 4.0f), Vector3(1.0f, 1.0f, 1.0f));
	mgr.initialize(types, bodies_at_origin(1), unreachable_grid_format);

	mgr.integrate_brownian(0.5f, 0.0f, 7, 0, PeriodicBox()).wait();

	HostRigidBodyData result;
	mgr.bodies().copy_to_host(result, 1);
	REQUIRE(result.position[0].x == Catch::Approx(0.0f).margin(1e-6));
	REQUIRE(result.position[0].y == Catch::Approx(0.0f).margin(1e-6));
	REQUIRE(result.position[0].z == Catch::Approx(0.0f).margin(1e-6));
}

TEST_CASE("integrate_brownian stays finite over many steps at finite temperature",
		  "[device][rigidbody][brownian]") {
	initialize_backend_once();
	Resource res(Global::single_resource_id);
	RigidBodyManager mgr({res});

	auto types = damped_type(Vector3(4.0f, 4.0f, 4.0f), Vector3(4.0f, 4.0f, 4.0f));
	mgr.initialize(types, bodies_at_origin(4), unreachable_grid_format);

	for (size_t step = 0; step < 64; ++step) {
		mgr.bodies().clear_forces();
		mgr.integrate_brownian(1e-3f, 0.6f, 12345, step, PeriodicBox()).wait();
	}

	HostRigidBodyData result;
	mgr.bodies().copy_to_host(result, 4);
	for (int i = 0; i < 4; ++i) {
		REQUIRE(std::isfinite(result.position[i].x));
		REQUIRE(std::isfinite(result.position[i].y));
		REQUIRE(std::isfinite(result.position[i].z));
		// Orientation must stay a rotation; normalize_orientation runs every step.
		REQUIRE(result.orientation[i].det() == Catch::Approx(1.0f).epsilon(1e-3));
	}
}

// =============================================================================
// The external load must reach BOTH integrators, not just the thermostatted one
// =============================================================================

TEST_CASE("integrate_motion applies the external load without add_langevin_forces",
		  "[device][rigidbody][external]") {
	initialize_backend_once();
	Resource res(Global::single_resource_id);
	RigidBodyManager mgr({res});

	auto types = damped_type(Vector3(1.0f, 1.0f, 1.0f), Vector3(1.0f, 1.0f, 1.0f));
	mgr.initialize(types, bodies_at_origin(1), unreachable_grid_format);

	const std::vector<Vector3> force{Vector3(5.0f, 0.0f, 0.0f)};
	const std::vector<Vector3> torque{Vector3(0.0f, 0.0f, 0.0f)};
	mgr.set_external_loads(force, torque);

	// Deliberately no add_langevin_forces() call: before the external term moved
	// out of RBLangevinForceKernel and into the integrators, this produced zero.
	const float dt = 0.25f;
	mgr.integrate_motion(dt, PeriodicBox()).wait();

	HostRigidBodyData result;
	mgr.bodies().copy_to_host(result, 1);

	// Substeps 0 and 2 each half-kick: p = 2 * 0.5 * dt * F * impulse_to_momentum.
	const float expected_p = dt * 5.0f * constants::impulse_to_momentum;
	REQUIRE(result.momentum[0].x == Catch::Approx(expected_p).epsilon(1e-4));
}

TEST_CASE("integrate_motion applies external torque in the lab frame",
		  "[device][rigidbody][external]") {
	initialize_backend_once();
	Resource res(Global::single_resource_id);
	RigidBodyManager mgr({res});

	// The companion force test above leaves torque at zero, so nothing pinned
	// the rotational half of the external-load move out of
	// RBLangevinForceKernel. What matters is the frame: substeps 0 and 2 apply
	// orientation.transpose() to (torque + external_torque), so external_torque
	// must be a lab-frame quantity, matching legacy's lab-frame `torque`
	// accumulator (RigidBody.h:90) that constantTorque was added to.
	//
	// Inertia is deliberately huge so substep 1's rotation angle
	// (dt * L / I * 1e4) is ~1e-8 rad: both half-kicks then see the same
	// orientation and the expected value stays closed form.
	auto types = damped_type(Vector3(1.0f, 1.0f, 1.0f), Vector3(1.0f, 1.0f, 1.0f));
	types[0].inertia = Vector3(1.0e15f, 1.0e15f, 1.0e15f);

	// Body rotated +90 deg about x, so lab z maps onto body y and the transpose
	// is actually exercised - an identity orientation would pass either way.
	HostRigidBodyData host = bodies_at_origin(1);
	host.orientation[0] = rotation_matrix_x(2.0f); // Cayley t=2 is exactly 90 deg
	mgr.initialize(types, host, unreachable_grid_format);

	const float torque_z = 4.0f;
	const std::vector<Vector3> force{Vector3(0.0f, 0.0f, 0.0f)};
	const std::vector<Vector3> torque{Vector3(0.0f, 0.0f, torque_z)};
	mgr.set_external_loads(force, torque);

	const float dt = 0.25f;
	mgr.integrate_motion(dt, PeriodicBox()).wait();

	HostRigidBodyData result;
	mgr.bodies().copy_to_host(result, 1);

	// Rx(90)^T * (0,0,T) = (0,T,0): the lab-z torque lands on body y.
	const float expected = dt * torque_z * constants::impulse_to_momentum;
	const Vector3 L = result.angular_momentum[0];
	INFO("angular_momentum = " << L.x << " " << L.y << " " << L.z);
	CHECK(L.x == Catch::Approx(0.0f).margin(1e-3f * expected));
	CHECK(L.y == Catch::Approx(expected).epsilon(1e-4));
	CHECK(L.z == Catch::Approx(0.0f).margin(1e-3f * expected));
}
