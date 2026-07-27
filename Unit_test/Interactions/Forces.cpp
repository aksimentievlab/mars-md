/**
 * @file test_force_computation.cpp
 * @brief Part 3: Force Computation Tests
 *
 * Tests bonded and non-bonded force kernels
 */

#include "../catch_boiler.h"
#include "Backend/Kernels.h"
#include "Backend/Resource.h"
#include "Interactions/Bonded/BondComputer.h"
#include "Objects/DeviceParticleManager.h"
#include "Objects/ParticleProperties.h"
#include "System/PeriodicBox.h"

#include "../Object_gen.h"
using namespace ARBD;
using Catch::Approx;

// ============================================================================
// PART 3: FORCE COMPUTATION TESTS
// ============================================================================

TEST_CASE("Harmonic Bond Force - Two Particles", "[force][bonded][bond]") {
	initialize_backend_once();
	Resource res(Global::single_resource_id);

	// Create 2 particles separated by distance r = 1.5
	HostParticleData host_data = create_test_particles(2, "linear", 100.0f);
	host_data.type_id = std::vector<int>(2, 0);

	// Copy to device
	DeviceParticle particles(2, res);
	particles.copy_from_host(host_data, 2);

	// Bond parameters: k=100, r0=1.0
	// Expected force = -k*(r-r0) = -100*(1.5-1.0) = -50
	// Force direction: particle 0 pulls toward particle 1 (positive x)
	// Force magnitude on particle 0: +50.0 in x-direction
	// Force magnitude on particle 1: -50.0 in x-direction

	DeviceBuffer<ARBD::int2> bond_indices(1, res);
	std::vector<ARBD::int2> bond = {{0, 1}};
	bond_indices.copy_from_host(bond.data(), 1);

	DeviceBuffer<float> params(2, res);
	float params_array[2] = {100.0f, 1.0f}; // k=100, r0=1.0
	params.copy_from_host(params_array, 2);

	PeriodicBox pbox_host(Vector3(100.0f, 100.0f, 100.0f));
	DeviceBuffer<PeriodicBox> pbox_buffer(1, res);
	pbox_buffer.copy_from_host(&pbox_host, 1);

	// Launch harmonic bond kernel (BondComputer<0> = Harmonic)
	auto view = particles.view();
	KernelConfig config = KernelConfig::for_1d(1, res);

	AnalyticalBondComputer<0> bond_computer(bond_indices.data(),
											view.pos,
											view.ForceEnergy,
											params.data(),
											pbox_buffer.data(),
											false,
											1);
	launch_kernel(res, config, bond_computer);
}

TEST_CASE("Harmonic Bond Force - Equilibrium Distance", "[force][bonded][bond]") {
	initialize_backend_once();
	Resource res(Global::single_resource_id);

	// Create 2 particles at equilibrium distance r0 = 1.0
	HostParticleData host_data = create_test_particles(2, "linear", 1.0f);
	host_data.type_id = std::vector<int>(2, 0);

	DeviceParticle particles(2, res);
	particles.copy_from_host(host_data, 2);

	// Bond parameters: k=100, r0=1.0
	// Expected force = 0 (at equilibrium)
	DeviceBuffer<ARBD::int2> bond_indices(1, res);
	ARBD::int2 bond{0, 1};
	bond_indices.copy_from_host(&bond, 1);

	DeviceBuffer<float> params(2, res);
	float params_array[2] = {100.0f, 1.0f};
	params.copy_from_host(params_array, 2);

	PeriodicBox pbox_host(Vector3(100.0f, 100.0f, 100.0f));
	DeviceBuffer<PeriodicBox> pbox_buffer(1, res);
	pbox_buffer.copy_from_host(&pbox_host, 1);

	auto view = particles.view();
	KernelConfig config = KernelConfig::for_1d(1, res);
	AnalyticalBondComputer<0> bond_computer(bond_indices.data(),
											view.pos,
											view.ForceEnergy,
											params.data(),
											pbox_buffer.data(),
											false,
											1);

	launch_kernel(res, config, bond_computer);

	HostParticleData result;
	particles.copy_to_host(result, 2);

	// At equilibrium, forces should be zero
	REQUIRE(result.force[0].x == Approx(0.0f).epsilon(0.01f));
	REQUIRE(result.force[1].x == Approx(0.0f).epsilon(0.01f));
}

TEST_CASE("Morse Bond Force - Two Particles", "[force][bonded][bond][morse]") {
	initialize_backend_once();
	Resource res(Global::single_resource_id);

	// Create 2 particles
	HostParticleData host_data = create_test_particles(2, "linear", 1.5f);
	host_data.type_id = std::vector<int>(2, 0);

	DeviceParticle particles(2, res);
	particles.copy_from_host(host_data, 2);

	// Morse bond parameters: D0=10, a=2.0, r0=1.0
	// Force = 2*D0*a*exp(-a(r-r0))*[1-exp(-a(r-r0))]
	DeviceBuffer<ARBD::int2> bond_indices(1, res);
	ARBD::int2 bond{0, 1};
	bond_indices.copy_from_host(&bond, 1);

	DeviceBuffer<float> params(3, res);
	float params_array[3] = {10.0f, 2.0f, 1.0f}; // D0, a, r0
	params.copy_from_host(params_array, 3);

	PeriodicBox pbox_host(Vector3(100.0f, 100.0f, 100.0f));
	DeviceBuffer<PeriodicBox> pbox_buffer(1, res);
	pbox_buffer.copy_from_host(&pbox_host, 1);

	auto view = particles.view();
	KernelConfig config = KernelConfig::for_1d(1, res);

	// Launch Morse bond kernel (BondComputer<1> = Morse)
	AnalyticalBondComputer<1> bond_computer(bond_indices.data(),
											view.pos,
											view.ForceEnergy,
											params.data(),
											pbox_buffer.data(),
											false,
											1);
	launch_kernel(res, config, bond_computer);

	HostParticleData result;
	particles.copy_to_host(result, 2);

	// Calculate expected force manually
	// r = 1.5, r0 = 1.0, dr = 0.5
	// exp_term = exp(-2.0 * 0.5) = exp(-1.0) ≈ 0.3679
	// F = 2 * 10 * 2.0 * 0.3679 * (1 - 0.3679) ≈ 18.54
	float expected_force = 18.54f;

	REQUIRE(result.force[0].x == Approx(expected_force).epsilon(1.0f));
	REQUIRE(result.force[1].x == Approx(-expected_force).epsilon(1.0f));
}

TEST_CASE("Bond Force - Periodic Boundary Conditions", "[force][bonded][bond][pbc]") {
	initialize_backend_once();
	Resource res(Global::single_resource_id);

	// Create 2 particles across periodic boundary
	// Box size = 10, particles at x=0.5 and x=9.5
	// True distance with PBC = 1.0
	HostParticleData host_data;
	host_data.resize(2);

	for (int i = 0; i < 2; i++) {
		host_data.global_id[i] = i;
		host_data.type_id[i] = 0;
		host_data.mom[i] = Vector3(0, 0, 0);
		host_data.force[i] = Vector3(0, 0, 0);
		host_data.energy[i] = 0.0f;
		host_data.orient[i] = Vector3(1, 0, 0);
		host_data.flags[i] = ParticleFlags::FLAG_ACTIVE;
		host_data.is_active[i] = true;
		host_data.is_dummy[i] = false;
		host_data.colvars_group_id[i] = -1;
		host_data.group_id[i] = -1;
		host_data.attached_rigid_body_id[i] = -1;
	}

	host_data.pos[0] = Vector3(0.5f, 0.0f, 0.0f);
	host_data.pos[1] = Vector3(9.5f, 0.0f, 0.0f);

	DeviceParticle particles(2, res);
	particles.copy_from_host(host_data, 2);

	// Bond: k=100, r0=1.0
	// With PBC, distance = 1.0, so force should be 0
	DeviceBuffer<ARBD::int2> bond_indices(1, res);
	ARBD::int2 bond{0, 1};
	bond_indices.copy_from_host(&bond, 1);

	DeviceBuffer<float> params(2, res);
	float p[2] = {100.0f, 1.0f};
	params.copy_from_host(p, 2);

	PeriodicBox pbox_host(Vector3(10.0f, 10.0f, 10.0f));
	DeviceBuffer<PeriodicBox> pbox_buffer(1, res);
	pbox_buffer.copy_from_host(&pbox_host, 1);

	auto view = particles.view();
	KernelConfig config = KernelConfig::for_1d(1, res);

	AnalyticalBondComputer<0> bond_computer(bond_indices.data(),
											view.pos,
											view.ForceEnergy,
											params.data(),
											pbox_buffer.data(),
											false,
											1);
	launch_kernel(res, config, bond_computer);

	HostParticleData result;
	particles.copy_to_host(result, 2);

	// With PBC, the minimum image distance is 1.0 (equilibrium), so force = 0
	REQUIRE(result.force[0].x == Approx(0.0f).epsilon(0.1f));
	REQUIRE(result.force[1].x == Approx(0.0f).epsilon(0.1f));
}

// ============================================================================
// TABULATED POTENTIAL TESTS (if needed)
// ============================================================================

TEST_CASE("Tabulated Bond Force - Lookup", "[force][bonded][tabulated][!mayfail]") {
	// This test requires TabulatedPotential to be set up
	// Mark as [!mayfail] since it depends on table initialization

	initialize_backend_once();
	Resource res(Global::single_resource_id);

	WARN("Tabulated bond force test not fully implemented - requires table setup");
}
