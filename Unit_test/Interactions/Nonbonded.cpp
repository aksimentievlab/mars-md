/**
 * @file test_pairwise_nonbonded.cpp
 * @brief Tests for pairwise nonbonded force kernels
 *
 * Tests the CORRECTED kernels from PairwiseNonbonded.h:
 * - SoftcoreForceKernel (LJ repulsion)
 * - ColumbForceKernel (electrostatics)
 * - Combined kernel
 *
 * NOTE: Uses existing kernel definitions, does NOT define new ones!
 */

#include "../catch_boiler.h"

#include "Backend/Kernels.h"
#include "Backend/Resource.h"
#include "Interactions/Nonbonded/Columb.h"
#include "Interactions/Nonbonded/Pairwise.h"
#include "Objects/DeviceParticleManager.h"
#include "PatchOperation/Pairlist.h"
#include "System/PeriodicBox.h"
#include "Types/Types.h"

using namespace ARBD;

namespace {

/**
 * @brief Helper to create simple 2-particle system
 */
std::pair<HostParticleData, std::vector<ParticleType>> create_two_particles(arbd_real separation) {
	HostParticleData host_data(std::vector<int>{0, 1});
	host_data.global_id = {0, 1};
	host_data.type_id = {0, 0};
	Vector3 zero(0.0f, 0.0f, 0.0f);
	Vector3 separation_vec(separation, 0.0f, 0.0f);
	host_data.pos = std::vector<Vector3>{zero, separation_vec};
	host_data.mom = std::vector<Vector3>{zero, zero};
	host_data.force = std::vector<Vector3>{zero, zero};
	host_data.orient = std::vector<Vector3>{zero, zero};
	host_data.flags = std::vector<uint32_t>{0, 0};

	// Particle type
	std::vector<ParticleType> types({ParticleType("A")});
	types[0].name = "A";
	types[0].mass = 1.0f;
	types[0].charge = 1.0f; // Unit charge
	types[0].radius = 1.0f; // Unit radius
	types[0].eps = 1.0f;	// Unit epsilon
	types[0].diffusion = Vector3(1.0f, 1.0f, 1.0f);

	return {host_data, types};
}

} // anonymous namespace

TEST_CASE("Softcore LJ - Repulsion", "[forces][nonbonded][softcore]") {
	initialize_backend_once();
	Resource res(Global::single_resource_id);

	// Create 2 particles at distance 0.8 (inside sigma=1.0)
	auto [host_data, types] = create_two_particles(0.8f);

	DeviceParticle particles(2, res);
	particles.copy_from_host(host_data, 2);

	// Create particle types
	DeviceParticleTypes type_manager(types, res);

	// Build pairlist
	auto pairlist = create_pairlist(PairlistBuilderType::ZOrder, res, 10, 10);
	pairlist->build_pairlist(particles.pos(), 2, 2.0f);
	REQUIRE(pairlist->get_num_pairs() == 1);

	// Create periodic box
	PeriodicBox pbox_host(Vector3(100, 100, 100));
	DeviceBuffer<PeriodicBox> pbox_buffer(1, res);
	pbox_buffer.copy_from_host(&pbox_host, 1);

	// Create softcore kernel
	auto particle_view = particles.view();
	auto type_view = type_manager.view();

	SoftcoreForceKernel kernel{particle_view,
							   type_view,
							   pbox_buffer.data(),
							   pairlist->get_num_pairs()};

	// Launch
	KernelConfig config = KernelConfig::for_1d(pairlist->get_num_pairs(), res);
	Event evt = launch_kernel(res, config, kernel);
	evt.wait();

	// Check forces
	HostParticleData result;
	particles.copy_to_host(result, 2);

	// At r=0.8 < sigma=1.0, should be repulsive (pushing apart)
	REQUIRE(result.force[0].x < 0.0f); // Particle 0 pushed left
	REQUIRE(result.force[1].x > 0.0f); // Particle 1 pushed right

	// Forces should be equal and opposite
	REQUIRE(std::abs(result.force[0].x + result.force[1].x) < 1e-5f);

	LOGINFO("Softcore forces at r=0.8: F0={:.3f}, F1={:.3f}", result.force[0].x, result.force[1].x);
}

TEST_CASE("Softcore LJ - No Force Outside Core", "[forces][nonbonded][softcore]") {
	Resource res;

	// Create 2 particles at distance 1.5 (outside sigma=1.0)
	auto [host_data, types] = create_two_particles(1.5f);

	DeviceParticle particles(2, res);
	particles.copy_from_host(host_data, 2);

	DeviceParticleTypes type_manager(types, res);
	type_manager.copy_from_host(types);

	auto pairlist = create_pairlist(PairlistBuilderType::CellList, res, 10, 10);
	pairlist->build_pairlist(particles.pos(), 2, 2.0f);

	PeriodicBox pbox_host(Vector3(100, 100, 100));
	DeviceBuffer<PeriodicBox> pbox_buffer(1, res);
	pbox_buffer.copy_from_host(&pbox_host, 1);

	auto particle_view = particles.view();
	auto type_view = type_manager.view();

	SoftcoreForceKernel kernel{particle_view,
							   &type_view,
							   pairlist->get_neighbor_pairs().data(),
							   pbox_buffer.data(),
							   pairlist->get_num_pairs(),
							   false};

	launch_kernel(res, KernelConfig::for_1d(pairlist->get_num_pairs(), res), kernel);
	res.synchronize();

	HostParticleData result;
	particles.copy_to_host(result, 2);

	// Outside core - should have zero force
	REQUIRE(result.force[0].x == Catch::Matchers::WithinAbs(0.0f, 1e-6f));
	REQUIRE(result.force[1].x == Catch::Matchers::WithinAbs(0.0f, 1e-6f));
}

TEST_CASE("Coulomb - Repulsion Between Like Charges", "[forces][nonbonded][coulomb]") {
	Resource res;

	// Create 2 particles at distance 2.0
	auto [host_data, types] = create_two_particles(2.0f);

	// Both have charge +1.0
	types[0].charge = 1.0f;

	DeviceParticle particles(2, res);
	particles.copy_from_host(host_data, 2);

	DeviceParticleTypes type_manager(types, res);
	type_manager.copy_from_host(types);

	auto pairlist = create_pairlist(PairlistBuilderType::CellList, res, 10, 10);
	pairlist->build_pairlist(particles.pos(), 2, 3.0f);

	PeriodicBox pbox_host(Vector3(100, 100, 100));
	DeviceBuffer<PeriodicBox> pbox_buffer(1, res);
	pbox_buffer.copy_from_host(&pbox_host, 1);

	auto particle_view = particles.view();
	auto type_view = type_manager.view();

	ColumbForceKernel<float> kernel{particle_view,
									&type_view,
									pairlist->get_neighbor_pairs().data(),
									pbox_buffer.data(),
									pairlist->get_num_pairs(),
									false};

	launch_kernel(res, KernelConfig::for_1d(pairlist->get_num_pairs(), res), kernel);

	HostParticleData result;
	particles.copy_to_host(result, 2);

	// F = COULOMB * q1 * q2 / r^2 = 332.06 * 1 * 1 / 4 = 83.015
	float expected_force = constants::COULOMB * 1.0f * 1.0f / (2.0f * 2.0f);

	// Like charges repel
	REQUIRE(result.force[0].x == Catch::Matchers::WithinRel(-expected_force, 0.01f));
	REQUIRE(result.force[1].x == Catch::Matchers::WithinRel(expected_force, 0.01f));

	// Equal and opposite
	REQUIRE(std::abs(result.force[0].x + result.force[1].x) < 1e-4f);
}

TEST_CASE("Coulomb - Attraction Between Opposite Charges", "[forces][nonbonded][coulomb]") {
	Resource res;

	auto [host_data, types] = create_two_particles(2.0f);

	// Create two types: +1 and -1
	types.push_back(types[0]);
	types[0].charge = 1.0f;
	types[1].charge = -1.0f;

	// Assign different types
	host_data.type_id = {0, 1};

	DeviceParticle particles(2, res);
	particles.copy_from_host(host_data, 2);

	DeviceParticleManager type_manager(res);
	type_manager.copy_from_host(types);

	auto pairlist = create_pairlist(PairlistBuilderType::CellList, res, 10, 10);
	pairlist->build_pairlist(particles.pos(), 2, 3.0f);

	PeriodicBox pbox_host(Vector3(100, 100, 100));
	DeviceBuffer<PeriodicBox> pbox_buffer(1, res);
	pbox_buffer.copy_from_host(&pbox_host, 1);

	auto particle_view = particles.view();
	auto type_view = type_manager.view();

	ColumbForceKernel kernel{particle_view,
							 &type_view,
							 pairlist->get_neighbor_pairs().data(),
							 pbox_buffer.data(),
							 pairlist->get_num_pairs(),
							 false};

	launch_kernel(res, KernelConfig::for_1d(pairlist->get_num_pairs(), res), kernel);
	res.synchronize();

	HostParticleData result;
	particles.copy_to_host(result, 2);

	// F = COULOMB * q1 * q2 / r^2 = 332.06 * 1 * (-1) / 4 = -83.015
	float expected_force = constants::COULOMB * 1.0f * (-1.0f) / (2.0f * 2.0f);

	// Opposite charges attract (forces point toward each other)
	REQUIRE(result.force[0].x == Catch::Matchers::WithinRel(-expected_force, 0.01));
	REQUIRE(result.force[1].x == Catch::Matchers::WithinRel(expected_force, 0.01));

	// Equal and opposite
	REQUIRE(std::abs(result.force[0].x + result.force[1].x) < 1e-4f);

	LOGINFO("Coulomb forces (opposite charges): F0={:.3f}, F1={:.3f}",
			result.force[0].x,
			result.force[1].x);
}

TEST_CASE("Combined Softcore + Coulomb", "[forces][nonbonded][combined]") {
	Resource res;

	// Particles at r=0.9 (inside softcore, close enough for strong Coulomb)
	auto [host_data, types] = create_two_particles(0.9f);
	types[0].charge = 1.0f;
	types[0].eps = 1.0f;
	types[0].radius = 1.0f;

	DeviceParticle particles(2, res);
	particles.copy_from_host(host_data, 2);

	DeviceParticleManager type_manager(res);
	type_manager.copy_from_host(types);

	auto pairlist = create_pairlist(PairlistBuilderType::CellList, res, 10, 10);
	pairlist->build_pairlist(particles.pos(), 2, 2.0f);

	PeriodicBox pbox_host(Vector3(100, 100, 100));
	DeviceBuffer<PeriodicBox> pbox_buffer(1, res);
	pbox_buffer.copy_from_host(&pbox_host, 1);

	auto particle_view = particles.view();
	auto type_view = type_manager.view();

	SoftcoreColumbKernel kernel{particle_view,
								&type_view,
								pairlist->get_neighbor_pairs().data(),
								pbox_buffer.data(),
								pairlist->get_num_pairs(),
								false};

	launch_kernel(res, KernelConfig::for_1d(pairlist->get_num_pairs(), res), kernel);
	res.synchronize();

	HostParticleData result;
	particles.copy_to_host(result, 2);

	// Both should contribute to repulsion
	REQUIRE(result.force[0].x < 0.0f);
	REQUIRE(result.force[1].x > 0.0f);

	// Forces should be equal and opposite
	REQUIRE(std::abs(result.force[0].x + result.force[1].x) < 1e-4f);

	// Combined force should be stronger than either alone
	// (Both LJ and Coulomb repel at this distance)

	LOGINFO("Combined forces at r=0.9: F0={:.3f}, F1={:.3f}", result.force[0].x, result.force[1].x);
}
