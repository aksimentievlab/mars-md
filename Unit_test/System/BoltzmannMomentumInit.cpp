#include "../catch_boiler.h"
#include "Constants.h"
#include "Objects/ParticleProperties.h"
#include "SimManager.h"
#include "System/PatchManager.h"
#include "System/SimSystem.h"
#include <cmath>
#include <vector>

using namespace MARS;
using namespace Tests;
using Catch::Approx;

namespace {

void configure_free_system(SimSystem& sys,
						   std::vector<ParticleIO>& particles,
						   IntegratorType integrator,
						   const Vector3& initial_momentum,
						   int num_particles,
						   float mass) {
	const float box = 400.0f;
	sys.set_box_size(box, box, box);
	sys.set_periodicity(true, true, true);
	sys.set_temperature(300.0f);
	sys.set_cutoff(10.0f);
	sys.set_pairlist_cutoff(20.0f);
	sys.set_timestep(2e-5f);
	sys.set_num_steps(0);
	sys.set_output_period(1.0f);
	sys.set_energy_output_period(1.0f);
	sys.set_output_name("boltzmann_momentum_test");
	sys.set_estimated_particles(num_particles + 16);
	sys.set_base_seed(12345); // deterministic: generator uses base_seed + 2
	sys.set_particle_integrator_type(integrator);

	ParticleType ptype("Ar");
	ptype.mass = mass;
	const float D = 149.0f;
	ptype.diffusion = Vector3(D, D, D);
	sys.add_particle_type(ptype);

	particles.clear();
	particles.reserve(num_particles);
	for (int i = 0; i < num_particles; ++i) {
		ParticleIO p;
		p.id = i;
		p.type_name = "Ar";
		p.position = Vector3(box * 0.5f, box * 0.5f, box * 0.5f);
		p.momentum = initial_momentum;
		p.force = Vector3(0.0f);
		p.energy = 0.0f;
		particles.push_back(p);
	}
}

/// Pull every particle's momentum back from the (single) patch after init().
std::vector<Vector3> gather_momenta(SimSystem& sys, int num_particles) {
	auto* pm = sys.get_patch_manager();
	REQUIRE(pm->get_num_patches() == 1);
	HostParticleData data;
	pm->get_patch(0).copy_particles_to_host(data, 0, num_particles);
	REQUIRE(data.size() == static_cast<size_t>(num_particles));
	return data.mom;
}

} // namespace

TEST_CASE("Langevin with no input momentum is Boltzmann-seeded",
		  "[SimManager][Langevin][momentum]") {
	initialize_backend_once();

	const int num_particles = 4000;
	const float mass = 39.948f; // amu
	std::vector<Resource> resources = {Resource(::Global::single_resource_id)};
	SimSystem sys(resources);
	std::vector<ParticleIO> particles;
	configure_free_system(sys,
						  particles,
						  IntegratorType::Langevin,
						  Vector3(0.0f),
						  num_particles,
						  mass);

	SimManager manager(sys);
	manager.set_initial_particles(particles);
	REQUIRE_NOTHROW(manager.init());

	const std::vector<Vector3> mom = gather_momenta(sys, num_particles);

	// Not all zero - seeding actually happened.
	double sum_sq = 0.0;
	Vector3 mean(0.0f);
	for (const auto& p : mom) {
		mean += p;
		sum_sq += p.length2();
	}
	REQUIRE(sum_sq > 0.0);
	mean = mean / static_cast<double>(num_particles);

	// Center-of-mass drift removed (v_com = 0): mean momentum ~ 0 per component.
	// Scale tolerance to a single-particle standard deviation / sqrt(N).
	const float kT = sys.get_temperature_struct().kT;
	const double C = constants::SQRT_CAL_TO_JOULE;
	const double expected_var = kT * mass * C * C; // <p_i^2> per component
	const double sd = std::sqrt(expected_var);
	const double mean_tol = 3.0 * sd / std::sqrt(static_cast<double>(num_particles));
	REQUIRE(std::abs(mean.x) < mean_tol);
	REQUIRE(std::abs(mean.y) < mean_tol);
	REQUIRE(std::abs(mean.z) < mean_tol);

	// Per-component variance matches Maxwell-Boltzmann: Var(p_i) = kT * m * C^2.
	double var_x = 0.0, var_y = 0.0, var_z = 0.0;
	for (const auto& p : mom) {
		var_x += (p.x - mean.x) * (p.x - mean.x);
		var_y += (p.y - mean.y) * (p.y - mean.y);
		var_z += (p.z - mean.z) * (p.z - mean.z);
	}
	var_x /= num_particles;
	var_y /= num_particles;
	var_z /= num_particles;

	// 10% tolerance: deterministic seed, N=4000 (relative sampling error ~2%).
	REQUIRE(var_x == Approx(expected_var).epsilon(0.10));
	REQUIRE(var_y == Approx(expected_var).epsilon(0.10));
	REQUIRE(var_z == Approx(expected_var).epsilon(0.10));
}

TEST_CASE("Provided momentum is not overwritten by Boltzmann seeding",
		  "[SimManager][Langevin][momentum]") {
	initialize_backend_once();

	const int num_particles = 128;
	const float mass = 12.0f;
	const Vector3 given(1.5f, -2.5f, 0.75f);
	std::vector<Resource> resources = {Resource(::Global::single_resource_id)};
	SimSystem sys(resources);
	std::vector<ParticleIO> particles;
	configure_free_system(sys, particles, IntegratorType::Langevin, given, num_particles, mass);

	SimManager manager(sys);
	manager.set_initial_particles(particles);
	REQUIRE_NOTHROW(manager.init());

	const std::vector<Vector3> mom = gather_momenta(sys, num_particles);
	for (const auto& p : mom) {
		REQUIRE(p.x == Approx(given.x));
		REQUIRE(p.y == Approx(given.y));
		REQUIRE(p.z == Approx(given.z));
	}
}

TEST_CASE("Brownian dynamics is not Boltzmann-seeded", "[SimManager][Brownian][momentum]") {
	initialize_backend_once();

	const int num_particles = 128;
	const float mass = 12.0f;
	std::vector<Resource> resources = {Resource(::Global::single_resource_id)};
	SimSystem sys(resources);
	std::vector<ParticleIO> particles;
	configure_free_system(sys,
						  particles,
						  IntegratorType::Brownian,
						  Vector3(0.0f),
						  num_particles,
						  mass);

	SimManager manager(sys);
	manager.set_initial_particles(particles);
	REQUIRE_NOTHROW(manager.init());

	const std::vector<Vector3> mom = gather_momenta(sys, num_particles);
	for (const auto& p : mom) {
		REQUIRE(p.length2() == Approx(0.0));
	}
}
