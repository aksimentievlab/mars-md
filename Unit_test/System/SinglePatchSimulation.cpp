#include "../catch_boiler.h"
#include "Objects/ParticleProperties.h"
#include "SimManager.h"
#include "System/PatchManager.h"
#include "System/SimSystem.h"
#include "System/SystemState.h"
#include <cmath>
#include <vector>

using namespace MARS;
using namespace Tests;
using Catch::Approx;

/**
 * @brief End-to-end smoke test for the single-patch/single-GPU pipeline:
 * SimSystem -> SimManager::init() (decompose + distribute particles) ->
 * SimManager::run() (force calc + integration) -> Patch device storage.
 *
 * Free (grid-less, pairwise-force-less) Brownian dynamics is used so the
 * expected behavior is pure diffusion: MSD = 6*D*t. This does not re-validate
 * the BD kernel's statistics (see Compute/BDKernels.cpp for that) - it
 * validates that SimManager/SimSystem/PatchManager/Patch actually wire
 * particle data and the integrator together correctly for one patch on one
 * resource.
 */
TEST_CASE("Single patch single GPU BD simulation runs end-to-end",
		  "[SimManager][SingleResource][integration]") {
	initialize_backend_once();

	std::vector<Resource> resources = {Resource(::Global::single_resource_id)};
	SimSystem sys(resources);

	const float box = 400.0f; // large relative to expected displacement to avoid wrapping
	sys.set_box_size(box, box, box);
	sys.set_periodicity(true, true, true);
	sys.set_temperature(300.0f);
	sys.set_cutoff(10.0f);
	sys.set_pairlist_cutoff(20.0f);
	sys.set_timestep(2e-5f); // ns
	const int num_steps = 2000;
	sys.set_num_steps(num_steps);
	sys.set_output_period(500.0f);
	sys.set_energy_output_period(500.0f);
	sys.set_output_name("single_patch_smoke_test");
	sys.set_estimated_particles(256);
	sys.set_particle_integrator_type(IntegratorType::Brownian);

	ParticleType ptype("Ar");
	ptype.mass = 39.948f;
	const float D = 149.0f; // Angstrom^2/ns
	ptype.diffusion = Vector3(D, D, D);
	sys.add_particle_type(ptype);

	const int num_particles = 200;
	std::vector<ParticleIO> particles;
	particles.reserve(num_particles);
	for (int i = 0; i < num_particles; ++i) {
		ParticleIO p;
		p.id = i;
		p.type_name = "Ar";
		p.position = Vector3(box * 0.5f, box * 0.5f, box * 0.5f);
		p.momentum = Vector3(0.0f, 0.0f, 0.0f);
		p.force = Vector3(0.0f, 0.0f, 0.0f);
		p.energy = 0.0f;
		particles.push_back(p);
	}

	SimManager manager(sys);
	manager.set_initial_particles(particles);

	REQUIRE_NOTHROW(manager.init());

	// --- Verify decomposition + particle distribution wiring ---
	REQUIRE(sys.has_patch_manager());
	auto* patch_manager = sys.get_patch_manager();
	REQUIRE(patch_manager->get_num_patches() == 1);
	REQUIRE(patch_manager->get_patch(0).get_particle_count() == static_cast<idx_t>(num_particles));

	// --- Run the simulation loop (force calc + integration for num_steps) ---
	REQUIRE_NOTHROW(manager.run());

	// --- Verify particles actually moved via diffusion ---
	HostParticleData final_data;
	patch_manager->get_patch(0).copy_particles_to_host(final_data, 0, num_particles);
	REQUIRE(final_data.size() == static_cast<size_t>(num_particles));

	const float total_time = sys.get_timestep() * num_steps;
	const float expected_msd = 6.0f * D * total_time;

	float msd = 0.0f;
	for (int i = 0; i < num_particles; ++i) {
		Vector3 disp = final_data.pos[i] - Vector3(box * 0.5f, box * 0.5f, box * 0.5f);
		// Minimum-image correction in case a particle wrapped across the periodic box
		disp.x -= box * roundf(disp.x / box);
		disp.y -= box * roundf(disp.y / box);
		disp.z -= box * roundf(disp.z / box);

		REQUIRE(std::isfinite(disp.x));
		REQUIRE(std::isfinite(disp.y));
		REQUIRE(std::isfinite(disp.z));

		msd += disp.length2();
	}
	msd /= static_cast<float>(num_particles);

	INFO("Expected MSD = " << expected_msd << " (D=" << D << ", t=" << total_time << ")");
	INFO("Measured MSD = " << msd);

	// Loose tolerance: this is a wiring smoke test, not a statistics test.
	REQUIRE(msd == Approx(expected_msd).epsilon(0.35));
}
