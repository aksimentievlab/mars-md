#include "../catch_boiler.h"

#include "Backend/Resource.h"
#include "Objects/DeviceParticleManager.h"
#include "PatchOperation/Integrator.h"

using namespace ARBD;
using Catch::Approx;
TEST_CASE("RandomTest", "[free][random]") {
	const int N = 100000;
	std::vector<float> samples_x, samples_y, samples_z;

	openrand::Philox rng(12345, 0);

	for (int i = 0; i < N; i++) {
		openrand::float4 uniform = rng.draw_float4();

		// Box-Muller transform
		float r1 = sqrtf(-2.0f * logf(uniform.x));
		float theta1 = 2.0f * 3.1415926535f * uniform.y;
		float r2 = sqrtf(-2.0f * logf(uniform.z));
		float theta2 = 2.0f * 3.1415926535f * uniform.w;

		samples_x.push_back(r1 * cosf(theta1));
		samples_y.push_back(r1 * sinf(theta1));
		samples_z.push_back(r2 * cosf(theta2));
	}

	// Calculate mean and variance
	auto calc_stats = [](const std::vector<float>& v) {
		double mean = 0.0, var = 0.0;
		for (float x : v)
			mean += x;
		mean /= v.size();
		for (float x : v)
			var += (x - mean) * (x - mean);
		var /= v.size();
		return std::make_pair(mean, var);
	};

	auto [mean_x, var_x] = calc_stats(samples_x);
	auto [mean_y, var_y] = calc_stats(samples_y);
	auto [mean_z, var_z] = calc_stats(samples_z);

	std::cout << "X: mean=" << mean_x << ", var=" << var_x << ", std=" << sqrt(var_x) << std::endl;
	std::cout << "Y: mean=" << mean_y << ", var=" << var_y << ", std=" << sqrt(var_y) << std::endl;
	std::cout << "Z: mean=" << mean_z << ", var=" << var_z << ", std=" << sqrt(var_z) << std::endl;

	// Expected: mean ≈ 0, var ≈ 1, std ≈ 1
	REQUIRE(mean_x == Approx(0.0).margin(0.01));
	REQUIRE(var_x == Approx(1.0).margin(0.02));
	REQUIRE(var_y == Approx(1.0).margin(0.02));
	REQUIRE(var_z == Approx(1.0).margin(0.02));
}
TEST_CASE("IntegratorTest", "[free][bd]") {
	initialize_backend_once();
	Resource res(Global::single_resource_id);

	// Create particle type
	ParticleType ptype("Ar");
	ptype.mass = 39.948f; // amu (Argon)

	float D = 149.0f;					// Å²/ns (calculated as kT/(gamma*mass))
	ptype.diffusion = Vector3(D, D, D); // Å²/ns ✓

	std::vector<ParticleType> types = {ptype};
	DeviceParticleTypes device_types(types, res);

	// Setup 100 particles at center
	DeviceParticle particles(100, res);
	HostParticleData init;
	init.resize(100);
	for (int i = 0; i < 100; i++) {
		init.pos[i] = Vector3(50.0f, 50.0f, 50.0f);
		init.type_id[i] = 0;
		init.force[i] = Vector3(0, 0, 0); // Zero force
	}
	particles.copy_from_host(init, 100);

	// Simulation parameters
	float dt = 2e-5f; // 20 fs in ns
	float temperature = 300.0f;
	float kT = ARBD::constants::BOLTZMANN * temperature; // 0.596 kcal/mol
	int num_steps = 10000;								 // 0.2 ns total
	Vector3 box_size(100.0f, 100.0f, 100.0f);

	// Expected MSD = 6*D*t
	float total_time = dt * num_steps;
	float expected_msd = 6.0f * D * total_time;

	std::cout << "=== Test Parameters ===" << std::endl;
	std::cout << "Diffusion: " << D << " Å²/ns" << std::endl;
	std::cout << "Timestep: " << dt << " ns = " << dt * 1e6 << " fs" << std::endl;
	std::cout << "Total time: " << total_time << " ns" << std::endl;
	std::cout << "Expected MSD = " << expected_msd << " Å²" << std::endl;

	// Run integration
	auto particle_view = particles.view();
	auto type_view = device_types.view();

	for (int step = 0; step < num_steps; step++) {
		launch_BD<float>(res, particle_view, type_view, dt, step, kT, 100, box_size, 5, step);
	}

	// Calculate MSD
	HostParticleData final;
	particles.copy_to_host(final, 100);

	float msd = 0.0f;
	for (int i = 0; i < 100; i++) {
		Vector3 disp = final.pos[i] - init.pos[i];

		// Apply minimum image convention for PBC (use roundf!)
		disp.x -= box_size.x * roundf(disp.x / box_size.x); // ✓
		disp.y -= box_size.y * roundf(disp.y / box_size.y); // ✓
		disp.z -= box_size.z * roundf(disp.z / box_size.z); // ✓

		msd += disp.length2();
		std::cout << "final pos: " << final.pos[i].x << ", " << final.pos[i].y << ", "
				  << final.pos[i].z << std::endl;
		std::cout << final.force[i].x << ", " << final.force[i].y << ", " << final.force[i].z
				  << std::endl;
	}
	msd /= 100.0f;

	std::cout << "Measured MSD = " << msd << " Å²" << std::endl;
	std::cout << "Ratio = " << msd / expected_msd << std::endl;

	REQUIRE(msd == Approx(expected_msd).epsilon(0.2));
}
