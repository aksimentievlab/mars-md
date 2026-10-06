/**
 * @file RestartContinuation.cpp
 * @brief Native restart must continue the stochastic stream: a run split at step N
 *        and resumed from the written .restart with firstStep N reproduces the
 *        continuous run; resuming with firstStep 0 (old behavior) does not.
 */

#include "../catch_boiler.h"
#include "IO/ConfigParser.h"
#include "Objects/ParticleProperties.h"
#include "SimManager.h"
#include "System/PatchManager.h"
#include "System/SimSystem.h"

#include <charconv>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <vector>

using namespace MARS;
using namespace Tests;
using Catch::Approx;

namespace {

constexpr int kParticles = 64;
constexpr int kSegment = 50;  // steps per segment
constexpr float kBox = 400.0f; // no wrapping over 2*kSegment steps

/// Distinct start positions near the box center.
std::vector<Vector3> start_positions() {
	std::vector<Vector3> p;
	p.reserve(kParticles);
	for (int i = 0; i < kParticles; ++i) {
		p.emplace_back(180.0f + 10.0f * float(i % 4),
					   180.0f + 10.0f * float((i / 4) % 4),
					   180.0f + 10.0f * float(i / 16));
	}
	return p;
}

/// Free BD (no forces) on one patch, reorder off so particle index == id.
std::map<int, Vector3> run_segment(const std::string& name,
								   size_t first_step,
								   int steps,
								   const std::vector<Vector3>& start) {
	std::vector<Resource> resources = {Resource(::Global::single_resource_id)};
	SimSystem sys(resources);
	sys.set_box_size(kBox, kBox, kBox);
	sys.set_periodicity(true, true, true);
	sys.set_temperature(300.0f);
	sys.set_cutoff(10.0f);
	sys.set_pairlist_cutoff(10.0f);
	sys.set_timestep(2e-5f);
	sys.set_num_steps(steps);
	sys.set_first_step(first_step);
	sys.set_reorder_period(0);
	sys.set_output_period(static_cast<float>(steps));
	sys.set_energy_output_period(static_cast<float>(steps));
	sys.set_output_name(name);
	sys.set_estimated_particles(kParticles);
	sys.set_particle_integrator_type(IntegratorType::Brownian);

	ParticleType ptype("Ar");
	ptype.mass = 39.948f;
	ptype.diffusion = Vector3(149.0f, 149.0f, 149.0f);
	sys.add_particle_type(ptype);

	std::vector<ParticleIO> particles;
	for (int i = 0; i < kParticles; ++i) {
		ParticleIO p;
		p.id = i;
		p.type_name = "Ar";
		p.position = start[i];
		p.momentum = Vector3(0.0f);
		p.force = Vector3(0.0f);
		p.energy = 0.0f;
		particles.push_back(p);
	}

	SimManager manager(sys);
	manager.set_initial_particles(particles);
	manager.init();
	manager.run();

	HostParticleData out;
	sys.get_patch_manager()->get_patch(0).copy_particles_to_host(out, 0, kParticles);
	std::map<int, Vector3> by_id;
	for (size_t i = 0; i < out.size(); ++i)
		by_id[static_cast<int>(out.global_id[i])] = out.pos[i];
	return by_id;
}

/// Positions from a native .restart ("type x y z" per line), in file order.
std::vector<Vector3> read_restart(const std::string& path) {
	std::ifstream in(path);
	REQUIRE(in.good());
	std::vector<Vector3> pos;
	std::string line;
	while (std::getline(in, line)) {
		std::vector<std::string> tok;
		size_t a = 0;
		while (a < line.size()) {
			const size_t b = line.find(' ', a);
			const size_t e = b == std::string::npos ? line.size() : b;
			if (e > a)
				tok.push_back(line.substr(a, e - a));
			a = e + 1;
		}
		REQUIRE(tok.size() == 4);
		float v[3];
		for (int k = 0; k < 3; ++k) {
			const auto& s = tok[k + 1];
			REQUIRE(std::from_chars(s.data(), s.data() + s.size(), v[k]).ec == std::errc{});
		}
		pos.emplace_back(v[0], v[1], v[2]);
	}
	return pos;
}

} // namespace

TEST_CASE("Restart with firstStep continues the BD noise stream",
		  "[SimManager][restart][integration]") {
	initialize_backend_once();

	const std::vector<Vector3> start = start_positions();

	// Reference: one uninterrupted run of 2*kSegment steps.
	const auto continuous = run_segment("restart_test_continuous", 0, 2 * kSegment, start);

	// Segment 1, then resume from the file it wrote.
	const auto seg1 = run_segment("restart_test_seg1", 0, kSegment, start);
	const std::vector<Vector3> restart = read_restart("restart_test_seg1.restart");
	REQUIRE(restart.size() == static_cast<size_t>(kParticles));

	SECTION("the .restart file round-trips positions bit-exactly") {
		for (int i = 0; i < kParticles; ++i) {
			INFO("particle " << i);
			CHECK(restart[i].x == seg1.at(i).x);
			CHECK(restart[i].y == seg1.at(i).y);
			CHECK(restart[i].z == seg1.at(i).z);
		}
	}

	SECTION("firstStep N reproduces the continuous run") {
		const auto resumed = run_segment("restart_test_seg2", kSegment, kSegment, restart);
		for (int i = 0; i < kParticles; ++i) {
			INFO("particle " << i);
			CHECK(resumed.at(i).x == Approx(continuous.at(i).x).margin(1e-4));
			CHECK(resumed.at(i).y == Approx(continuous.at(i).y).margin(1e-4));
			CHECK(resumed.at(i).z == Approx(continuous.at(i).z).margin(1e-4));
		}
	}

	SECTION("firstStep 0 (old behavior) replays segment 1's noise and diverges") {
		const auto replayed = run_segment("restart_test_seg2_replay", 0, kSegment, restart);
		int matching = 0;
		for (int i = 0; i < kParticles; ++i) {
			if ((replayed.at(i) - continuous.at(i)).length() < 1e-2f)
				++matching;
		}
		// Replayed kicks repeat segment 1's displacement exactly.
		for (int i = 0; i < kParticles; ++i) {
			const Vector3 d1 = seg1.at(i) - start[i];
			const Vector3 d2 = replayed.at(i) - restart[i];
			CHECK(d2.x == Approx(d1.x).margin(1e-3));
			CHECK(d2.y == Approx(d1.y).margin(1e-3));
			CHECK(d2.z == Approx(d1.z).margin(1e-3));
		}
		CHECK(matching == 0);
	}
}

TEST_CASE("ConfigParser reads firstStep and rejects negatives", "[parser][restart]") {
	initialize_backend_once();
	std::vector<Resource> resources = {Resource(::Global::single_resource_id)};
	const std::string path = "restart_test_firststep.bd";

	SECTION("firstStep is stored on SimSystem") {
		{
			std::ofstream out(path);
			out << "steps 100\nfirstStep 1500\n";
		}
		SimSystem sys(resources);
		ConfigParser parser(sys, path);
		CHECK(sys.get_first_step() == 1500);
		CHECK(sys.get_num_steps() == 100);
	}

	SECTION("default is 0") {
		{
			std::ofstream out(path);
			out << "steps 100\n";
		}
		SimSystem sys(resources);
		ConfigParser parser(sys, path);
		CHECK(sys.get_first_step() == 0);
	}

	SECTION("negative firstStep throws") {
		{
			std::ofstream out(path);
			out << "steps 100\nfirstStep -5\n";
		}
		SimSystem sys(resources);
		CHECK_THROWS_AS(ConfigParser(sys, path), Exception);
	}

	std::filesystem::remove(path);
}
