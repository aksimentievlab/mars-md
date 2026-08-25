#include "../catch_boiler.h"

#include "Backend/Resource.h"
#include "Constants.h"
#include "Objects/DeviceParticleManager.h"
#include "PatchOperation/Integrator.h"
#include "System/PeriodicBox.h"

#include <cmath>
#include <vector>

using namespace MARS;
using Catch::Approx;

namespace {

constexpr int kN = 4096;	 // independent walkers; SEM on a variance ~ 2.2%
constexpr float kDt = 2e-5f; // ns
constexpr float kMass = 39.948f;
constexpr float kGamma = 50.0f; // 1/ns, as the NPC configs' transDamping
constexpr float kTemp = 300.0f;

float kT() {
	return constants::BOLTZMANN * kTemp;
}

/// Langevin equilibrium for BAOAB's O step: p' = c p + sqrt(kT m (1-c^2)) U xi
/// is an Ornstein-Uhlenbeck update whose stationary variance is kT m U^2.
float expected_p2() {
	return kT() * kMass * constants::SQRT_CAL_TO_JOULE * constants::SQRT_CAL_TO_JOULE;
}

std::vector<ParticleType> langevin_type() {
	ParticleType p("Ar");
	p.mass = kMass;
	p.trans_damping = Vector3(kGamma, kGamma, kGamma);
	const float D = kT() / (kGamma * kMass);
	p.diffusion = Vector3(D, D, D);
	return {p};
}

HostParticleData at_rest(int n) {
	HostParticleData init;
	init.resize(n);
	for (int i = 0; i < n; ++i) {
		init.pos[i] = Vector3(50.0f, 50.0f, 50.0f);
		init.mom[i] = Vector3(0.0f, 0.0f, 0.0f);
		init.force[i] = Vector3(0.0f, 0.0f, 0.0f);
		init.type_id[i] = 0;
	}
	return init;
}

/// Run BAOAB from rest with zero force. Returns the final host state.
HostParticleData run_baoab(const Resource& res, uint64_t seed, int steps) {
	auto types = langevin_type();
	DeviceParticleTypes device_types(types, res);
	DeviceParticle particles(kN, res);
	particles.copy_from_host(at_rest(kN), kN);

	PeriodicBox box(Vector3(100.0f, 100.0f, 100.0f));
	auto view = particles.view();
	auto tview = device_types.view();
	for (int s = 0; s < steps; ++s) {
		launch_BAOAB<float>(res, view, tview, box, kDt, s, kT(), kN, seed, 0,
							nullptr, Vector3{0.0f, 0.0f, 0.0f},
							1);
	}
	HostParticleData out;
	particles.copy_to_host(out, kN);
	return out;
}

/// Run overdamped BD from a common start. Box is large enough never to wrap.
HostParticleData run_bd(const Resource& res, uint64_t seed, int steps) {
	auto types = langevin_type();
	DeviceParticleTypes device_types(types, res);
	DeviceParticle particles(kN, res);
	particles.copy_from_host(at_rest(kN), kN);

	PeriodicBox box(Vector3(1.0e6f, 1.0e6f, 1.0e6f));
	auto view = particles.view();
	auto tview = device_types.view();
	for (int s = 0; s < steps; ++s) {
		launch_BD<float>(res, view, tview, kDt, s, kT(), kN, box, seed, /*base_ctr=*/0,
						 /*grid_configs=*/nullptr, /*electric_field=*/Vector3{0.0f, 0.0f, 0.0f},
						 /*interpolation_scheme=*/1);
	}
	HostParticleData out;
	particles.copy_to_host(out, kN);
	return out;
}

float component(const Vector3& v, int axis) {
	return (axis == 0) ? v.x : (axis == 1) ? v.y : v.z;
}

float mean_sq(const HostParticleData& d, int axis) {
	double acc = 0.0;
	for (int i = 0; i < kN; ++i) {
		const double c = component(d.mom[i], axis);
		acc += c * c;
	}
	return float(acc / kN);
}

/// Pearson correlation of one momentum component between two runs.
float correlation(const HostParticleData& a, const HostParticleData& b, int axis) {
	double ma = 0, mb = 0;
	for (int i = 0; i < kN; ++i) {
		ma += component(a.mom[i], axis);
		mb += component(b.mom[i], axis);
	}
	ma /= kN;
	mb /= kN;
	double num = 0, va = 0, vb = 0;
	for (int i = 0; i < kN; ++i) {
		const double da = component(a.mom[i], axis) - ma;
		const double db = component(b.mom[i], axis) - mb;
		num += da * db;
		va += da * da;
		vb += db * db;
	}
	if (va <= 0.0 || vb <= 0.0) {
		return 1.0f; // degenerate: treat as fully correlated so the test fails
	}
	return float(num / std::sqrt(va * vb));
}

} // namespace

TEST_CASE("BAOAB reaches the Langevin equilibrium temperature", "[free][baoab][thermostat]") {
	initialize_backend_once();
	Resource res(Global::single_resource_id);

	// gamma*dt = 1e-3, so momentum relaxes over ~1000 steps; 8000 is ~8
	// relaxation times from rest.
	const HostParticleData out = run_baoab(res, /*seed=*/20260816u, /*steps=*/8000);

	const float want = expected_p2();
	INFO("expected <p_i^2> = kT*m*SQRT_CAL_TO_JOULE^2 = " << want);
	for (int axis = 0; axis < 3; ++axis) {
		CAPTURE(axis);
		CHECK(mean_sq(out, axis) == Approx(want).epsilon(0.08));
	}
	CHECK(mean_sq(out, 0) < 10.0f * want);
}

TEST_CASE("Stochastic integrators decorrelate across seeds", "[free][baoab][rng]") {
	initialize_backend_once();
	Resource res(Global::single_resource_id);

	const HostParticleData a = run_baoab(res, 20260816u, 2000);
	const HostParticleData b = run_baoab(res, 20260817u, 2000);

	bool any_differs = false;
	for (int i = 0; i < kN && !any_differs; ++i) {
		any_differs = (a.mom[i].x != b.mom[i].x);
	}
	REQUIRE(any_differs);

	// With kN samples the null correlation has sd ~ 1/sqrt(kN) = 0.016, so 0.15
	// is ~9 sigma from independent, and the aliased case sat essentially at 1.
	for (int axis = 0; axis < 3; ++axis) {
		CAPTURE(axis);
		const float r = correlation(a, b, axis);
		INFO("correlation between adjacent seeds = " << r);
		CHECK(std::abs(r) < 0.15f);
	}
}

TEST_CASE("BD decorrelates across seeds", "[free][bd][rng]") {
	initialize_backend_once();
	Resource res(Global::single_resource_id);

	// BD carries the same Philox fix as BAOAB but nothing pinned it: the
	// existing BDKernels MSD check only needs noise that varies per *step*,
	// which BD had even while adjacent seeds were aliased onto one sequence.
	const HostParticleData a = run_bd(res, 20260816u, 500);
	const HostParticleData b = run_bd(res, 20260817u, 500);

	// Positive control: the walk has to actually move, or the correlation
	// below would be comparing two piles of zeros.
	const float d2 = kT() / (kGamma * kMass) * 2.0f * kDt * 500.0f; // <dx^2> = 2 D t
	double msd = 0.0;
	for (int i = 0; i < kN; ++i) {
		const float dx = a.pos[i].x - 50.0f;
		msd += double(dx) * dx;
	}
	CHECK(float(msd / kN) == Approx(d2).epsilon(0.15));

	for (int axis = 0; axis < 3; ++axis) {
		CAPTURE(axis);
		double ma = 0, mb = 0;
		for (int i = 0; i < kN; ++i) {
			ma += component(a.pos[i], axis);
			mb += component(b.pos[i], axis);
		}
		ma /= kN;
		mb /= kN;
		double num = 0, va = 0, vb = 0;
		for (int i = 0; i < kN; ++i) {
			const double da = component(a.pos[i], axis) - ma;
			const double db = component(b.pos[i], axis) - mb;
			num += da * db;
			va += da * da;
			vb += db * db;
		}
		const float r = float(num / std::sqrt(va * vb));
		INFO("correlation between adjacent seeds = " << r);
		CHECK(std::abs(r) < 0.15f);
	}
}
