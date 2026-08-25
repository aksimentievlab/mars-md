/**
 * @file Nonbonded.cpp
 * @brief Nonbonded pair potentials: Debye-Huckel, ONC, Gaussian, bare
 *        Coulomb and softcore.
 *
 * @details References are closed-form in double; never difference mars_real.
 * @see Nonbonded.md
 */

#include "../catch_boiler.h"
#include "Backend/Kernels.h"
#include "Constants.h"
#include "Interactions/Nonbonded/Columb.h"
#include "Interactions/Nonbonded/Pairwise.h"
#include "Objects/DeviceParticleManager.h"
#include "System/PeriodicBox.h"
#include <cmath>
#include <utility>
#include <vector>
using namespace MARS;
using Catch::Approx;

namespace {

constexpr double kLambda = 10.0, kEpsilon = 80.0;
constexpr double kKappa = 0.1, kSz = 80.0, kZ = 6.86;
constexpr double kAmp = 2.5, kSigma = 3.0;

double dh_energy(double r, double qi, double qj) {
	return qi * qj * constants::COULOMB / kEpsilon * std::exp(-r / kLambda) / r;
}

double onc_dielectric(double r) {
	const double e = std::exp(r / kZ);
	const double d = e - 1.0;
	return kSz * (1.0 - (r * r) / (kZ * kZ) * e / (d * d));
}

double onc_energy(double r, double qi, double qj) {
	return qi * qj * constants::COULOMB * std::exp(-kKappa * r) / (onc_dielectric(r) * r);
}

double gauss_energy(double r) {
	return kAmp * std::exp(-(r * r) / (kSigma * kSigma));
}

/// -dU/dr of a double-precision reference.
template<typename F>
double fd_force(F u, double r) {
	const double h = 1e-5 * std::max(1.0, r);
	return -(u(r + h) - u(r - h)) / (2.0 * h);
}

/**
 * @brief Helper to create simple 2-particle system
 */
std::pair<HostParticleData, std::vector<ParticleType>> create_two_particles(mars_real separation) {
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

TEST_CASE("Debye-Huckel energy matches closed form", "[nonbonded][electrostatics][debye]") {
	DebyeHuckelPotential dh{mars_real(kLambda), mars_real(kEpsilon)};
	for (double r : {2.0, 5.0, 12.0, 25.0}) {
		REQUIRE(dh.compute(mars_real(r), 1.0f, -1.0f).energy ==
				Approx(dh_energy(r, 1.0, -1.0)).epsilon(1e-5));
	}
}

TEST_CASE("Debye-Huckel force is -dU/dr", "[nonbonded][electrostatics][debye]") {
	DebyeHuckelPotential dh{mars_real(kLambda), mars_real(kEpsilon)};
	for (double r : {2.0, 5.0, 12.0, 25.0}) {
		const double ref = fd_force([](double x) { return dh_energy(x, 1.0, -1.0); }, r);
		REQUIRE(dh.compute(mars_real(r), 1.0f, -1.0f).force_magnitude == Approx(ref).epsilon(1e-4));
	}
}

TEST_CASE("Debye-Huckel decays faster than bare Coulomb", "[nonbonded][electrostatics][debye]") {
	DebyeHuckelPotential dh{mars_real(kLambda), mars_real(kEpsilon)};
	for (double r : {1.0, 10.0, 30.0}) {
		REQUIRE(dh.compute(mars_real(r), 1.0f, 1.0f).energy < constants::COULOMB / kEpsilon / r);
	}
}

TEST_CASE("ONC dielectric rises toward its plateau", "[nonbonded][electrostatics][onc]") {
	OnckElecPotential onc{mars_real(kKappa), mars_real(kSz), mars_real(kZ)};

	double prev = onc.dielectric(mars_real(0.5));
	for (double r = 1.0; r <= 40.0; r += 1.0) {
		const double eps = onc.dielectric(mars_real(r));
		REQUIRE(eps > prev);
		REQUIRE(eps <= kSz);
		prev = eps;
	}
	// Approach is asymptotic and slow: still ~1.2% short at r=60, so check far out.
	REQUIRE(onc.dielectric(mars_real(120)) == Approx(kSz).epsilon(1e-3));
}

TEST_CASE("ONC dielectric matches closed form", "[nonbonded][electrostatics][onc]") {
	OnckElecPotential onc{mars_real(kKappa), mars_real(kSz), mars_real(kZ)};
	for (double r : {1.0, 3.0, 7.0, 15.0, 30.0}) {
		REQUIRE(onc.dielectric(mars_real(r)) == Approx(onc_dielectric(r)).epsilon(1e-4));
	}
}

TEST_CASE("ONC dielectric_deriv matches finite difference", "[nonbonded][electrostatics][onc]") {
	OnckElecPotential onc{mars_real(kKappa), mars_real(kSz), mars_real(kZ)};
	for (double r : {1.0, 3.0, 7.0, 15.0}) {
		const double h = 1e-5 * std::max(1.0, r);
		const double fd = (onc_dielectric(r + h) - onc_dielectric(r - h)) / (2.0 * h);
		REQUIRE(fd > 0.0); // eps(r) is increasing
		REQUIRE(onc.dielectric_deriv(mars_real(r)) == Approx(fd).epsilon(1e-3));
	}
}

TEST_CASE("ONC energy matches closed form", "[nonbonded][electrostatics][onc]") {
	OnckElecPotential onc{mars_real(kKappa), mars_real(kSz), mars_real(kZ)};
	for (double r : {2.0, 5.0, 12.0}) {
		REQUIRE(onc.compute(mars_real(r), 1.0f, -1.0f).energy ==
				Approx(onc_energy(r, 1.0, -1.0)).epsilon(1e-4));
	}
}

TEST_CASE("ONC force is -dU/dr", "[nonbonded][electrostatics][onc]") {
	OnckElecPotential onc{mars_real(kKappa), mars_real(kSz), mars_real(kZ)};
	for (double r : {2.0, 5.0, 12.0}) {
		const double ref = fd_force([](double x) { return onc_energy(x, 1.0, -1.0); }, r);
		REQUIRE(onc.compute(mars_real(r), 1.0f, -1.0f).force_magnitude ==
				Approx(ref).epsilon(1e-3));
	}
}

TEST_CASE("ONC floors the singularity at r=0", "[nonbonded][electrostatics][onc]") {
	OnckElecPotential onc{mars_real(kKappa), mars_real(kSz), mars_real(kZ)};
	const auto fe = onc.compute(mars_real(0), 1.0f, 1.0f);
	REQUIRE(std::isfinite(fe.energy));
	REQUIRE(std::isfinite(fe.force_magnitude));
	REQUIRE(fe.energy == Approx(onc.compute(OnckElecPotential::MIN_DISTANCE, 1.0f, 1.0f).energy));
}

TEST_CASE("Gaussian energy matches closed form", "[nonbonded][gaussian]") {
	GaussianPotential g{mars_real(kAmp), mars_real(kSigma), mars_real(100)};
	for (double r : {0.0, 1.0, 3.0, 6.0}) {
		REQUIRE(g.compute(mars_real(r)).energy == Approx(gauss_energy(r)).epsilon(1e-5));
	}
}

TEST_CASE("Gaussian force is -dU/dr", "[nonbonded][gaussian]") {
	GaussianPotential g{mars_real(kAmp), mars_real(kSigma), mars_real(100)};
	for (double r : {0.5, 1.0, 3.0, 6.0}) {
		REQUIRE(g.compute(mars_real(r)).force_magnitude ==
				Approx(fd_force(gauss_energy, r)).epsilon(1e-4));
	}
}

TEST_CASE("Gaussian truncates at cutoff", "[nonbonded][gaussian]") {
	GaussianPotential g{mars_real(kAmp), mars_real(kSigma), mars_real(25)}; // cutoff 5

	REQUIRE(g.compute(mars_real(4.9)).energy > 0.0f);
	REQUIRE(g.compute(mars_real(5.1)).energy == 0.0f);
	REQUIRE(g.compute(mars_real(5.1)).force_magnitude == 0.0f);

	// r=12 not 50: by 50 the Gaussian underflows float to a true zero.
	GaussianPotential uncapped{mars_real(kAmp), mars_real(kSigma), mars_real(0)};
	REQUIRE(uncapped.compute(mars_real(12)).energy > 0.0f);
	REQUIRE(g.compute(mars_real(12)).energy == 0.0f);
}

// ============================================================================
// BARE COULOMB AND SOFTCORE
// Merged from the former Nonbonded.cpp; see Nonbonded.md
// ============================================================================

TEST_CASE("Coulomb energy matches closed form", "[nonbonded][electrostatics][coulomb]") {
	const double r = 2.0, qi = 1.0, qj = 1.0;
	const ScalarForceEnergy fe =
		ColumbPotential::compute(Vector3(float(r), 0.0f, 0.0f), float(r), float(qi), float(qj));

	REQUIRE(fe.energy == Approx(qi * qj * constants::COULOMB / r).epsilon(1e-5));
}

TEST_CASE("Coulomb force is -dU/dr", "[nonbonded][electrostatics][coulomb]") {
	const double qi = 1.0, qj = 1.0;
	auto u = [&](double r) { return qi * qj * constants::COULOMB / r; };

	for (double r : {1.5, 2.0, 5.0}) {
		INFO("r = " << r);
		const ScalarForceEnergy fe =
			ColumbPotential::compute(Vector3(float(r), 0.0f, 0.0f), float(r), float(qi), float(qj));
		REQUIRE(fe.force_magnitude == Approx(fd_force(u, r)).epsilon(1e-4));
	}
}

TEST_CASE("Coulomb kernel repels like charges and attracts opposite",
		  "[nonbonded][electrostatics][coulomb][kernel]") {
	initialize_backend_once();
	Resource res(Global::single_resource_id);

	// Particle 0 at the origin, particle 1 at +x, so a repulsive pair must
	// push 0 toward -x. The kernel previously applied this backwards.
	auto [host_data, types] = create_two_particles(2.0f);
	types.push_back(types[0]);
	types[0].charge = 1.0f;

	const float expected = constants::COULOMB * 1.0f / (2.0f * 2.0f);

	auto run = [&](float q1) {
		types[1].charge = q1;
		HostParticleData data = host_data;
		data.type_id = {0, 1};

		DeviceParticle particles(2, res);
		particles.copy_from_host(data, 2);
		particles.clear_forces();
		DeviceParticleTypes type_manager(types, res);

		PeriodicBox pbox_host(Vector3(100.0f, 100.0f, 100.0f));
		DeviceBuffer<PeriodicBox> pbox_buffer(1, res);
		pbox_buffer.copy_from_host(&pbox_host, 1);

		const MARS::int2 pair_host{0, 1};
		DeviceBuffer<MARS::int2> pairs(1, res);
		pairs.copy_from_host(&pair_host, 1);

		ColumbForceKernel kernel{particles.view(),
								 type_manager.view(),
								 pairs.data(),
								 pbox_buffer.data(),
								 1.0f,
								 1};
		launch_kernel(res, KernelConfig::for_1d(1, res), kernel).wait();

		HostParticleData result;
		particles.copy_to_host(result, 2);
		return std::pair<float, float>{result.force[0].x, result.force[1].x};
	};

	auto [like0, like1] = run(1.0f);
	REQUIRE(like0 == Approx(-expected).epsilon(0.01));
	REQUIRE(like1 == Approx(expected).epsilon(0.01));
	REQUIRE(like0 + like1 == Approx(0.0f).margin(1e-3));

	auto [opp0, opp1] = run(-1.0f);
	REQUIRE(opp0 == Approx(expected).epsilon(0.01));
	REQUIRE(opp1 == Approx(-expected).epsilon(0.01));
	REQUIRE(opp0 + opp1 == Approx(0.0f).margin(1e-3));
}

TEST_CASE("Softcore energy is a shifted well inside the core, zero outside",
		  "[nonbonded][softcore]") {
	const float eps = 1.0f;
	const float rad6 = 1.0f;

	const ScalarForceEnergy inside =
		SoftcoreForceKernel::softcoreForce(Vector3(0.8f, 0.0f, 0.0f), eps, rad6);
	REQUIRE(inside.energy > 0.0f);

	const ScalarForceEnergy outside =
		SoftcoreForceKernel::softcoreForce(Vector3(1.5f, 0.0f, 0.0f), eps, rad6);
	REQUIRE(outside.energy == 0.0f);
}

TEST_CASE("Softcore force magnitude grows as particles are pushed together",
		  "[nonbonded][softcore]") {
	const float eps = 1.0f;
	const float rad6 = 1.0f;

	// softcoreForce returns dU/dr divided by r - it is meant to scale r_ij
	// rather than a unit vector - which is the opposite sign convention to the
	// tabulated path's -dU/dr. Only the magnitude is asserted; see Nonbonded.md.
	const float f_near = std::abs(
		SoftcoreForceKernel::softcoreForce(Vector3(0.7f, 0.0f, 0.0f), eps, rad6).force_magnitude);
	const float f_far = std::abs(
		SoftcoreForceKernel::softcoreForce(Vector3(0.9f, 0.0f, 0.0f), eps, rad6).force_magnitude);

	REQUIRE(f_near > f_far);
	REQUIRE(f_far > 0.0f);
}
