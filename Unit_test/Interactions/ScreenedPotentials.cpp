/**
 * @file ScreenedPotentials.cpp
 * @brief Debye-Huckel, ONC electrostatics and Gaussian pair potentials.
 *
 * @details References are closed-form in double; never difference arbd_real.
 */

#include "../catch_boiler.h"
#include "Constants.h"
#include "Interactions/Nonbonded/Columb.h"
#include "Interactions/Nonbonded/Pairwise.h"
#include <cmath>

using namespace ARBD;
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

} // namespace

TEST_CASE("Debye-Huckel energy matches closed form", "[nonbonded][electrostatics][debye]") {
	DebyeHuckelPotential dh{arbd_real(kLambda), arbd_real(kEpsilon)};
	for (double r : {2.0, 5.0, 12.0, 25.0}) {
		REQUIRE(dh.compute(arbd_real(r), 1.0f, -1.0f).energy ==
				Approx(dh_energy(r, 1.0, -1.0)).epsilon(1e-5));
	}
}

TEST_CASE("Debye-Huckel force is -dU/dr", "[nonbonded][electrostatics][debye]") {
	DebyeHuckelPotential dh{arbd_real(kLambda), arbd_real(kEpsilon)};
	for (double r : {2.0, 5.0, 12.0, 25.0}) {
		const double ref = fd_force([](double x) { return dh_energy(x, 1.0, -1.0); }, r);
		REQUIRE(dh.compute(arbd_real(r), 1.0f, -1.0f).force_magnitude == Approx(ref).epsilon(1e-4));
	}
}

TEST_CASE("Debye-Huckel decays faster than bare Coulomb", "[nonbonded][electrostatics][debye]") {
	DebyeHuckelPotential dh{arbd_real(kLambda), arbd_real(kEpsilon)};
	for (double r : {1.0, 10.0, 30.0}) {
		REQUIRE(dh.compute(arbd_real(r), 1.0f, 1.0f).energy < constants::COULOMB / kEpsilon / r);
	}
}

TEST_CASE("ONC dielectric rises toward its plateau", "[nonbonded][electrostatics][onc]") {
	OncElecPotential onc{arbd_real(kKappa), arbd_real(kSz), arbd_real(kZ)};

	double prev = onc.dielectric(arbd_real(0.5));
	for (double r = 1.0; r <= 40.0; r += 1.0) {
		const double eps = onc.dielectric(arbd_real(r));
		REQUIRE(eps > prev);
		REQUIRE(eps <= kSz);
		prev = eps;
	}
	// Approach is asymptotic and slow: still ~1.2% short at r=60, so check far out.
	REQUIRE(onc.dielectric(arbd_real(120)) == Approx(kSz).epsilon(1e-3));
}

TEST_CASE("ONC dielectric matches closed form", "[nonbonded][electrostatics][onc]") {
	OncElecPotential onc{arbd_real(kKappa), arbd_real(kSz), arbd_real(kZ)};
	for (double r : {1.0, 3.0, 7.0, 15.0, 30.0}) {
		REQUIRE(onc.dielectric(arbd_real(r)) == Approx(onc_dielectric(r)).epsilon(1e-4));
	}
}

TEST_CASE("ONC dielectric_deriv matches finite difference", "[nonbonded][electrostatics][onc]") {
	OncElecPotential onc{arbd_real(kKappa), arbd_real(kSz), arbd_real(kZ)};
	for (double r : {1.0, 3.0, 7.0, 15.0}) {
		const double h = 1e-5 * std::max(1.0, r);
		const double fd = (onc_dielectric(r + h) - onc_dielectric(r - h)) / (2.0 * h);
		REQUIRE(fd > 0.0); // eps(r) is increasing
		REQUIRE(onc.dielectric_deriv(arbd_real(r)) == Approx(fd).epsilon(1e-3));
	}
}

TEST_CASE("ONC energy matches closed form", "[nonbonded][electrostatics][onc]") {
	OncElecPotential onc{arbd_real(kKappa), arbd_real(kSz), arbd_real(kZ)};
	for (double r : {2.0, 5.0, 12.0}) {
		REQUIRE(onc.compute(arbd_real(r), 1.0f, -1.0f).energy ==
				Approx(onc_energy(r, 1.0, -1.0)).epsilon(1e-4));
	}
}

TEST_CASE("ONC force is -dU/dr", "[nonbonded][electrostatics][onc]") {
	OncElecPotential onc{arbd_real(kKappa), arbd_real(kSz), arbd_real(kZ)};
	for (double r : {2.0, 5.0, 12.0}) {
		const double ref = fd_force([](double x) { return onc_energy(x, 1.0, -1.0); }, r);
		REQUIRE(onc.compute(arbd_real(r), 1.0f, -1.0f).force_magnitude == Approx(ref).epsilon(1e-3));
	}
}

TEST_CASE("ONC floors the singularity at r=0", "[nonbonded][electrostatics][onc]") {
	OncElecPotential onc{arbd_real(kKappa), arbd_real(kSz), arbd_real(kZ)};
	const auto fe = onc.compute(arbd_real(0), 1.0f, 1.0f);
	REQUIRE(std::isfinite(fe.energy));
	REQUIRE(std::isfinite(fe.force_magnitude));
	REQUIRE(fe.energy == Approx(onc.compute(OncElecPotential::MIN_DISTANCE, 1.0f, 1.0f).energy));
}

TEST_CASE("Gaussian energy matches closed form", "[nonbonded][gaussian]") {
	GaussianPotential g{arbd_real(kAmp), arbd_real(kSigma), arbd_real(100)};
	for (double r : {0.0, 1.0, 3.0, 6.0}) {
		REQUIRE(g.compute(arbd_real(r)).energy == Approx(gauss_energy(r)).epsilon(1e-5));
	}
}

TEST_CASE("Gaussian force is -dU/dr", "[nonbonded][gaussian]") {
	GaussianPotential g{arbd_real(kAmp), arbd_real(kSigma), arbd_real(100)};
	for (double r : {0.5, 1.0, 3.0, 6.0}) {
		REQUIRE(g.compute(arbd_real(r)).force_magnitude ==
				Approx(fd_force(gauss_energy, r)).epsilon(1e-4));
	}
}

TEST_CASE("Gaussian truncates at cutoff", "[nonbonded][gaussian]") {
	GaussianPotential g{arbd_real(kAmp), arbd_real(kSigma), arbd_real(25)}; // cutoff 5

	REQUIRE(g.compute(arbd_real(4.9)).energy > 0.0f);
	REQUIRE(g.compute(arbd_real(5.1)).energy == 0.0f);
	REQUIRE(g.compute(arbd_real(5.1)).force_magnitude == 0.0f);

	// r=12 not 50: by 50 the Gaussian underflows float to a true zero.
	GaussianPotential uncapped{arbd_real(kAmp), arbd_real(kSigma), arbd_real(0)};
	REQUIRE(uncapped.compute(arbd_real(12)).energy > 0.0f);
	REQUIRE(g.compute(arbd_real(12)).energy == 0.0f);
}
