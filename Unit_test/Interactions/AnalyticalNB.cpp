#include "../catch_boiler.h"
#include "Backend/Kernels.h"
#include "Constants.h"
#include "Interactions/Nonbonded/AnalyticalPairKernels.h"
#include "Interactions/Nonbonded/Columb.h"
#include "Interactions/Nonbonded/Pairwise.h"
#include "Objects/DeviceParticleManager.h"
#include "System/PeriodicBox.h"
#include <cmath>
#include <utility>
#include <vector>
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

/**
 * @brief Run AnalyticalPairKernel over a single pair and read the result back.
 * @param terms Bitmask of AnalyticalPairTerm values to enable
 * @param separation Distance between the two particles along +x
 * @param q1 Charge of the second particle (the first is always +1)
 */
HostParticleData run_pair(const Resource& res,
						  uint32_t terms,
						  arbd_real separation,
						  float q1 = 1.0f,
						  arbd_real cutoff_squared = arbd_real(0)) {
	auto [host_data, types] = create_two_particles(separation);
	types.push_back(types[0]);
	types[0].charge = 1.0f;
	types[1].charge = q1;
	host_data.type_id = {0, 1};

	DeviceParticle particles(2, res);
	particles.copy_from_host(host_data, 2);
	particles.clear_forces();
	DeviceParticleTypes type_manager(types, res);

	PeriodicBox pbox_host(Vector3(1000.0f, 1000.0f, 1000.0f));
	DeviceBuffer<PeriodicBox> pbox_buffer(1, res);
	pbox_buffer.copy_from_host(&pbox_host, 1);

	const ARBD::int2 pair_host{0, 1};
	DeviceBuffer<ARBD::int2> pairs(1, res);
	pairs.copy_from_host(&pair_host, 1);

	auto view = particles.view();
	AnalyticalPairKernel kernel{};
	kernel.neighbor_pairs = pairs.data();
	kernel.positions = view.pos;
	kernel.force_energy = view.ForceEnergy;
	kernel.type_ids = view.type_id;
	kernel.types = type_manager.view();
	kernel.pbox = pbox_buffer.data();
	kernel.enabled_terms = terms;
	kernel.debye_huckel = DebyeHuckelPotential{arbd_real(kLambda), arbd_real(kEpsilon)};
	kernel.onck = OnckElecPotential{arbd_real(kKappa), arbd_real(kSz), arbd_real(kZ)};
	kernel.gaussian = GaussianPotential{arbd_real(kAmp), arbd_real(kSigma), arbd_real(0)};
	kernel.get_energy = true;
	kernel.num_pairs = 1;
	kernel.cutoff_squared = cutoff_squared;

	launch_analytical_pair_nonbonded(res, kernel, 1).wait();

	HostParticleData result;
	particles.copy_to_host(result, 2, /*need_energy=*/true);
	return result;
}

} // anonymous namespace

// ============================================================================
// Each term, driven through the kernel on real particles and checked against
// the closed-form double-precision references above. See AnalyticalNB.md.
// ============================================================================

TEST_CASE("Coulomb pair matches closed form", "[nonbonded][analytical][coulomb]") {
	initialize_backend_once();
	Resource res(Global::single_resource_id);

	const double r = 2.0;
	auto u = [&](double d) { return constants::COULOMB / d; };
	const HostParticleData out = run_pair(res, PAIR_TERM_COULOMB, arbd_real(r));

	// unit vector points 0 -> 1, and force_magnitude is -dU/dr, applied as
	// -force to particle 0. Like charges therefore push 0 toward -x.
	REQUIRE(out.force[0].x == Approx(-fd_force(u, r)).epsilon(1e-3));
	REQUIRE(out.force[1].x == Approx(fd_force(u, r)).epsilon(1e-3));
	REQUIRE(out.energy[0] + out.energy[1] == Approx(u(r)).epsilon(1e-3));
}

TEST_CASE("Debye-Huckel pair matches closed form", "[nonbonded][analytical][debye]") {
	initialize_backend_once();
	Resource res(Global::single_resource_id);

	const double r = 4.0;
	auto u = [&](double d) { return dh_energy(d, 1.0, 1.0); };
	const HostParticleData out = run_pair(res, PAIR_TERM_DEBYE_HUCKEL, arbd_real(r));

	REQUIRE(out.force[0].x == Approx(-fd_force(u, r)).epsilon(1e-3));
	REQUIRE(out.energy[0] + out.energy[1] == Approx(u(r)).epsilon(1e-3));
}

TEST_CASE("ONC pair matches closed form", "[nonbonded][analytical][onc]") {
	initialize_backend_once();
	Resource res(Global::single_resource_id);

	const double r = 5.0;
	auto u = [&](double d) { return onc_energy(d, 1.0, 1.0); };
	const HostParticleData out = run_pair(res, PAIR_TERM_ONCK, arbd_real(r));

	REQUIRE(out.force[0].x == Approx(-fd_force(u, r)).epsilon(2e-2));
	REQUIRE(out.energy[0] + out.energy[1] == Approx(u(r)).epsilon(1e-3));
}

TEST_CASE("Gaussian pair matches closed form and ignores charge",
		  "[nonbonded][analytical][gaussian]") {
	initialize_backend_once();
	Resource res(Global::single_resource_id);

	const double r = 2.0;
	const HostParticleData out = run_pair(res, PAIR_TERM_GAUSSIAN, arbd_real(r));

	REQUIRE(out.force[0].x == Approx(-fd_force(gauss_energy, r)).epsilon(1e-3));
	REQUIRE(out.energy[0] + out.energy[1] == Approx(gauss_energy(r)).epsilon(1e-3));

	// No charge dependence: flipping the sign of q1 must change nothing.
	const HostParticleData flipped = run_pair(res, PAIR_TERM_GAUSSIAN, arbd_real(r), -1.0f);
	REQUIRE(flipped.force[0].x == Approx(out.force[0].x).epsilon(1e-5));
}

TEST_CASE("Opposite charges attract", "[nonbonded][analytical][coulomb]") {
	initialize_backend_once();
	Resource res(Global::single_resource_id);

	const HostParticleData like = run_pair(res, PAIR_TERM_COULOMB, arbd_real(2), 1.0f);
	const HostParticleData opp = run_pair(res, PAIR_TERM_COULOMB, arbd_real(2), -1.0f);

	REQUIRE(like.force[0].x < 0.0f);           // pushed away from particle 1
	REQUIRE(opp.force[0].x > 0.0f);            // pulled toward particle 1
	REQUIRE(opp.force[0].x == Approx(-like.force[0].x).epsilon(1e-4));
}

TEST_CASE("Analytical pair forces obey Newton's third law",
		  "[nonbonded][analytical]") {
	initialize_backend_once();
	Resource res(Global::single_resource_id);

	const uint32_t all = PAIR_TERM_COULOMB | PAIR_TERM_DEBYE_HUCKEL | PAIR_TERM_ONCK |
						 PAIR_TERM_GAUSSIAN | PAIR_TERM_SOFTCORE;
	const HostParticleData out = run_pair(res, all, arbd_real(2));

	REQUIRE(out.force[0].x + out.force[1].x == Approx(0.0f).margin(1e-3));
	REQUIRE(out.force[0].y + out.force[1].y == Approx(0.0f).margin(1e-6));
	REQUIRE(out.force[0].z + out.force[1].z == Approx(0.0f).margin(1e-6));
}

TEST_CASE("One launch applying two terms equals the sum of each alone",
		  "[nonbonded][analytical][combined]") {
	initialize_backend_once();
	Resource res(Global::single_resource_id);

	const arbd_real r = arbd_real(2);
	const float f_coulomb = run_pair(res, PAIR_TERM_COULOMB, r).force[0].x;
	const float f_gauss = run_pair(res, PAIR_TERM_GAUSSIAN, r).force[0].x;
	const float f_both = run_pair(res, PAIR_TERM_COULOMB | PAIR_TERM_GAUSSIAN, r).force[0].x;

	REQUIRE(f_both == Approx(f_coulomb + f_gauss).epsilon(1e-4));
}

TEST_CASE("An empty term mask leaves forces untouched", "[nonbonded][analytical]") {
	initialize_backend_once();
	Resource res(Global::single_resource_id);

	const HostParticleData out = run_pair(res, PAIR_TERM_NONE, arbd_real(2));
	REQUIRE(out.force[0].x == Approx(0.0f).margin(1e-9));
	REQUIRE(out.force[1].x == Approx(0.0f).margin(1e-9));
	REQUIRE(out.energy[0] == Approx(0.0f).margin(1e-9));
}

TEST_CASE("Pairs beyond the cutoff contribute nothing", "[nonbonded][analytical][cutoff]") {
	initialize_backend_once();
	Resource res(Global::single_resource_id);

	// cutoff 3 keeps a pair at r=2 and drops one at r=5.
	const arbd_real cutoff2 = arbd_real(9);
	const HostParticleData inside =
		run_pair(res, PAIR_TERM_COULOMB, arbd_real(2), 1.0f, cutoff2);
	const HostParticleData outside =
		run_pair(res, PAIR_TERM_COULOMB, arbd_real(5), 1.0f, cutoff2);

	REQUIRE(inside.force[0].x != Approx(0.0f).margin(1e-6));
	REQUIRE(outside.force[0].x == Approx(0.0f).margin(1e-9));
	REQUIRE(outside.energy[0] == Approx(0.0f).margin(1e-9));
}

TEST_CASE("Softcore repels harder as particles are pushed together",
		  "[nonbonded][analytical][softcore]") {
	initialize_backend_once();
	Resource res(Global::single_resource_id);

	// Unit radius and eps for both types, so the core sits at r = 1.
	const float near = run_pair(res, PAIR_TERM_SOFTCORE, arbd_real(0.7)).force[0].x;
	const float far = run_pair(res, PAIR_TERM_SOFTCORE, arbd_real(0.9)).force[0].x;

	// Inside the core the pair repels, so particle 0 is pushed toward -x, and
	// the push grows as they close.
	REQUIRE(near < 0.0f);
	REQUIRE(far < 0.0f);
	REQUIRE(near < far);
}
