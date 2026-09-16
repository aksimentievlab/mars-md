/**
 * @brief Thermal-equilibration test for the BAOAB Langevin integrator.
 *
 * The seed test (BoltzmannMomentumInit) only checks t=0. This drives the real
 * BAOABIntegrate/BAOAB_LastUpdate kernels on a 3D harmonic oscillator for many
 * steps and asserts equipartition holds:
 *
 *   <KE> = <PE> = (3/2) N kT   (kcal/mol)
 *
 * KE checks the O (Ornstein-Uhlenbeck) step; PE checks that the B (kick) and A
 * (drift) steps couple momentum and position with consistent units. A unit error
 * in the B/A coupling pumps energy - the integrator settles at the wrong
 * temperature or diverges - which is invisible to a formula/finite-difference
 * test but shows up here. Parameters sit in the light-mass / weak-damping regime
 * (mass 100 amu, gamma 20/ns, dt 2e-5 ns) where such a bug bites hardest.
 */
#include "../catch_boiler.h"
#include "Constants.h"
#include "Objects/DeviceParticle.h"
#include "PatchOperation/Integrator/BAOAB.h"
#include "System/PeriodicBox.h"
#include "Types/Vector3.h"
#include <algorithm>
#include <vector>

using namespace MARS;
using namespace Tests;
using Catch::Approx;

TEST_CASE("BAOAB holds equipartition on a harmonic oscillator",
		  "[BAOAB][Langevin][integrator][equilibration]") {
	// --- Physical/regime parameters (residue-bead regime) ---
	const int N = 1000;
	const float mass = 100.0f;			// amu
	const float gamma = 20.0f;			// 1/ns (weak damping)
	const float dt = 2.0e-5f;			// ns
	const float kT = 295.0f * constants::BOLTZMANN; // kcal/mol
	// Spring constant chosen so omega*dt ~ 0.05 (well-resolved, stable):
	// omega^2 = k * FCF * 1e4 / mass, FCF = FORCE_CONVERSION_FACTOR.
	const float k_spring = 1.5f;		// kcal/mol/AA^2

	const int num_steps = 40000;
	const int warmup = 20000; // ~8 damping times (1/gamma = 2500 steps)

	// --- Host particle arrays wired into a ParticleView ---
	std::vector<int> id(N), type_id(N, 0);
	std::vector<uint32_t> flags(N, 0u);
	std::vector<Vector3> pos(N, Vector3(0.0f));
	std::vector<Vector3> mom(N, Vector3(0.0f));
	std::vector<Vector3> force(N, Vector3(0.0f));
	std::vector<Vector3> orient(N, Vector3(1.0f, 0.0f, 0.0f));
	std::vector<Vector3> external_force(N, Vector3(0.0f));
	std::vector<int> attached(N, -1);
	for (int i = 0; i < N; ++i)
		id[i] = i;

	ParticleView pv{};
	pv.id = id.data();
	pv.type_id = type_id.data();
	pv.pos = pos.data();
	pv.mom = mom.data();
	pv.ForceEnergy = force.data();
	pv.orient = orient.data();
	pv.flags = flags.data();
	pv.external_force = external_force.data();
	pv.attached_rigid_body_id = attached.data();

	// --- Single particle type wired into a ParticleTypeView ---
	std::vector<float> type_mass{mass};
	std::vector<Vector3> type_damping{Vector3(gamma, gamma, gamma)};
	ParticleTypeView ptv{};
	ptv.mass = type_mass.data();
	ptv.trans_damping = type_damping.data();

	const PeriodicBox box; // default: non-periodic, wrap() is a no-op
	const uint64_t base_seed = 0xC0FFEEULL;
	const uint32_t base_ctr = 0;
	const float C2 = constants::SQRT_CAL_TO_JOULE * constants::SQRT_CAL_TO_JOULE;

	auto apply_harmonic_force = [&]() {
		for (int i = 0; i < N; ++i)
			force[i] = pos[i] * (-k_spring);
	};

	double ke_sum = 0.0, pe_sum = 0.0;
	double ke_max = 0.0;
	int samples = 0;
	bool deferred_pending = false;

	for (int s = 0; s < num_steps; ++s) {
		// Force at current positions (matches SimManager: force calc precedes
		// integration each step).
		apply_harmonic_force();

		// Pay the previous step's owed closing kick (B) now that the force at the
		// new position is available - mirrors Patch::integrate_motion.
		if (deferred_pending) {
			BAOAB_LastUpdate<float> last(pv,
										 ptv,
										 dt,
										 static_cast<size_t>(s),
										 kT,
										 static_cast<idx_t>(N),
										 base_seed,
										 base_ctr,
										 nullptr,
										 Vector3(0.0f),
										 0);
			for (int i = 0; i < N; ++i)
				last(static_cast<idx_t>(i));
		}

		// Opening B, drift A, O, drift A.
		BAOABIntegrate<float> step(pv,
								   ptv,
								   box,
								   dt,
								   static_cast<size_t>(s),
								   kT,
								   static_cast<idx_t>(N),
								   base_seed,
								   base_ctr,
								   nullptr,
								   Vector3(0.0f),
								   0);
		for (int i = 0; i < N; ++i)
			step(static_cast<idx_t>(i));
		deferred_pending = true;

		if (s >= warmup) {
			double ke = 0.0, pe = 0.0;
			for (int i = 0; i < N; ++i) {
				ke += 0.5 * mom[i].length2() / mass;
				pe += 0.5 * k_spring * pos[i].length2();
			}
			ke /= C2; // recover kcal/mol (SimManager convention)
			ke_sum += ke;
			pe_sum += pe;
			ke_max = std::max(ke_max, ke);
			++samples;
		}
	}

	REQUIRE(samples > 0);
	const double avg_ke = ke_sum / samples;
	const double avg_pe = pe_sum / samples;
	const double expected = 1.5 * N * kT; // (3/2) N kT for 3 DOF

	INFO("expected (3/2)NkT = " << expected << " kcal/mol");
	INFO("avg KE = " << avg_ke << ", avg PE = " << avg_pe);

	// Diverging integrator: KE runs far above equipartition (the real-run symptom
	// was ~4x). Catch it before the ratio checks so the failure message is clear.
	REQUIRE(ke_max < 5.0 * expected);

	// Equipartition on both sides, 8% tolerance (statistical + O((omega*dt)^2) bias).
	REQUIRE(avg_ke == Approx(expected).epsilon(0.08));
	REQUIRE(avg_pe == Approx(expected).epsilon(0.08));
}
