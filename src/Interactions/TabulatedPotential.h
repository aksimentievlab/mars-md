#pragma once
#include "Header.h"
#include "Interactions.h"
#include "Types/Types.h"

namespace ARBD {

struct TabulatedPotential {
	DEVICE_PTR(arbd_real) __restrict__ pot; // Device pointer to potential table
	arbd_real step_inv;						// 1/step_size for interpolation
	unsigned int size;						// Table size
	arbd_real start;						// Starting value (e.g., 0 for bonds, -PI for dihedrals)
	bool is_periodic;						// Whether to wrap around (dihedrals)

	/**
	 * @usage:
	 * const ScalarForceEnergy fe = TabulatedPotential::compute(geom.distance, &tables[i]);
	 */
	KERNEL_FUNC static inline ScalarForceEnergy compute(arbd_real dx,
														const TabulatedPotential* table) {
		// Table lookup with linear interpolation
		arbd_real w = (dx - table->start) * table->step_inv;
		int home = static_cast<int>(floorf(w));
		w = w - home;

		// Handle periodic and non-periodic boundaries
		if (table->is_periodic) {
			home %= static_cast<int>(table->size);
			if (home < 0) {
				home += static_cast<int>(table->size);
			}
		} else {
			// Past the end of the table the potential is held at its final
			// value, so the force there is zero. Matches legacy ARBD's
			// TabulatedPotential::compute, which returns
			// `EnergyForce(v0[n-1], Vector3(0.0f))` for `home >= n` and yields
			// `du = 0` in the final bin.
			//
			// Clamping `home` alone (the previous behaviour here) is not
			// enough: `w` keeps growing with `dx`, so `dU*w + U0` below
			// linearly extrapolates the last bin's slope instead of holding the
			// endpoint value, and the returned force stays at that slope rather
			// than falling to zero. For a nonbonded table that ends at the
			// interaction cutoff, every out-of-range pair then acquires a
			// spurious constant force and an unbounded energy - a non-decaying
			// long-range interaction applied to whatever fraction of the pair
			// list lies beyond the cutoff.
			if (home >= static_cast<int>(table->size) - 1) {
				return ScalarForceEnergy{
					Vec2<arbd_real>{arbd_real(0), table->pot[table->size - 1]}};
			}
			// Below the table, clamp the index but keep the fractional weight,
			// so the first bin's slope still applies. Legacy does the same
			// (`home = home < 0 ? 0 : home`). Returning zero force here instead
			// would delete the repulsive core that keeps particles apart.
			if (home < 0) {
				home = 0;
			}
		}

		int home1 = table->is_periodic ? (home + 1) % static_cast<int>(table->size) : home + 1;
		if (!table->is_periodic && home1 >= static_cast<int>(table->size)) {
			home1 = table->size - 1;
		}

		arbd_real U0 = table->pot[home];
		arbd_real dU = table->pot[home1] - U0;

		// force_magnitude must be -dU/dr (not the raw energy slope): the
		// bond/angle/dihedral/nonbonded computers all apply it via the same
		// "-force to particle x, +force to particle y" convention that
		// AnalyticalForceComputer uses with an already-negated force (e.g.
		// harmonic bond's force = -k*(distance-r0)). Returning +dU/dr here
		// inverted every tabulated potential well into a force hilltop.
		return ScalarForceEnergy{
			Vec2<arbd_real>{-dU * table->step_inv, dU * w + U0}}; // Force magnitude, energy
	}
};
} // namespace ARBD
