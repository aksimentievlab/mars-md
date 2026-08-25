#pragma once
#include "Header.h"
#include "Interactions.h"
#include "Types/Types.h"

namespace MARS {

struct TabulatedPotential {
	DEVICE_PTR(mars_real) __restrict__ pot; // Device pointer to potential table
	mars_real step_inv;						// 1/step_size for interpolation
	unsigned int size;						// Table size
	mars_real start;						// Starting value (e.g., 0 for bonds, -PI for dihedrals)
	bool is_periodic;						// Whether to wrap around (dihedrals)

	/**
	 * @usage:
	 * const ScalarForceEnergy fe = TabulatedPotential::compute(geom.distance, &tables[i]);
	 */
	DEVICE static inline ScalarForceEnergy compute(mars_real dx, const TabulatedPotential* table) {
		// Table lookup with linear interpolation
		mars_real w = (dx - table->start) * table->step_inv;
		int home = static_cast<int>(floorf(w));
		w = w - home;

		// Handle periodic and non-periodic boundaries
		if (table->is_periodic) {
			home %= static_cast<int>(table->size);
			if (home < 0) {
				home += static_cast<int>(table->size);
			}
		} else {
			if (home >= static_cast<int>(table->size) - 1) {
				return ScalarForceEnergy{
					Vec2<mars_real>{mars_real(0), table->pot[table->size - 1]}};
			}

			if (home < 0) {
				home = 0;
			}
		}

		int home1 = table->is_periodic ? (home + 1) % static_cast<int>(table->size) : home + 1;
		if (!table->is_periodic && home1 >= static_cast<int>(table->size)) {
			home1 = table->size - 1;
		}

		mars_real U0 = table->pot[home];
		mars_real dU = table->pot[home1] - U0;

		return ScalarForceEnergy{
			Vec2<mars_real>{-dU * table->step_inv, dU * w + U0}}; // Force magnitude, energy
	}
};
} // namespace MARS
