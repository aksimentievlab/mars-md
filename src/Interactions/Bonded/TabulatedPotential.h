#pragma once
#include "../Interactions.h"
#include "Backend/Resource.h"
#include "Constants.h"
#include "Header.h"
#include "IO/TabulatedReader.h"
#include "Types/Types.h"
namespace ARBD {

struct TabulatedPotential {
	DEVICE_PTR(float) pot; // Device pointer to potential table
	float step_inv;		   // 1/step_size for interpolation
	unsigned int size;	   // Table size
	float start;		   // Starting value (e.g., 0 for bonds, -PI for dihedrals)
	bool is_periodic;	   // Whether to wrap around (dihedrals)

	// Unified interpolation function
	DEVICE static inline ForceEnergy compute(float distance, const TabulatedPotential* table) {
		// Table lookup with linear interpolation
		float t = distance * table->step_inv;
		int home = int(floorf(t));
		t = t - home;

		if (home < 0)
			home = 0;
		if (home >= table->size)
			home = table->size - 1;

		int home1 = (home + 1 >= table->size) ? table->size - 1 : home + 1;

		float U0 = table->pot[home];
		float dU = table->pot[home1] - U0;

		return {dU * table->step_inv, dU * t + U0}; // Force magnitude
	}
};
} // namespace ARBD
