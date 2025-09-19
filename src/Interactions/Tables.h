#pragma once
// For parsing .dat files
#include "Header.h"
#include "IO/Reader.h"
#include "Objects/ParticleProperties.h"

namespace ARBD {

struct EnergyDerivative {
	float energy;
	float derivative; // The derivative, dE/dx
};

// The reusable functor for all tabulated interactions
struct TabulatedBonds {
	const DEVICE_PTR(float) pot; // Pointer to the potential table (e.g., pot for a specific bond)
	const float step_inv;		 // Inverse step size for the table
	const unsigned int size;	 // Size of the table
	const float start;			 // The 'x' value corresponding to index 0 (e.g., -PI for dihedrals)
	const bool is_periodic;		 // Is the potential periodic (like a dihedral)?

	// The main operator: takes an 'x' and returns energy/derivative 'y'
	KERNEL_FUNC EnergyDerivative operator()(float x) const {
		float w = (x - start) * step_inv;
		int home = static_cast<int>(floorf(w));
		w = w - home;

		if (is_periodic) {
			home %= size;
			if (home < 0) {
				home += size;
			}
		} else {
			if (home < 0)
				home = 0;
			if (home >= static_cast<int>(size) - 1)
				home = size - 2;
		}

		int home1 = is_periodic ? (home + 1) % size : home + 1;

		float u0 = pot[home];
		float du = pot[home1] - u0;

		return {
			du * w + u0,   // Linearly interpolated energy
			-du * step_inv // Derivative (force is the negative derivative)
		};
	}
};

} // namespace ARBD
