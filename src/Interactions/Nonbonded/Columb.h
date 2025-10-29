#pragma once
#include "Header.h"

namespace ARBD {

struct Columb_kernel {
	KERNEL_FUNC void operator()(float* force,
								float* potential,
								float* distance,
								float* charge,
								float* charge_i,
								float* charge_j) {
		float r = distance[0];
		float qi = charge[0];
		float qj = charge[1];
		float force_x = qi * qj / (r * r);
		force[0] = force_x;
		potential[0] = qi * qj / r;
	}
};

} // namespace ARBD
