#pragma once

// #include "../useful.h"
#include "Types/Types.h"

namespace ARBD {

namespace InteractionKernels {
HOST DEVICE void __inline__ HarmonicBonds() {
	// std::cout << "Computes::BDIntegrate_inline" << std::endl;
	printf("Interaction::HarmonicBondsDummy()\n");
};

HOST DEVICE void __inline__ HarmonicBonds(Vector3* __restrict__ pos,
										  const Vector3* const __restrict__ force) {
	printf("Interaction::HarmonicBonds\n");
	// pos[idx] = pos[idx] + force[idx] * root_Dt + normal_sample_3D;
};
} // namespace InteractionKernels
} // namespace ARBD
