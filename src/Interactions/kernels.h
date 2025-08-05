#pragma once

// #include "../useful.h"
#include "Math/Types.h"

#ifdef __CUDACC__
    #define HOST __host__
    #define DEVICE __device__
#else
    #define HOST
    #define DEVICE
#endif
namespace ARBD {
namespace InteractionKernels {
    HOST DEVICE  void __inline__ HarmonicBonds() {
	// std::cout << "Computes::BDIntegrate_inline" << std::endl;
	printf("Interaction::HarmonicBondsDummy()\n");
    };

    HOST DEVICE  void __inline__ HarmonicBonds(Vector3* __restrict__ pos, const Vector3 * const __restrict__ force) {
	printf("Interaction::HarmonicBonds\n");
	// pos[idx] = pos[idx] + force[idx] * root_Dt + normal_sample_3D;
    };
}
}