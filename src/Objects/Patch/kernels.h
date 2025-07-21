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
namespace IntegratorKernels {
HOST DEVICE void __inline__ BDIntegrate() {
  // std::cout << "Computes::BDIntegrate_inline" << std::endl;
  printf("Integrator::BDIntegrate\n");
};

HOST DEVICE void __inline__ BDIntegrate(Vector3 *__restrict__ pos,
                                        const Vector3 *const __restrict__ force,
                                        const int &idx, float &root_Dt,
                                        Vector3 &normal_sample_3D) {
  printf("Integrator::BDIntegrate\n");
  pos[idx] = pos[idx] + force[idx] * root_Dt + normal_sample_3D;
};

} // namespace IntegratorKernels
} // namespace ARBD
