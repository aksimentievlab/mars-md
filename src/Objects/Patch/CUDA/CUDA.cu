#include "CUDA.h"

#ifdef USE_CUDA
// __global__ void BDIntegrate_kernel() {
//     if (threadIdx.x == 0) {
// 	printf("BDIntegrate_kernel()\n");
// 	IntegratorKernels::BDIntegrate();
//     }
// };

// void BDIntegrateCUDA::compute(Patch* p) {
//     printf("BDIntegrateCUDA::compute()\n");
//     BDIntegrate_kernel<<<1,32>>>();
// };

PatchCUDA::PatchCUDA() : Patch() {
    pos_force_d = momentum_d = rb_pos_d = rb_orient_d = rb_mom_d = rb_ang_mom_d = type_d = rb_type_d = nullptr;
}
#endif
