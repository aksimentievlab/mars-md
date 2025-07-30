#include "CUDA.h"

#ifdef USE_CUDA
__global__ void BDIntegrate_kernel() {
    if (threadIdx.x == 0) {
	printf("BDIntegrate_kernel()\n");
	IntegratorKernels::BDIntegrate();
    }
};

void BDIntegrateCUDA::compute(Patch* p) {
    printf("BDIntegrateCUDA::compute()\n");
    BDIntegrate_kernel<<<1,32>>>();
};
#endif
