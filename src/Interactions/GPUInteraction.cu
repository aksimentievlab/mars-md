#include "CUDA.h"

#ifdef USE_CUDA
__global__ void HarmonicBonds_kernel() {
    if (threadIdx.x == 0) {
	printf("HarmonicBonds_kernel()\n");
	InteractionKernels::HarmonicBonds();
    }
};

void LocalBondedCUDA::compute(Patch* p) {
    printf("HarmonicBondsCUDA::compute()\n");
    HarmonicBonds_kernel<<<1,32>>>();
};
#endif
