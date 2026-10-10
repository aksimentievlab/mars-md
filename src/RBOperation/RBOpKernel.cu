
#include "Backend/CUDA/KernelHelper.cuh"

#include "ApplyHostForce.h"
#include "RBEnergyKernels.h"
#include <cuda_runtime.h>

namespace MARS {
template Event launch_cuda_kernel(const Resource& resource,
								  const KernelConfig& config,
								  ApplyExternalForcesKernel kernel_func);

template Event launch_cuda_kernel_with_workitem(const Resource& resource,
												const KernelConfig& config,
												RBEnergyReduceKernel kernel_func);
}
