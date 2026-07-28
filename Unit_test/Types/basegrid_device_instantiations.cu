#include "Backend/CUDA/KernelHelper.cuh"
#include "Types/basegrid_device.h"
#include <cuda_runtime.h>

// Explicit template instantiation for the BaseGrid device test kernel.
// See basegrid_device.h for why this can't be an ad hoc lambda.
namespace ARBD {

template Event launch_cuda_kernel(const Resource& resource,
								  const KernelConfig& config,
								  Test::GridQueryKernel<float> kernel_func);

} // namespace ARBD
