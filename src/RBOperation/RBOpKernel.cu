#include "ApplyHostForce.h"
#include <cuda_runtime.h>
namespace ARBD {
template Event launch_cuda_kernel(const Resource& resource,
								  const KernelConfig& config,
								  ApplyExternalForcesKernel kernel_func);
}
