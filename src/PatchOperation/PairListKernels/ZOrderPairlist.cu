#include "Backend/CUDA/KernelHelper.cuh"
#include "PatchOperation/PairListKernels/ZOrderNeighbor.h"
#include "PatchOperation/ZOrderKernels/ZOrderKernels.h"
#include "Types/Types.h"
#include "ZOrderPairlist.h"
#include <cuda_runtime.h>

// Explicit template instantiations for ZOrderPairlist kernels
// These are needed so templates can be instantiated in CUDA compilation units
// instead of C++ files where they would get stub implementations
namespace ARBD {

// ZOrderNeighborKernel instantiation
template Event launch_cuda_kernel(const Resource& resource,
								  const KernelConfig& config,
								  ZOrderNeighborKernel kernel_func);

} // namespace ARBD
