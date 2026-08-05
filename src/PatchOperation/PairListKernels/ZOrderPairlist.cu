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

// Exact 27-cell neighbor search and its cell-range builder. Both are launched
// from ZOrderPairlist.cpp, a host-only translation unit, so without these
// explicit instantiations the compiler would emit the non-CUDA stub from
// KernelHelper.cuh and every pairlist build would throw NotImplementedError.
template Event launch_cuda_kernel(const Resource& resource,
								  const KernelConfig& config,
								  BuildCellRangesKernel kernel_func);

template Event launch_cuda_kernel(const Resource& resource,
								  const KernelConfig& config,
								  ZOrderCellNeighborKernel kernel_func);

} // namespace ARBD
