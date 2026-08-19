#include "Backend/CUDA/KernelHelper.cuh"
#include "PatchOperation/PairListKernels/ZOrderNeighbor.h"
#include "PatchOperation/ZOrderKernels/ZOrderKernels.h"
#include "Types/Types.h"
#include "ZOrderPairlist.h"
#include <cuda_runtime.h>

namespace ARBD {

// Exact 27-cell neighbor search, its cell-range builder, and the cell-neighbor table.
template Event launch_cuda_kernel(const Resource& resource,
								  const KernelConfig& config,
								  BuildCellRangesKernel kernel_func);

template Event launch_cuda_kernel(const Resource& resource,
								  const KernelConfig& config,
								  BuildCellNeighborsKernel kernel_func);

template Event launch_cuda_kernel(const Resource& resource,
								  const KernelConfig& config,
								  ZOrderCellNeighborKernel kernel_func);

} // namespace ARBD
