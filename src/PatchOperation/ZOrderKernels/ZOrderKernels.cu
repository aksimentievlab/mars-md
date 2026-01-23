#include "Backend/CUDA/KernelHelper.cuh"
#include "PatchOperation/ZOrderKernels/ZOrderKernels.h"
#include <cuda_runtime.h>

// Explicit template instantiations for ZOrderKernels
// These ensure CUDA compilation units provide implementations used by host code.
namespace ARBD {

template Event launch_cuda_kernel(const Resource& resource,
								  const KernelConfig& config,
								  BoundingBoxKernel kernel_func);

template Event launch_cuda_kernel(const Resource& resource,
								  const KernelConfig& config,
								  MortonEncodeKernel kernel_func);

template Event launch_cuda_kernel(const Resource& resource,
								  const KernelConfig& config,
								  InverseIndexKernel kernel_func);

template Event launch_cuda_kernel(const Resource& resource,
								  const KernelConfig& config,
								  ValidateZOrderKernel kernel_func);

template Event launch_cuda_kernel(const Resource& resource,
								  const KernelConfig& config,
								  ReorderDataKernel<Vector3_t<float>> kernel_func);

template Event launch_cuda_kernel(const Resource& resource,
								  const KernelConfig& config,
								  DisplacementKernel kernel_func);

template Event launch_cuda_kernel(const Resource& resource,
								  const KernelConfig& config,
								  MortonValidationKernel kernel_func);

} // namespace ARBD
