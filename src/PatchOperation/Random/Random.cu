#include "Backend/Buffer.h"
#include "Backend/CUDA/KernelHelper.cuh"
#include "RandomKernels.h"
#include "Types/Types.h"
#include <cuda_runtime.h>

// Explicit template instantiations for Random functors
// These are needed so templates can be instantiated in CUDA compilation units
// instead of C++ files where they would get stub implementations
namespace MARS {

// Note: get_buffer_pointer extracts the raw pointer from DeviceBuffer, so we use T*
template Event launch_cuda_kernel(const Resource& resource,
								  const KernelConfig& config,
								  UniformFunctor<float> kernel_func,
								  float* output);

template Event launch_cuda_kernel(const Resource& resource,
								  const KernelConfig& config,
								  UniformFunctor<uint32_t> kernel_func,
								  uint32_t* output);

template Event launch_cuda_kernel(const Resource& resource,
								  const KernelConfig& config,
								  UniformFunctor<int> kernel_func,
								  int* output);

template Event launch_cuda_kernel(const Resource& resource,
								  const KernelConfig& config,
								  GaussianFunctor<float> kernel_func,
								  float* output);

template Event launch_cuda_kernel(const Resource& resource,
								  const KernelConfig& config,
								  GaussianFunctor<Vector3> kernel_func,
								  Vector3* output);

} // namespace MARS
