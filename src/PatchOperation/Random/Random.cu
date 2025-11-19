#include "Backend/CUDA/KernelHelper.cuh"
#include "Random/RandomKernels.h"
#include "Types/Types.h"
#include "Types/Vector3.h"

// Include any other necessary headers
#include "Backend/Buffer.h"
#include "Backend/Events.h"
#include "Backend/Kernels.h"
#include "Backend/Resource.h"

namespace ARBD {
using BufferFloat = DeviceBuffer<float>;
using BufferVector3 = DeviceBuffer<ARBD::Vector3_t<float>>;

// Random kernel template instantiations
// UniformFunctor template instantiations
template Event launch_cuda_kernel(const Resource& resource,
								  const KernelConfig& config,
								  UniformFunctor<float>& kernel_func,
								  float min_val,
								  float max_val,
								  uint64_t seed_,
								  uint32_t current_ctr,
								  uint32_t global_seed_,
								  BufferFloat& output);
// GaussianFunctor template instantiations
template Event launch_cuda_kernel(const Resource& resource,
								  const KernelConfig& config,
								  GaussianFunctor<float>& kernel_func,
								  float mean,
								  float stddev,
								  uint64_t seed_,
								  uint32_t current_ctr,
								  uint32_t global_seed_,
								  BufferFloat& output);
template Event launch_cuda_kernel(const Resource& resource,
								  const KernelConfig& config,
								  GaussianFunctor<Vector3>& kernel_func,
								  float mean,
								  float stddev,
								  uint64_t seed_,
								  uint32_t current_ctr,
								  uint32_t global_seed_,
								  BufferVector3& output);

// Also need int version for UniformFunctor
template Event launch_cuda_kernel(const Resource& resource,
								  const KernelConfig& config,
								  UniformFunctor<int>& kernel_func,
								  int min_val,
								  int max_val,
								  uint64_t seed_,
								  uint32_t current_ctr,
								  uint32_t global_seed_,
								  DeviceBuffer<int>& output);
} // namespace ARBD
