#include "Backend/CUDA/KernelHelper.cuh"
#include "Math/Types.h"
#include "Math/Vector3.h"
#include "Random/RandomKernels.h"

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
template Event
launch_cuda_kernel_impl<std::tuple<>, std::tuple<BufferFloat&>, UniformFunctor<float>&>(
	const Resource& resource,
	size_t thread_count,
	const std::tuple<>& inputs,
	const std::tuple<BufferFloat&>& outputs,
	const KernelConfig& config,
	UniformFunctor<float>& kernel_func);

template Event launch_cuda_kernel_impl<std::tuple<>,
									   std::tuple<BufferVector3&>,
									   UniformFunctor<ARBD::Vector3_t<float>>&>(
	const Resource& resource,
	size_t thread_count,
	const std::tuple<>& inputs,
	const std::tuple<BufferVector3&>& outputs,
	const KernelConfig& config,
	UniformFunctor<ARBD::Vector3_t<float>>& kernel_func);

// GaussianFunctor template instantiations
template Event
launch_cuda_kernel_impl<std::tuple<>, std::tuple<BufferFloat&>, GaussianFunctor<float>&>(
	const Resource& resource,
	size_t thread_count,
	const std::tuple<>& inputs,
	const std::tuple<BufferFloat&>& outputs,
	const KernelConfig& config,
	GaussianFunctor<float>& kernel_func);

template Event launch_cuda_kernel_impl<std::tuple<>,
									   std::tuple<BufferVector3&>,
									   GaussianFunctor<ARBD::Vector3_t<float>>&>(
	const Resource& resource,
	size_t thread_count,
	const std::tuple<>& inputs,
	const std::tuple<BufferVector3&>& outputs,
	const KernelConfig& config,
	GaussianFunctor<ARBD::Vector3_t<float>>& kernel_func);

// Also need int version for UniformFunctor
template Event
launch_cuda_kernel_impl<std::tuple<>, std::tuple<DeviceBuffer<int>&>, UniformFunctor<int>&>(
	const Resource& resource,
	size_t thread_count,
	const std::tuple<>& inputs,
	const std::tuple<DeviceBuffer<int>&>& outputs,
	const KernelConfig& config,
	UniformFunctor<int>& kernel_func);
} // namespace ARBD
