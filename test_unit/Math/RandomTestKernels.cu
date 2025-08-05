#include "Backend/Buffer.h"
#include "Backend/CUDA/KernelHelper.cuh"
#include "Backend/Events.h"
#include "Backend/Kernels.h"
#include "Backend/Resource.h"
#include "Kernel_for_test.h"
#include "Random/RandomKernels.h"

// This file contains the explicit template instantiations needed by CUDA.
// You tell nvcc exactly which kernels to build.
namespace ARBD {
using BufferFloat = DeviceBuffer<float>;

template Event
launch_cuda_kernel_impl<std::tuple<BufferFloat&>, std::tuple<BufferFloat&>, TransformKernel>(
	const Resource&,
	size_t,
	const std::tuple<BufferFloat&>&,
	const std::tuple<BufferFloat&>&,
	const KernelConfig&,
	TransformKernel&&);

template Event launch_cuda_kernel_impl<std::tuple<BufferFloat&, BufferFloat&>,
									   std::tuple<BufferFloat&>,
									   CombineKernel>(const Resource&,
													  size_t,
													  const std::tuple<BufferFloat&, BufferFloat&>&,
													  const std::tuple<BufferFloat&>&,
													  const KernelConfig&,
													  CombineKernel&&);

// For UniformFunctor<int>
using BufferInt = DeviceBuffer<int>;
using BufferVector3 = DeviceBuffer<Vector3_t<float>>;
using BufferFloat = DeviceBuffer<float>;

// InitializeWalkersKernel instantiation
template Event
launch_cuda_kernel_impl<std::tuple<>, std::tuple<BufferVector3&>, InitializeWalkersKernel>(
	const Resource& resource,
	size_t thread_count,
	const std::tuple<>& inputs,
	const std::tuple<BufferVector3&>& outputs,
	const KernelConfig& config,
	InitializeWalkersKernel&& kernel_func);

// RandomWalkKernel instantiation
template Event
launch_cuda_kernel_impl<std::tuple<BufferVector3&>, std::tuple<BufferVector3&>, RandomWalkKernel>(
	const Resource& resource,
	size_t thread_count,
	const std::tuple<BufferVector3&>& inputs,
	const std::tuple<BufferVector3&>& outputs,
	const KernelConfig& config,
	RandomWalkKernel&& kernel_func);

// CalculateDistancesKernel instantiation
template Event
launch_cuda_kernel_impl<std::tuple<BufferVector3&>,
						std::tuple<BufferFloat&>,
						CalculateDistancesKernel>(const Resource& resource,
												  size_t thread_count,
												  const std::tuple<BufferVector3&>& inputs,
												  const std::tuple<BufferFloat&>& outputs,
												  const KernelConfig& config,
												  CalculateDistancesKernel&& kernel_func);

// SimpleKernel instantiation
template Event
launch_cuda_kernel_impl<std::tuple<BufferFloat&>, std::tuple<BufferFloat&>, SimpleKernel>(
	const Resource& resource,
	size_t thread_count,
	const std::tuple<BufferFloat&>& inputs,
	const std::tuple<BufferFloat&>& outputs,
	const KernelConfig& config,
	SimpleKernel&& kernel_func);

// SmoothingFilterKernel instantiation
template Event
launch_cuda_kernel_impl<std::tuple<BufferFloat&>, std::tuple<BufferFloat&>, SmoothingFilterKernel>(
	const Resource& resource,
	size_t thread_count,
	const std::tuple<BufferFloat&>& inputs,
	const std::tuple<BufferFloat&>& outputs,
	const KernelConfig& config,
	SmoothingFilterKernel&& kernel_func);

// GradientCalculationKernel instantiation
template Event
launch_cuda_kernel_impl<std::tuple<BufferFloat&>,
						std::tuple<BufferFloat&>,
						GradientCalculationKernel>(const Resource& resource,
												   size_t thread_count,
												   const std::tuple<BufferFloat&>& inputs,
												   const std::tuple<BufferFloat&>& outputs,
												   const KernelConfig& config,
												   GradientCalculationKernel&& kernel_func);
} // namespace ARBD
