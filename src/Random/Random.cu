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

// --- EXPLICIT TEMPLATE INSTANTIATIONS ---
// This is where you tell nvcc which versions of your templates to build.

using BufferFloat = DeviceBuffer<float>;

template Event
launch_cuda_kernel_impl<std::tuple<>, std::tuple<BufferFloat&>, UniformFunctor<float>&>(
	const Resource&,
	size_t,
	const std::tuple<>&,
	const std::tuple<BufferFloat&>&,
	const KernelConfig&,
	UniformFunctor<float>&);

// For GaussianFunctor<float>
template Event
launch_cuda_kernel_impl<std::tuple<>, std::tuple<BufferFloat&>, GaussianFunctor<float>&>(
	const Resource&,
	size_t,
	const std::tuple<>&,
	const std::tuple<BufferFloat&>&,
	const KernelConfig&,
	GaussianFunctor<float>&);

// For GaussianFunctor<Vector3_t<float>>
using BufferVec3f = DeviceBuffer<Vector3_t<float>>;
template Event
launch_cuda_kernel_impl<std::tuple<>, std::tuple<BufferVec3f&>, GaussianFunctor<Vector3_t<float>>&>(
	const Resource&,
	size_t,
	const std::tuple<>&,
	const std::tuple<BufferVec3f&>&,
	const KernelConfig&,
	GaussianFunctor<Vector3_t<float>>&);

using BufferInt = DeviceBuffer<int>;
template Event launch_cuda_kernel_impl<std::tuple<>, std::tuple<BufferInt&>, UniformFunctor<int>&>(
	const Resource&,
	size_t,
	const std::tuple<>&,
	const std::tuple<BufferInt&>&,
	const KernelConfig&,
	UniformFunctor<int>&);
} // namespace ARBD
