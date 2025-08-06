#pragma once

#include "Backend/Buffer.h"
#include "Backend/Events.h"
#include "Backend/Kernels.h"
#include "Backend/Resource.h"

#ifdef __CUDACC__
// Only include CUDA headers when compiling with nvcc
#include "CUDAManager.h"
#include <cuda_runtime.h>
#include <thrust/tuple.h>
using namespace cuda::std;
#endif

namespace ARBD {

#ifdef __CUDACC__
// Forward declarations for CUDA-specific functions (defined in .cu files)
template<typename Functor, typename... Args>
__global__ void cuda_kernel_wrapper(size_t n, Functor kernel, Args... args) {
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) {
		kernel(i, args...);
	}
}
template<typename Functor, typename... Args>
void launch_cuda_wrapper_impl(dim3 grid,
							  dim3 block,
							  size_t shared_mem,
							  cudaStream_t stream,
							  size_t thread_count,
							  Functor&& kernel_func,
							  Args&&... args) {
	cuda_kernel_wrapper<<<grid, block, shared_mem, stream>>>(thread_count,
															 std::forward<Functor>(kernel_func),
															 std::forward<Args>(args)...);
}
#endif

/**
 * @brief Generic CUDA kernel implementation with full template support.
 *
 * This function handles all the CUDA setup, dependency management, and cleanup
 * while delegating the actual kernel launch to launch_cuda_wrapper_impl.
 *
 * By placing this in a header file, it can be instantiated for any user-defined
 * kernel types without requiring explicit instantiations.
 */
template<typename InputTuple, typename OutputTuple, typename Functor, typename... Args>
Event launch_cuda_kernel_impl(const Resource& resource,
							  size_t thread_count,
							  const InputTuple& inputs,
							  const OutputTuple& outputs,
							  const KernelConfig& config,
							  Functor&& kernel_func,
							  Args&&... args) {

#ifdef __CUDACC__
	KernelConfig local_config = config;
	local_config.auto_configure(thread_count, resource);

	dim3 grid(local_config.grid_size.x, local_config.grid_size.y, local_config.grid_size.z);
	dim3 block(local_config.block_size.x, local_config.block_size.y, local_config.block_size.z);

	// Ensure grid dimensions are valid (CUDA requires all dimensions >= 1)
	if (grid.x == 0){
		grid.x = 1;
	}
	if (grid.y == 0){
		grid.y = 1;
	}
	if (grid.z == 0){
		grid.z = 1;
	}

	// Get device and stream
	auto& device =
		const_cast<CUDA::Manager::Device&>(CUDA::Manager::devices()[resource.id]);
	cudaStream_t stream = device.get_next_stream();

	// Extract buffer pointers for kernel invocation
	auto input_pointers = get_buffer_pointers(inputs);
	auto output_pointers = get_buffer_pointers(outputs);
	auto all_pointers = std::tuple_cat(input_pointers, output_pointers);

	// Launch the kernel with extracted pointers by unpacking them
	std::apply(
		[&](auto&&... pointers) {
			launch_cuda_wrapper_impl(grid,
									 block,
									 0,
									 stream,
									 thread_count,
									 std::forward<Functor>(kernel_func),
									 pointers...,
									 std::forward<Args>(args)...);
		},
		all_pointers);

	// Check for kernel launch errors
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		throw std::runtime_error("CUDA kernel launch failed: " +
								 std::string(cudaGetErrorString(error)));
	}

	// Synchronize to catch any runtime errors
	error = cudaStreamSynchronize(stream);
	if (error != cudaSuccess) {
		throw std::runtime_error("CUDA kernel execution failed: " +
								 std::string(cudaGetErrorString(error)));
	}

	// Create and record completion event
	CUDA::Event event;
	event.record(stream);
	return Event(event.get(), resource);
#else
	// Fallback for non-CUDA compilation - should not be reached
	throw_not_implemented("launch_cuda_kernel_impl can only be used in CUDA compilation units");
#endif
}

} // namespace ARBD
