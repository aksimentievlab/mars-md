#pragma once

#include "../Buffer.h"
#include "../Events.h"
#include "../KernelConfig.h"
#include "../Resource.h"

#ifdef __CUDACC__
// Only include CUDA headers when compiling with nvcc
#include "CUDAManager.h"
#include <cuda_runtime.h>
#include <thrust/tuple.h>
using namespace cuda::std;
#endif

namespace ARBD {

// Generic kernel wrapper that can call any functor
#ifdef __CUDACC__
template<typename Functor, typename... Args>
__global__ void cuda_kernel_wrapper(idx_t n, Functor kernel, Args... args) {
	idx_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) {
		kernel(i, args...);
	}
}

/**
 * @brief Generic CUDA kernel implementation with full template support.
 *
 * This function handles all the CUDA setup, dependency management, and cleanup
 * while delegating the actual kernel launch to launch_cuda_wrapper_impl.
 *
 * By placing this in a header file, it can be instantiated for any user-defined
 * kernel types without requiring explicit instantiations.
 */

template<typename Functor, typename... Args>
Event launch_cuda_kernel(const Resource& resource,
						 const KernelConfig& config,
						 Functor kernel_func,
						 Args... args) {
	// Get queue from config
	cudaStream_t stream = static_cast<cudaStream_t>(config.get_queue(resource));

	// Handle dependencies
	for (const auto& dep_event : config.dependencies.get_cuda_events()) {
		CUDA_CHECK(cudaStreamWaitEvent(stream, dep_event, 0));
	}
	// Auto-configure if needed
	KernelConfig local_config = config;

	// Set device context
	int old_device;
	CUDA_CHECK(cudaGetDevice(&old_device));
	CUDA_CHECK(cudaSetDevice(static_cast<int>(resource.id)));

	// Launch kernel using generic wrapper
	dim3 grid(local_config.grid_size.x, local_config.grid_size.y, local_config.grid_size.z);
	dim3 block(local_config.block_size.x, local_config.block_size.y, local_config.block_size.z);
	idx_t thread_count =
		local_config.problem_size.x * local_config.problem_size.y * local_config.problem_size.z;
	cuda_kernel_wrapper<<<grid, block, local_config.shared_memory, stream>>>(
		thread_count,
		kernel_func,
		get_buffer_pointer(args)...);

	// Check for launch errors
	CUDA_CHECK(cudaGetLastError());

	// Create completion event
	cudaEvent_t completion_event;
	CUDA_CHECK(cudaEventCreateWithFlags(&completion_event, cudaEventDisableTiming));
	CUDA_CHECK(cudaEventRecord(completion_event, stream));

	// Restore device context
	CUDA_CHECK(cudaSetDevice(old_device));

	return Event(completion_event, resource);
#else
throw_not_implemented("launch_cuda_kernel can only be used in CUDA compilation units");
}
#endif
	// Overloaded version for input and output buffers
	template<typename InputTuple, typename OutputTuple, typename Functor, typename... Args>
	Event launch_cuda_kernel(const Resource& resource,
							 idx_t thread_count,
							 const KernelConfig& config,
							 Functor&& kernel_func,
							 const InputTuple& inputs,
							 const OutputTuple& outputs,
							 Args&&... args) {

#ifdef __CUDACC__
		KernelConfig local_config = config;
		// Auto-configuration should be done by caller before calling this function

		dim3 grid(local_config.grid_size.x, local_config.grid_size.y, local_config.grid_size.z);
		dim3 block(local_config.block_size.x, local_config.block_size.y, local_config.block_size.z);

		// Ensure grid dimensions are valid (CUDA requires all dimensions >= 1)
		if (grid.x == 0) {
			grid.x = 1;
		}
		if (grid.y == 0) {
			grid.y = 1;
		}
		if (grid.z == 0) {
			grid.z = 1;
		}

		// Get device and stream
		auto& device = const_cast<CUDA::Manager::Device&>(CUDA::Manager::devices()[resource.id]);

		// Ensure the correct CUDA device context is active for this device/stream
		int previous_device = -1;
		CUDA_CHECK(cudaGetDevice(&previous_device));
		CUDA_CHECK(cudaSetDevice(static_cast<int>(device.id())));

		cudaStream_t stream = device.get_next_stream();

		// Honor dependency events by making the stream wait on them
		for (cudaEvent_t dep_evt : config.dependencies.get_cuda_events()) {
			cudaError_t wait_err = cudaStreamWaitEvent(stream, dep_evt, 0);
			if (wait_err != cudaSuccess) {
				// Restore previous device before throwing
				cudaSetDevice(previous_device);
				throw std::runtime_error(std::string("Failed to wait on dependency event: ") +
										 cudaGetErrorString(wait_err));
			}
		}

		// Extract buffer pointers for kernel invocation
		auto input_pointers = get_buffer_tuples(inputs);
		auto output_pointers = get_buffer_tuples(outputs);
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
			// Restore previous device before throwing
			cudaSetDevice(previous_device);
			throw std::runtime_error("CUDA kernel launch failed: " +
									 std::string(cudaGetErrorString(error)));
		}

		// Synchronize to catch any runtime errors
		error = cudaStreamSynchronize(stream);
		if (error != cudaSuccess) {
			// Restore previous device before throwing
			cudaSetDevice(previous_device);
			throw std::runtime_error("CUDA kernel execution failed: " +
									 std::string(cudaGetErrorString(error)));
		}

		// Create a raw CUDA event that outlives the local RAII wrapper
		cudaEvent_t completion_event;
		CUDA_CHECK(cudaEventCreateWithFlags(&completion_event, cudaEventDisableTiming));
		CUDA_CHECK(cudaEventRecord(completion_event, stream));

		// Restore previous device context
		cudaSetDevice(previous_device);
		return Event(completion_event, resource);
#else
	// Fallback for non-CUDA compilation - should not be reached
	throw_not_implemented("launch_cuda_kernel_impl can only be used in CUDA compilation units");
#endif
	}

} // namespace ARBD
