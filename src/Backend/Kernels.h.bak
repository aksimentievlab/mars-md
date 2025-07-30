#pragma once
#include <functional>
#include <future>
#include <memory>
#include <thread>
#include <tuple>
#include <type_traits>
#include <vector>
#ifdef USE_CUDA
#include "CUDA/CUDAManager.h"
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#endif

#ifdef USE_SYCL
#include "SYCL/SYCLManager.h"
#include <sycl/sycl.hpp>
#endif

#ifdef USE_METAL
#include "METAL/METALManager.h"
#include "Metal/Metal.hpp"
#endif
#include "ARBDException.h"
#include "ARBDLogger.h"
#include "Buffer.h"
#include "Events.h"
#include "Resource.h"

namespace ARBD {

struct kerneldim3 {
	size_t x = 1, y = 1, z = 1;
};

struct KernelConfig {
	kerneldim3 grid_size = {0,
							0,
							0}; // Grid dimensions. A zero value in x signals auto-calculation.
	kerneldim3 block_size = {256, 1, 1}; // Block/Work-group dimensions.
	size_t shared_memory = 0;			 // Shared memory in bytes (primarily for CUDA).
	bool async = false;					 // If false, the host will wait for completion.
	EventList dependencies;				 // Events this kernel must wait for.

	// Auto-configure grid size and validate block size for the target resource
	void auto_configure(size_t thread_count, const Resource& resource = {}) {
		// Validate and clamp block size based on resource type
		validate_block_size(resource);

		// Auto-configure grid size for a 1D problem if not specified
		if (grid_size.x == 0 && grid_size.y == 0 && grid_size.z == 0) {
			if (block_size.x > 0) {
				grid_size.x = (thread_count + block_size.x - 1) / block_size.x;
			}
			grid_size.y = 1;
			grid_size.z = 1;
		}
	}

  private:
	void validate_block_size(const Resource& resource) {
#ifdef USE_SYCL
		if (resource.type == ResourceType::SYCL) {
			try {
				auto& device = SYCL::SYCLManager::get_device(resource.id);
				size_t max_work_group_size =
					device.get_device().get_info<sycl::info::device::max_work_group_size>();

				// Get max work-item sizes for each dimension
				auto max_work_item_sizes =
					device.get_device().get_info<sycl::info::device::max_work_item_sizes<3>>();

				// Clamp each dimension to device limits
				block_size.x = std::min(block_size.x, static_cast<size_t>(max_work_item_sizes[0]));
				block_size.y = std::min(block_size.y, static_cast<size_t>(max_work_item_sizes[1]));
				block_size.z = std::min(block_size.z, static_cast<size_t>(max_work_item_sizes[2]));

				// Ensure total work-group size doesn't exceed device limit
				size_t total_work_items = block_size.x * block_size.y * block_size.z;
				if (total_work_items > max_work_group_size) {
					// Scale down proportionally
					double scale_factor =
						std::sqrt(static_cast<double>(max_work_group_size) / total_work_items);
					block_size.x = std::max(1UL, static_cast<size_t>(block_size.x * scale_factor));
					block_size.y = std::max(1UL, static_cast<size_t>(block_size.y * scale_factor));
					block_size.z = std::max(1UL, static_cast<size_t>(block_size.z * scale_factor));
				}

				LOGDEBUG("SYCL block size clamped to ({}, {}, {}) for device with max work-group "
						 "size {}",
						 block_size.x,
						 block_size.y,
						 block_size.z,
						 max_work_group_size);

			} catch (const sycl::exception& e) {
				LOGWARN("Failed to query SYCL device limits, using default block size: {}",
						e.what());
				// Fall back to safe defaults
				block_size = {256, 1, 1};
			}
		}
#endif

#ifdef USE_CUDA
		if (resource.type == ResourceType::CUDA) {
			try {
				auto& device = CUDA::CUDAManager::devices()[resource.id];
				cudaDeviceProp prop;
				CUDA_CHECK(cudaGetDeviceProperties(&prop, device.id()));

				// Clamp each dimension to CUDA limits
				block_size.x = std::min(block_size.x, static_cast<size_t>(prop.maxThreadsDim[0]));
				block_size.y = std::min(block_size.y, static_cast<size_t>(prop.maxThreadsDim[1]));
				block_size.z = std::min(block_size.z, static_cast<size_t>(prop.maxThreadsDim[2]));

				// Ensure total threads per block doesn't exceed limit
				size_t total_threads = block_size.x * block_size.y * block_size.z;
				if (total_threads > static_cast<size_t>(prop.maxThreadsPerBlock)) {
					double scale_factor =
						std::sqrt(static_cast<double>(prop.maxThreadsPerBlock) / total_threads);
					block_size.x = std::max(1UL, static_cast<size_t>(block_size.x * scale_factor));
					block_size.y = std::max(1UL, static_cast<size_t>(block_size.y * scale_factor));
					block_size.z = std::max(1UL, static_cast<size_t>(block_size.z * scale_factor));
				}

				LOGDEBUG("CUDA block size clamped to ({}, {}, {}) for device with max threads per "
						 "block {}",
						 block_size.x,
						 block_size.y,
						 block_size.z,
						 prop.maxThreadsPerBlock);

			} catch (...) {
				LOGWARN("Failed to query CUDA device limits, using default block size");
				block_size = {256, 1, 1};
			}
		}
#endif

		// For CPU, any block size is technically fine since we use std::thread
		if (resource.type == ResourceType::CPU) {
			// CPU execution doesn't have the same constraints, but we should be reasonable
			// Limit to something sensible to avoid creating too many threads
			const size_t max_cpu_threads = std::thread::hardware_concurrency() * 4;
			size_t total_threads = block_size.x * block_size.y * block_size.z;
			if (total_threads > max_cpu_threads) {
				block_size.x = std::min(block_size.x, max_cpu_threads);
				block_size.y = 1;
				block_size.z = 1;
			}
		}
	}
};

// Generic kernel function signature
template<typename... Args>
using KernelFunction = std::function<void(size_t, Args...)>;

/**
 * @brief Dispatches a functor-based kernel to the appropriate backend at runtime.
 *
 * This overload is selected when the kernel is a callable object (like a lambda or functor).
 * It checks the resource type and calls the corresponding backend-specific launch function.
 *
 * @tparam Functor The type of the callable kernel object.
 * @tparam Args The types of the arguments to the kernel.
 * @param resource The computational resource (CPU, CUDA, SYCL) to execute on.
 * @param thread_count The total number of threads to launch.
 * @param config The configuration for the kernel launch.
 * @param kernel_func The callable kernel object.
 * @param args The arguments to be passed to the kernel.
 * @return An Event object that can be used to track the kernel's completion.
 */
template<typename InputTuple, typename OutputTuple, typename Functor, typename... Args>
Event launch_kernel(const Resource& resource,
					size_t thread_count,
					const KernelConfig& config,
					InputTuple& inputs,
					OutputTuple& outputs,
					Functor&& kernel_func,
					Args&&... args) {
	try {
#ifdef USE_CUDA

		return launch_cuda_kernel(resource,
								  thread_count,
								  inputs,
								  outputs,
								  config,
								  std::forward<Functor>(kernel_func),
								  std::forward<Args>(args)...);

#elif defined(USE_SYCL)

		return launch_sycl_kernel(resource,
								  thread_count,
								  inputs,
								  outputs,
								  config,
								  std::forward<Functor>(kernel_func),
								  std::forward<Args>(args)...);

#elif defined(USE_METAL)
		throw_value_error("METAL backend requires a kernel name (string), not a functor. "
						  "Please use the named-kernel overload of launch_kernel.");
#else
		return launch_cpu_kernel(resource,
								 thread_count,
								 inputs,
								 outputs,
								 config,
								 std::forward<Functor>(kernel_func),
								 std::forward<Args>(args)...);
#endif
	} catch (const std::exception& e) {
		LOGERROR("Error in launch_kernel: {}", e.what());
		throw;
	}
}

/**
 * @brief Dispatches a name-based kernel to the appropriate backend at runtime.
 *
 * This overload is selected when the kernel is identified by a std::string.
 * It is primarily intended for backends like Metal.
 *
 * @tparam Args The types of the arguments to the kernel.
 * @param resource The computational resource (METAL) to execute on.
 * @param thread_count The total number of threads to launch.
 * @param config The configuration for the kernel launch.
 * @param kernel_name The name of the kernel to execute.
 * @param args The arguments to be passed to the kernel.
 * @return An Event object that can be used to track the kernel's completion.
 */
// template<typename... Args>
// Event launch_kernel(const Resource& resource,
// 					size_t thread_count,
// 					const KernelConfig& config,
// 					const std::string& kernel_name,
// 					Args&&... args) {
// 	switch (resource.type) {
// #ifdef USE_METAL
// 	case ResourceType::METAL:
// 		return launch_metal_kernel(resource,
// 								   thread_count,
// 								   config,
// 								   kernel_name,
// 								   std::forward<Args>(args)...);
// #endif
// 	case ResourceType::CUDA:
// 	case ResourceType::SYCL:
// 	case ResourceType::CPU:
// 		throw_value_error("CUDA, SYCL, and CPU backends require a functor, not a kernel name. "
// 						  "Please use the functor-based overload of launch_kernel.");
// 	default:
// 		throw_not_implemented("Unsupported resource type for named kernel launch.");
// 	}
// }

#ifdef USE_CUDA
// Forward declaration - actual implementation in KernelHelper.cu
template<typename Functor, typename... Args>
__global__ void cuda_kernel_wrapper(size_t n, Functor kernel, Args... args);
template<typename InputTuple, typename OutputTuple, typename Functor, typename... Args>
Event launch_cuda_kernel_impl(const Resource& resource,
							  size_t thread_count,
							  const InputTuple& inputs,
							  const OutputTuple& outputs,
							  const KernelConfig& config,
							  Functor&& kernel_func,
							  Args&&... args);

template<typename InputTuple, typename OutputTuple, typename Functor, typename... Args>
Event launch_cuda_kernel(const Resource& resource,
						 size_t thread_count,
						 const InputTuple& inputs,
						 const OutputTuple& outputs,
						 const KernelConfig& config,
						 Functor&& kernel_func,
						 Args&&... args) {
	return launch_cuda_kernel_impl(resource,
								   thread_count,
								   inputs,
								   outputs,
								   config,
								   std::forward<Functor>(kernel_func),
								   std::forward<Args>(args)...);
}
#endif

#ifdef USE_CUDA
// Validation function that works on both host and device
template<typename T>
void validate_block_size(const T& block_size, const cudaDeviceProp& prop) {
	size_t total_threads = block_size.x * block_size.y * block_size.z;
	if (total_threads > static_cast<size_t>(prop.maxThreadsPerBlock)) {
// Use host-side logging only
#ifndef __CUDA_ARCH__
		LOGDEBUG("Block size ({}, {}, {}) exceeds max threads per block ({})",
				 block_size.x,
				 block_size.y,
				 block_size.z,
				 prop.maxThreadsPerBlock);
		LOGWARN("Invalid block size configuration");
#endif
		throw std::runtime_error("Block size exceeds device limits");
	}
}
#endif
#ifdef USE_SYCL
template<typename InputTuple, typename OutputTuple, typename Functor, typename... Args>
Event launch_sycl_kernel(const Resource& resource,
						 size_t thread_count,
						 const InputTuple& inputs,
						 const OutputTuple& outputs,
						 const KernelConfig& config,
						 Functor&& kernel_func,
						 Args&&... args) {

	KernelConfig local_config = config;
	// Pass the resource to auto_configure so it can validate device limits
	local_config.auto_configure(thread_count, resource);

	// 1. Define the execution range using your validated KernelConfig settings.
	sycl::range<1> global_range(local_config.grid_size.x * local_config.block_size.x);
	sycl::range<1> local_range(local_config.block_size.x);
	sycl::nd_range<1> execution_range(global_range, local_range);

	// 2. Get the correct SYCL queue from your manager.
	auto& queue = SYCL::SYCLManager::get_device(resource.id).get_next_queue();

	// 3. Submit the command group to the queue.
	auto sycl_event = queue.get().submit([&](sycl::handler& h) {
		// 4. Use your EventList to manage dependencies.
		h.depends_on(config.dependencies.get_sycl_events());

		// 5. Get raw USM pointers from the input/output buffer tuples.
		auto input_pointers = get_buffer_pointers(inputs);
		auto output_pointers = get_buffer_pointers(outputs);

		// Combine all pointers and arguments for the kernel lambda.
		auto kernel_args = std::tuple_cat(input_pointers,
										  output_pointers,
										  std::make_tuple(std::forward<Args>(args)...));

		// 6. Launch the parallel_for kernel with validated configuration.
		h.parallel_for(execution_range, [=](sycl::nd_item<1> item) {
			size_t i = item.get_global_id(0);
			if (i < thread_count) {
				// 7. Unpack the pointers and arguments and call the user's kernel functor.
				std::apply([&](auto&&... unpacked_args) { kernel_func(i, unpacked_args...); },
						   kernel_args);
			}
		});
	});

	// 8. Handle synchronous execution if requested.
	if (!config.async) {
		sycl_event.wait();
	}

	// 9. Return your backend-agnostic Event wrapper.
	return Event(sycl_event, resource);
}
#endif

#ifdef USE_METAL
template<typename InputTuple, typename OutputTuple, typename... Args>
Event launch_metal_kernel(const Resource& resource,
						  size_t thread_count,
						  const InputTuple& inputs,
						  const OutputTuple& outputs,
						  const KernelConfig& config,
						  const std::string& kernel_name,
						  Args&&... args) {

	// Wait for dependencies
	config.dependencies.wait_all();

	// Get Metal pipeline state
	MTL::ComputePipelineState* pipeline =
		METAL::METALManager::get_compute_pipeline_state(kernel_name);

	auto& device = METAL::METALManager::get_current_device();
	auto& queue = device.get_next_queue();

	// Create command buffer and encoder
	void* cmd_buffer_ptr = queue.create_command_buffer();
	MTL::CommandBuffer* cmd_buffer = static_cast<MTL::CommandBuffer*>(cmd_buffer_ptr);
	MTL::ComputeCommandEncoder* encoder = cmd_buffer->computeCommandEncoder();

	encoder->setComputePipelineState(pipeline);

	// Get pointers from buffer objects and set buffer arguments
	EventList deps;
	auto input_ptr = inputs.get_read_access(deps);
	auto output_ptr = outputs.get_write_access(deps);

	int buffer_index = 0;
	void* input_buffer_ptr = METAL::METALManager::get_metal_buffer_from_ptr(
		const_cast<void*>(static_cast<const void*>(input_ptr)));
	void* output_buffer_ptr = METAL::METALManager::get_metal_buffer_from_ptr(output_ptr);

	encoder->setBuffer(static_cast<const MTL::Buffer*>(input_buffer_ptr), 0, buffer_index++);
	encoder->setBuffer(static_cast<const MTL::Buffer*>(output_buffer_ptr), 0, buffer_index++);

	// Configure grid size
	KernelConfig local_config = config;
	local_config.auto_configure(thread_count);

	// Dispatch
	MTL::Size grid_size = MTL::Size::Make(local_config.grid_size.x,
										  local_config.grid_size.y,
										  local_config.grid_size.z);
	// 1. Get the max size from the pipeline state.
	NS::UInteger max_threads_per_group = pipeline->maxTotalThreadsPerThreadgroup();

	// 2. Ensure your desired block size is also an NS::UInteger.
	NS::UInteger desired_threads_per_group = config.block_size.x;

	// 3. Now std::min works correctly as both types match.
	NS::UInteger final_threads_per_group =
		std::min(desired_threads_per_group, max_threads_per_group);
	// Dispatch the kernel
	MTL::Size threadgroup_size = MTL::Size::Make(local_config.block_size.x,
												 local_config.block_size.y,
												 local_config.block_size.z);

	encoder->dispatchThreads(grid_size, threadgroup_size);
	encoder->endEncoding();

	ARBD::METAL::Event metal_event(cmd_buffer_ptr);
	metal_event.commit();

	if (!config.async) {
		metal_event.wait();
	}

	return Event(metal_event, resource);
}
#endif

// Helper to extract raw pointers from buffer tuples
template<typename Tuple, std::size_t... I>
auto extract_buffer_pointers_impl(Tuple& tuple, std::index_sequence<I...>) {
	return std::make_tuple(std::get<I>(tuple).data()...);
}

template<typename Tuple>
auto extract_buffer_pointers(Tuple& tuple) {
	return extract_buffer_pointers_impl(tuple,
										std::make_index_sequence<std::tuple_size_v<Tuple>>{});
}

// CPU fallback implementation (always available)
template<typename InputTuple, typename OutputTuple, typename Functor, typename... Args>
Event launch_cpu_kernel(const Resource& resource,
						size_t thread_count,
						InputTuple& inputs,
						OutputTuple& outputs,
						const KernelConfig& config,
						Functor&& kernel_func,
						Args... args) {

	// Wait for dependencies before starting
	config.dependencies.wait_all();

	// Extract raw pointers from buffer tuples
	auto input_ptrs = extract_buffer_pointers(inputs);
	auto output_ptrs = extract_buffer_pointers(outputs);

	// Determine the number of concurrent threads to use
	unsigned int num_threads = std::thread::hardware_concurrency();
	if (num_threads == 0) {
		num_threads = 1;
	} // Fallback if concurrency is not detectable

	std::vector<std::thread> threads;
	size_t chunk_size = (thread_count + num_threads - 1) / num_threads;

	// Launch threads, giving each a "chunk" of the work
	for (unsigned int t = 0; t < num_threads; ++t) {
		threads.emplace_back([=]() {
			size_t start = t * chunk_size;
			size_t end = std::min(start + chunk_size, thread_count);
			for (size_t i = start; i < end; ++i) {
				// Create combined tuple of all pointer arguments for the kernel
				auto all_args = std::tuple_cat(input_ptrs, output_ptrs);
				// Use std::apply to unpack the tuple and call the kernel
				std::apply([&](auto&&... unpacked_args) { kernel_func(i, unpacked_args...); },
						   all_args);
			}
		});
	}

	// Wait for all CPU threads to complete
	for (auto& thread : threads) {
		if (thread.joinable()) {
			thread.join();
		}
	}

	// For a CPU launch, the event is immediately considered complete
	return Event(nullptr, resource);
}
// Make KernelChain a template on the Backend type
template<typename Backend>
class KernelChain {
  private:
	const Resource& resource_;
	EventList events_;

  public:
	explicit KernelChain(const Resource& resource) : resource_(resource) {}

	// Overload for functor-based kernels (CUDA, SYCL, CPU)
	template<typename InputTuple, typename OutputTuple, typename Functor, typename... Args>
	KernelChain& then(size_t thread_count,
					  InputTuple& inputs,
					  OutputTuple& outputs,
					  Functor&& kernel,
					  const KernelConfig& config = {},
					  Args&&... args) {

		KernelConfig new_config = config;
		new_config.dependencies = events_; // Chain the events automatically
		new_config.async = true;

		// The magic happens here: call the generic, overloaded launch_kernel
		// The compiler will pick the correct version based on the Backend template parameter.
		Event completion_event = launch_kernel<Backend>(resource_,
														thread_count,
														inputs,
														outputs,
														new_config,
														std::forward<Functor>(kernel),
														std::forward<Args>(args)...);

		// The new event becomes the dependency for the next step in the chain
		events_.clear();
		events_.add(completion_event);

		return *this;
	}

	// Overload for name-based kernels (Metal)
	template<typename InputTuple, typename OutputTuple, typename... Args>
	KernelChain& then(size_t thread_count,
					  InputTuple& inputs,
					  OutputTuple& outputs,
					  const std::string& kernel_name,
					  const KernelConfig& config = {},
					  Args&&... args) {

		KernelConfig new_config = config;
		new_config.dependencies = events_;
		new_config.async = true;

		Event completion_event = launch_kernel<Backend>(resource_,
														thread_count,
														inputs,
														outputs,
														new_config,
														kernel_name,
														std::forward<Args>(args)...);

		events_.clear();
		events_.add(completion_event);

		return *this;
	}

	void wait() {
		events_.wait_all();
	}
};

// ============================================================================
// Utility Functions
// ============================================================================
template<typename T>
struct CopyFunctor {
	HOST DEVICE void operator()(size_t i, const T* src, T* dst) const {
		dst[i] = src[i];
	}
};

template<typename T>
struct FillFunctor {
	T value;
	HOST DEVICE void operator()(size_t i, T* out) const {
		out[i] = value;
	}
};

// --- Backend-Agnostic API Functions ---
template<typename Backend, typename T>
Event copy_async(const Resource& resource,
				 const DeviceBuffer<T>& source,
				 DeviceBuffer<T>& destination,
				 const KernelConfig& config = {}) {

	if (source.size() != destination.size()) {
		throw std::runtime_error("Buffer size mismatch in copy_async");
	}

	// Launch the generic copy kernel on the specified backend
	return launch_kernel<Backend>(resource,
								  source.size(),
								  std::tie(source),
								  std::tie(destination),
								  config,
								  CopyFunctor<T>{});
}

template<typename Backend, typename T>
Event fill_async(const Resource& resource,
				 DeviceBuffer<T>& buffer,
				 const T& value,
				 const KernelConfig& config = {}) {

	// Launch the generic fill kernel on the specified backend
	return launch_kernel<Backend>(resource,
								  buffer.size(),
								  std::tie(),
								  std::tie(buffer),
								  config,
								  FillFunctor<T>{value});
}

// ============================================================================
// Result Wrapper for Kernel Calls
// ============================================================================

template<typename T>
struct KernelResult {
	T result;
	Event completion_event;

	KernelResult(T&& res, Event&& event)
		: result(std::forward<T>(res)), completion_event(std::move(event)) {}

	void wait() {
		completion_event.wait();
	}
	bool is_ready() const {
		return completion_event.is_complete();
	}

	T get() {
		wait();
		return std::move(result);
	}
};

} // namespace ARBD
