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
#include <thrust/tuple.h>
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

// ============================================================================
// Configuration and Types
// ============================================================================

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

	void auto_configure(size_t thread_count, const Resource& resource = {});

  private:
	void validate_block_size(const Resource& resource);
};

template<typename... Args>
using KernelFunction = std::function<void(size_t, Args...)>;

// ============================================================================
// Type Traits for Buffer Detection
// ============================================================================

/**
 * @brief A convenient alias for the Buffer class using the active backend policy.
 */
template<typename T>
using DeviceBuffer = Buffer<T, BackendPolicy>;

template<typename T>
struct is_device_buffer : std::false_type {};

template<typename T>
struct is_device_buffer<DeviceBuffer<T>> : std::true_type {};

template<typename T>
constexpr bool is_device_buffer_v = is_device_buffer<std::decay_t<T>>::value;

template<typename T>
struct is_string : std::false_type {};

template<>
struct is_string<std::string> : std::true_type {};

template<>
struct is_string<const std::string> : std::true_type {};

template<>
struct is_string<const char*> : std::true_type {};

template<>
struct is_string<char*> : std::true_type {};

template<typename T>
constexpr bool is_string_v = is_string<std::decay_t<T>>::value;

// ============================================================================
// Device Kernel Launchers (CUDA, SYCL, METAL)
// ============================================================================

/**
 * @brief Core device kernel launcher - tuple-based interface
 */
template<typename InputTuple, typename OutputTuple, typename Functor, typename... Args>
std::enable_if_t<!is_device_buffer_v<InputTuple> && !is_device_buffer_v<OutputTuple>, Event>
launch_kernel(const Resource& resource,
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
 * @brief Name-based kernel launcher (for Metal)
 */
template<typename KernelName, typename... Args>
std::enable_if_t<is_string_v<KernelName>, Event> launch_kernel(const Resource& resource,
															   size_t thread_count,
															   const KernelConfig& config,
															   KernelName&& kernel_name,
															   Args&&... args) {
	switch (resource.type) {
#ifdef USE_METAL
	case ResourceType::METAL:
		return launch_metal_kernel(resource,
								   thread_count,
								   config,
								   std::forward<KernelName>(kernel_name),
								   std::forward<Args>(args)...);
#endif
	case ResourceType::CUDA:
	case ResourceType::SYCL:
	case ResourceType::CPU:
		throw_value_error("CUDA, SYCL, and CPU backends require a functor, not a kernel name.");
	default:
		throw_not_implemented("Unsupported resource type for named kernel launch.");
	}
}

/**
 * @brief Single output buffer (generators like Random)
 */
template<typename OutputBuffer, typename Functor, typename... Args>
std::enable_if_t<is_device_buffer_v<OutputBuffer> && !is_device_buffer_v<Functor> &&
					 !is_string_v<Functor>,
				 Event>
launch_kernel(const Resource& resource,
			  size_t thread_count,
			  const KernelConfig& config,
			  OutputBuffer& output_buffer,
			  Functor&& kernel_func,
			  Args&&... args) {

	auto inputs = std::make_tuple();
	auto outputs = std::make_tuple(std::ref(output_buffer));

	return launch_kernel(resource,
						 thread_count,
						 config,
						 inputs,
						 outputs,
						 std::forward<Functor>(kernel_func),
						 std::forward<Args>(args)...);
}

/**
 * @brief Single input + single output buffers (transforms)
 */
template<typename InputBuffer, typename OutputBuffer, typename Functor, typename... Args>
std::enable_if_t<is_device_buffer_v<InputBuffer> && is_device_buffer_v<OutputBuffer> &&
					 !is_device_buffer_v<Functor> && !is_string_v<Functor>,
				 Event>
launch_kernel(const Resource& resource,
			  size_t thread_count,
			  const KernelConfig& config,
			  InputBuffer& input_buffer,
			  OutputBuffer& output_buffer,
			  Functor&& kernel_func,
			  Args&&... args) {

	auto inputs = std::make_tuple(std::ref(input_buffer));
	auto outputs = std::make_tuple(std::ref(output_buffer));

	return launch_kernel(resource,
						 thread_count,
						 config,
						 inputs,
						 outputs,
						 std::forward<Functor>(kernel_func),
						 std::forward<Args>(args)...);
}

/**
 * @brief Dual input + single output buffers (binary operations)
 */
template<typename InputBuffer1,
		 typename InputBuffer2,
		 typename OutputBuffer,
		 typename Functor,
		 typename... Args>
std::enable_if_t<is_device_buffer_v<InputBuffer1> && is_device_buffer_v<InputBuffer2> &&
					 is_device_buffer_v<OutputBuffer> && !is_device_buffer_v<Functor> &&
					 !is_string_v<Functor>,
				 Event>
launch_kernel(const Resource& resource,
			  size_t thread_count,
			  const KernelConfig& config,
			  InputBuffer1& input_buffer1,
			  InputBuffer2& input_buffer2,
			  OutputBuffer& output_buffer,
			  Functor&& kernel_func,
			  Args&&... args) {

	auto inputs = std::make_tuple(std::ref(input_buffer1), std::ref(input_buffer2));
	auto outputs = std::make_tuple(std::ref(output_buffer));

	return launch_kernel(resource,
						 thread_count,
						 config,
						 inputs,
						 outputs,
						 std::forward<Functor>(kernel_func),
						 std::forward<Args>(args)...);
}

// ============================================================================
// CPU Kernel Launcher (Host-only)
// ============================================================================

/**
 * @brief CPU kernel launcher - tuple-based interface
 */
template<typename InputTuple, typename OutputTuple, typename Functor, typename... Args>
Event launch_cpu_kernel(const Resource& resource,
						size_t thread_count,
						const InputTuple& inputs,
						const OutputTuple& outputs,
						const KernelConfig& config,
						Functor&& kernel_func,
						Args... args) {

	config.dependencies.wait_all();

	auto input_ptrs = extract_buffer_pointers(inputs);
	auto output_ptrs = extract_buffer_pointers(outputs);

	unsigned int num_threads = std::thread::hardware_concurrency();
	if (num_threads == 0) {
		num_threads = 1;
	}

	std::vector<std::thread> threads;
	size_t chunk_size = (thread_count + num_threads - 1) / num_threads;

	for (unsigned int t = 0; t < num_threads; ++t) {
		threads.emplace_back([=]() {
			size_t start = t * chunk_size;
			size_t end = std::min(start + chunk_size, thread_count);
			for (size_t i = start; i < end; ++i) {
				auto all_args = std::tuple_cat(input_ptrs, output_ptrs);
				std::apply([&](auto&&... unpacked_args) { kernel_func(i, unpacked_args...); },
						   all_args);
			}
		});
	}

	for (auto& thread : threads) {
		if (thread.joinable()) {
			thread.join();
		}
	}

	return Event(nullptr, resource);
}

/**
 * @brief CPU kernel launcher - single output buffer
 */
template<typename OutputBuffer, typename Functor, typename... Args>
Event launch_cpu_kernel(const Resource& resource,
						size_t thread_count,
						const KernelConfig& config,
						OutputBuffer& output_buffer,
						Functor&& kernel_func,
						Args&&... args) {

	auto inputs = std::make_tuple();
	auto outputs = std::make_tuple(std::ref(output_buffer));

	return launch_cpu_kernel(resource,
							 thread_count,
							 inputs,
							 outputs,
							 config,
							 std::forward<Functor>(kernel_func),
							 std::forward<Args>(args)...);
}

/**
 * @brief CPU kernel launcher - single input, single output buffer
 */
template<typename InputBuffer, typename OutputBuffer, typename Functor, typename... Args>
Event launch_cpu_kernel(const Resource& resource,
						size_t thread_count,
						const KernelConfig& config,
						InputBuffer& input_buffer,
						OutputBuffer& output_buffer,
						Functor&& kernel_func,
						Args&&... args) {

	auto inputs = std::make_tuple(std::ref(input_buffer));
	auto outputs = std::make_tuple(std::ref(output_buffer));

	return launch_cpu_kernel(resource,
							 thread_count,
							 inputs,
							 outputs,
							 config,
							 std::forward<Functor>(kernel_func),
							 std::forward<Args>(args)...);
}

// ============================================================================
// Backend-Specific Implementations
// ============================================================================

#ifdef USE_CUDA
// Forward declarations
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

// Implementation
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
	local_config.auto_configure(thread_count, resource);

	sycl::range<1> global_range(local_config.grid_size.x * local_config.block_size.x);
	sycl::range<1> local_range(local_config.block_size.x);
	sycl::nd_range<1> execution_range(global_range, local_range);

	auto& queue = SYCL::SYCLManager::get_device(resource.id).get_next_queue();

	auto sycl_event = queue.get().submit([&](sycl::handler& h) {
		h.depends_on(config.dependencies.get_sycl_events());

		auto input_pointers = get_buffer_pointers(inputs);
		auto output_pointers = get_buffer_pointers(outputs);

		auto kernel_args = std::tuple_cat(input_pointers,
										  output_pointers,
										  std::make_tuple(std::forward<Args>(args)...));

		h.parallel_for(execution_range, [=](sycl::nd_item<1> item) {
			size_t i = item.get_global_id(0);
			if (i < thread_count) {
				std::apply([&](auto&&... unpacked_args) { kernel_func(i, unpacked_args...); },
						   kernel_args);
			}
		});
	});

	if (!config.async) {
		sycl_event.wait();
	}

	return Event(sycl_event, resource);
}
#endif

#ifdef USE_METAL
template<typename... Args>
Event launch_metal_kernel(const Resource& resource,
						  size_t thread_count,
						  const KernelConfig& config,
						  const std::string& kernel_name,
						  Args&&... args) {

	config.dependencies.wait_all();

	MTL::ComputePipelineState* pipeline =
		METAL::METALManager::get_compute_pipeline_state(kernel_name);

	auto& device = METAL::METALManager::get_current_device();
	auto& queue = device.get_next_queue();

	void* cmd_buffer_ptr = queue.create_command_buffer();
	MTL::CommandBuffer* cmd_buffer = static_cast<MTL::CommandBuffer*>(cmd_buffer_ptr);
	MTL::ComputeCommandEncoder* encoder = cmd_buffer->computeCommandEncoder();

	encoder->setComputePipelineState(pipeline);

	// Handle buffer arguments dynamically
	int buffer_index = 0;
	auto bind_arg = [&](auto&& arg) {
		using ArgType = std::decay_t<decltype(arg)>;
		if constexpr (is_device_buffer_v<ArgType>) {
			EventList deps;
			auto ptr = arg.get_write_access(deps);
			void* metal_buffer = METAL::METALManager::get_metal_buffer_from_ptr(ptr);
			encoder->setBuffer(static_cast<const MTL::Buffer*>(metal_buffer), 0, buffer_index++);
		} else if constexpr (std::is_arithmetic_v<ArgType>) {
			// Handle scalar arguments as needed
		}
	};

	(bind_arg(std::forward<Args>(args)), ...);

	KernelConfig local_config = config;
	local_config.auto_configure(thread_count);

	MTL::Size grid_size = MTL::Size::Make(local_config.grid_size.x,
										  local_config.grid_size.y,
										  local_config.grid_size.z);

	NS::UInteger max_threads_per_group = pipeline->maxTotalThreadsPerThreadgroup();
	NS::UInteger desired_threads_per_group = config.block_size.x;
	NS::UInteger final_threads_per_group =
		std::min(desired_threads_per_group, max_threads_per_group);

	MTL::Size threadgroup_size = MTL::Size::Make(final_threads_per_group, 1, 1);

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

// ============================================================================
// Helper Functions
// ============================================================================

template<typename Tuple, std::size_t... I>
auto extract_buffer_pointers_impl(Tuple& tuple, std::index_sequence<I...>) {
	return std::make_tuple(std::get<I>(tuple).data()...);
}

template<typename Tuple>
auto extract_buffer_pointers(Tuple& tuple) {
	return extract_buffer_pointers_impl(tuple,
										std::make_index_sequence<std::tuple_size_v<Tuple>>{});
}

// ============================================================================
// Utility Functors
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

// ============================================================================
// High-Level Utility Functions
// ============================================================================

template<typename Backend, typename T>
Event copy_async(const Resource& resource,
				 const DeviceBuffer<T>& source,
				 DeviceBuffer<T>& destination,
				 const KernelConfig& config = {}) {

	if (source.size() != destination.size()) {
		throw std::runtime_error("Buffer size mismatch in copy_async");
	}

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

	return launch_kernel<Backend>(resource,
								  buffer.size(),
								  std::tie(),
								  std::tie(buffer),
								  config,
								  FillFunctor<T>{value});
}

// ============================================================================
// Kernel Chaining
// ============================================================================

template<typename Backend>
class KernelChain {
  private:
	const Resource& resource_;
	EventList events_;

  public:
	explicit KernelChain(const Resource& resource) : resource_(resource) {}

	template<typename InputTuple, typename OutputTuple, typename Functor, typename... Args>
	KernelChain& then(size_t thread_count,
					  InputTuple& inputs,
					  OutputTuple& outputs,
					  Functor&& kernel,
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
														std::forward<Functor>(kernel),
														std::forward<Args>(args)...);

		events_.clear();
		events_.add(completion_event);
		return *this;
	}

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

namespace ARBD {

// KernelConfig implementation
inline void KernelConfig::auto_configure(size_t thread_count, const Resource& resource) {
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

inline void KernelConfig::validate_block_size(const Resource& resource) {
#ifdef USE_SYCL
	if (resource.type == ResourceType::SYCL) {
		try {
			auto& device = SYCL::SYCLManager::get_device(resource.id);
			size_t max_work_group_size =
				device.get_device().get_info<sycl::info::device::max_work_group_size>();

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

			LOGDEBUG(
				"SYCL block size clamped to ({}, {}, {}) for device with max work-group size {}",
				block_size.x,
				block_size.y,
				block_size.z,
				max_work_group_size);

		} catch (const sycl::exception& e) {
			LOGWARN("Failed to query SYCL device limits, using default block size: {}", e.what());
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

			LOGDEBUG(
				"CUDA block size clamped to ({}, {}, {}) for device with max threads per block {}",
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
		const size_t max_cpu_threads = std::thread::hardware_concurrency() * 4;
		size_t total_threads = block_size.x * block_size.y * block_size.z;
		if (total_threads > max_cpu_threads) {
			block_size.x = std::min(block_size.x, max_cpu_threads);
			block_size.y = 1;
			block_size.z = 1;
		}
	}
}

} // namespace ARBD
