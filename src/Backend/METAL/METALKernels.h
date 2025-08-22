#pragma once
#ifdef USE_METAL
#include "../Events.h"
#include "../Header.h"
#include "../KernelConfig.h"
#include "../Resource.h"
#include "METALManager.h"
#include "Metal/Metal.hpp"

namespace ARBD {
/**
 * @brief Name-based kernel launcher (for Metal), does not work.
 */

template<typename InputTuple, typename OutputTuple, typename KernelName, typename... Args>
std::enable_if_t<is_string_v<KernelName>, Event> launch_kernel(const Resource& resource,
															   idx_t thread_count,
															   const InputTuple& inputs,
															   const OutputTuple& outputs,
															   const KernelConfig& config,
															   const std::string& kernel_name,
															   Args&&... args) {
	switch (resource.type) {
#ifdef USE_METAL
	case ResourceType::METAL:
		return launch_metal_kernel(resource,
								   thread_count,
								   inputs,
								   outputs,
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

// Helper functions for buffer binding
template<typename Tuple, std::idx_t... I>
void bind_tuple_to_encoder_impl(MTL::ComputeCommandEncoder* encoder,
								const Tuple& tuple,
								uint32_t& buffer_index,
								std::index_sequence<I...>) {
	((std::get<I>(tuple).bind_to_encoder(encoder, buffer_index++)), ...);
}

template<typename Tuple>
void bind_tuple_to_encoder(MTL::ComputeCommandEncoder* encoder,
						   const Tuple& tuple,
						   uint32_t& buffer_index) {
	bind_tuple_to_encoder_impl(encoder,
							   tuple,
							   buffer_index,
							   std::make_index_sequence<std::tuple_size_v<Tuple>>{});
}

template<typename... Args>
void bind_args_to_encoder(MTL::ComputeCommandEncoder* encoder,
						  uint32_t& buffer_index,
						  Args&&... args) {
	auto bind_arg = [&](auto&& arg) {
		using ArgType = std::decay_t<decltype(arg)>;
		if constexpr (std::is_arithmetic_v<ArgType> || std::is_trivial_v<ArgType>) {
			encoder->setBytes(&arg, sizeof(ArgType), buffer_index++);
		}
	};
	(bind_arg(std::forward<Args>(args)), ...);
}

// Grid configuration helper
struct MetalGridConfig {
	MTL::Size grid_size;
	MTL::Size threadgroup_size;
};

inline MetalGridConfig calculate_metal_grid_config(idx_t thread_count,
												   const KernelConfig& config,
												   MTL::ComputePipelineState* pipeline) {
	MetalGridConfig result;

	// Calculate optimal threadgroup size
	NS::UInteger max_threads = pipeline->maxTotalThreadsPerThreadgroup();
	NS::UInteger desired_threads = config.block_size.x;
	NS::UInteger final_threads = std::min(desired_threads, max_threads);

	result.threadgroup_size = MTL::Size::Make(final_threads, 1, 1);

	// Calculate grid size
	NS::UInteger num_threadgroups = (thread_count + final_threads - 1) / final_threads;
	result.grid_size = MTL::Size::Make(thread_count, 1, 1);

	return result;
}
/**
 * @example
 * Example: Launching a Metal kernel for vector operations
 *
 * @code
 * #include "Backend/Kernels.h"
 * #include "Math/Vector3.h"
 * using namespace ARBD;
 *
 * // Prepare Metal resource and buffers
 * Resource metal_res(ResourceType::METAL, 0);
 * constexpr idx_t n = 16;
 * std::vector<Vector3_t<float>> host_a(n), host_b(n), host_out(n);
 * for (idx_t i = 0; i < n; ++i) {
 *     host_a[i] = Vector3_t<float>(float(i), float(i+1), float(i+2));
 *     host_b[i] = Vector3_t<float>(float(2*i), float(2*i+1), float(2*i+2));
 * }
 * DeviceBuffer<Vector3_t<float>> buf_a(n), buf_b(n), buf_out(n);
 * buf_a.copy_from_host(host_a.data(), n);
 * buf_b.copy_from_host(host_b.data(), n);
 *
 * KernelConfig config;
 * config.async = false;
 * config.grid_size = {n, 1, 1};
 *
 * // Launch the Metal kernel by name
 * Event event = launch_metal_kernel(
 *     metal_res,
 *     n,
 *     std::make_tuple(buf_a, buf_b),
 *     std::forward_as_tuple(buf_out),
 *     config,
 *     "vector_operations_kernel"
 * );
 * event.wait();
 * buf_out.copy_to_host(host_out.data(), n);
 * @endcode
 *
 * Example: Launching a Metal kernel for matrix elementwise multiplication
 *
 * @code
 * #include "Backend/Kernels.h"
 * #include "Math/Matrix3.h"
 * using namespace ARBD;
 *
 * Resource metal_res(ResourceType::METAL, 0);
 * constexpr idx_t n = 4;
 * std::vector<Matrix3_t<float>> host_a(n), host_b(n), host_out(n);
 * for (idx_t i = 0; i < n; ++i) {
 *     Matrix3_t<float> m1, m2;
 *     m1.ex().x = float(i + 1); m1.ex().y = float(i + 2); m1.ex().z = float(i + 3);
 *     m1.ey().x = float(i + 4); m1.ey().y = float(i + 5); m1.ey().z = float(i + 6);
 *     m1.ez().x = float(i + 7); m1.ez().y = float(i + 8); m1.ez().z = float(i + 9);
 *     m2.ex().x = float(2 * (i + 1)); m2.ex().y = float(2 * (i + 2)); m2.ex().z = float(2 * (i +
 * 3)); m2.ey().x = float(2 * (i + 4)); m2.ey().y = float(2 * (i + 5)); m2.ey().z = float(2 * (i +
 * 6)); m2.ez().x = float(2 * (i + 7)); m2.ez().y = float(2 * (i + 8)); m2.ez().z = float(2 * (i +
 * 9)); host_a[i] = m1; host_b[i] = m2;
 * }
 * DeviceBuffer<Matrix3_t<float>> buf_a(n), buf_b(n), buf_out(n);
 * buf_a.copy_from_host(host_a.data(), n);
 * buf_b.copy_from_host(host_b.data(), n);
 *
 * KernelConfig config;
 * config.async = false;
 * config.grid_size = {n, 1, 1};
 *
 * Event event = launch_metal_kernel(
 *     metal_res,
 *     n,
 *     std::make_tuple(buf_a, buf_b),
 *     std::forward_as_tuple(buf_out),
 *     config,
 *     "matrix3_mult_kernel"
 * );
 * event.wait();
 * buf_out.copy_to_host(host_out.data(), n);
 * @endcode
 */

template<typename InputTuple, typename OutputTuple, typename... Args>
Event launch_metal_kernel(const Resource& resource,
						  idx_t thread_count,
						  const InputTuple& inputs,
						  const OutputTuple& outputs,
						  const KernelConfig& config,
						  const std::string& kernel_name,
						  Args&&... args) {

	// Wait for dependencies
	config.dependencies.wait_all();

	// Get Metal components
	auto* pipeline = METAL::Manager::get_compute_pipeline_state(kernel_name);
	if (!pipeline) {
		throw_value_error("Failed to get compute pipeline state for kernel: {}", kernel_name);
	}
	LOGINFO("Got compute pipeline state for kernel: {}", kernel_name);
	auto& device = METAL::Manager::get_current_device();
	auto& queue = device.get_next_queue();

	// Create command buffer and encoder
	void* cmd_buffer_ptr = queue.create_command_buffer();
	auto* cmd_buffer = static_cast<MTL::CommandBuffer*>(cmd_buffer_ptr);
	auto* encoder = cmd_buffer->computeCommandEncoder();

	encoder->setComputePipelineState(pipeline);

	// Enhanced buffer binding with proper error handling
	uint32_t buffer_index = 0;

	// Bind input buffers
	LOGINFO("Binding input buffers to encoder, starting at index {}", buffer_index);
	bind_tuple_to_encoder(encoder, inputs, buffer_index);
	LOGINFO("Input buffers bound, buffer_index is now {}", buffer_index);

	// Bind output buffers
	LOGINFO("Binding output buffers to encoder, starting at index {}", buffer_index);
	bind_tuple_to_encoder(encoder, outputs, buffer_index);
	LOGINFO("Output buffers bound, buffer_index is now {}", buffer_index);

	// Bind additional arguments
	bind_args_to_encoder(encoder, buffer_index, std::forward<Args>(args)...);

	// Configure and dispatch
	auto grid_config = calculate_metal_grid_config(thread_count, config, pipeline);
	LOGINFO("Dispatching Metal kernel: {} with grid size ({}, {}, {}) and threadgroup size ({}, "
			"{}, {})",
			kernel_name,
			grid_config.grid_size.width,
			grid_config.grid_size.height,
			grid_config.grid_size.depth,
			grid_config.threadgroup_size.width,
			grid_config.threadgroup_size.height,
			grid_config.threadgroup_size.depth);
	encoder->dispatchThreads(grid_config.grid_size, grid_config.threadgroup_size);
	encoder->endEncoding();
	LOGINFO("Metal kernel dispatch completed for: {}", kernel_name);
	LOGINFO("Config async setting: {}", config.async);

	// Create and return event
	ARBD::METAL::Event metal_event(cmd_buffer_ptr);
	if (!config.async) {
		LOGINFO("Committing Metal command buffer for kernel: {}", kernel_name);
		metal_event.commit();
		LOGINFO("Waiting for Metal command buffer completion for kernel: {}", kernel_name);
		metal_event.wait();
		LOGINFO("Metal command buffer completed for kernel: {}", kernel_name);

		// Check for command buffer errors
		MTL::CommandBuffer* pCmdBuffer = static_cast<MTL::CommandBuffer*>(cmd_buffer_ptr);
		auto status = pCmdBuffer->status();
		LOGINFO("Command buffer status: {}", (int)status);
		if (status == MTL::CommandBufferStatusError) {
			auto* error = pCmdBuffer->error();
			if (error) {
				LOGERROR("Metal command buffer error: {}",
						 error->localizedDescription()->utf8String());
			}
		}
	} else {
		metal_event.commit();
	}

	return Event(std::move(metal_event), resource);
}

template<typename InputBuffer, typename OutputBuffer, typename... Args>
std::enable_if_t<is_device_buffer_v<OutputBuffer> && !is_device_buffer_v<InputBuffer> &&
					 !is_string_v<InputBuffer>,
				 Event>
launch_metal_kernel(const Resource& resource,
					idx_t thread_count,
					const InputBuffer& input_buffer,
					const OutputBuffer& output_buffer,
					const KernelConfig& config,
					const std::string& kernel_name,
					Args&&... args) {
	auto input = std::make_tuple(std::ref(input_buffer));
	auto output = std::make_tuple(std::ref(output_buffer), std::ref(thread_count));
	return launch_metal_kernel(resource,
							   thread_count,
							   input,
							   output,
							   config,
							   kernel_name,
							   std::forward<Args>(args)...);
}
/*
template<typename... Args>
Event launch_metal_kernel(const Resource& resource,
						  idx_t thread_count,
						  const KernelConfig& config,
						  const std::string& kernel_name,
						  Args&&... args) {

	// --- Step 1: Get Pipeline and Command Encoder ---
	MTL::ComputePipelineState* pipeline =
		METAL::Manager::get_compute_pipeline_state(kernel_name);

	auto& device = METAL::Manager::get_current_device();
	auto& queue = device.get_next_queue();

	MTL::CommandBuffer* cmd_buffer =
		static_cast<MTL::CommandBuffer*>(queue.create_command_buffer());
	MTL::ComputeCommandEncoder* encoder = cmd_buffer->computeCommandEncoder();

	encoder->setComputePipelineState(pipeline);


	* @TODO: implement this
	* for (auto& shared_event_tuple : config.dependencies.get_metal_shared_events()) {
	*	encoder->waitForEvent(std::get<0>(shared_event_tuple), std::get<1>(shared_event_tuple));
	* }


	int buffer_index = 0;
	auto bind_arg = [&](auto&& arg) {
		using ArgType = std::decay_t<decltype(arg)>;
		if constexpr (is_device_buffer_v<ArgType>) {
			void* metal_buffer_ptr = arg.data(); // Assuming .data() gives the raw MTL::Buffer*
			encoder->setBuffer(static_cast<MTL::Buffer*>(metal_buffer_ptr), 0, buffer_index++);
		} else if constexpr (std::is_arithmetic_v<ArgType> || std::is_trivial_v<ArgType>) {
			// Copy the raw bytes of the argument directly into the command stream.
			encoder->setBytes(&arg, sizeof(ArgType), buffer_index++);
		}
	};

	// Use a fold expression to apply the binding logic to every argument.
	(bind_arg(std::forward<Args>(args)), ...);

	// --- Step 4: Dispatch Threads (This part was already correct) ---
	KernelConfig local_config = config;
	local_config.auto_configure(thread_count, resource);

	MTL::Size grid_size = MTL::Size::Make(thread_count, 1, 1);

	NS::UInteger max_threads_per_group = pipeline->maxTotalThreadsPerThreadgroup();
	NS::UInteger final_threads_per_group =
		std::min(static_cast<NS::UInteger>(config.block_size.x), max_threads_per_group);

	MTL::Size threadgroup_size = MTL::Size::Make(final_threads_per_group, 1, 1);

	encoder->dispatchThreads(grid_size, threadgroup_size);
	encoder->endEncoding();

	// --- Step 5: Commit and Return Event (This part was already correct) ---
	ARBD::METAL::Event metal_event(cmd_buffer);
	metal_event.commit();

	if (!config.async) {
		metal_event.wait();
	}

	// Return the generic Event wrapper
	return Event(std::move(metal_event), resource);
}
*/
} // namespace ARBD
#endif
