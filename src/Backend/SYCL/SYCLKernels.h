#pragma once
#ifdef USE_SYCL
#include "../Buffer.h"
#include "../Events.h"
#include "../Header.h"
#include "../KernelConfig.h"
#include "../Resource.h"
#include "SYCLManager.h"
#include <sycl/sycl.hpp>

namespace ARBD {

/**
 * @brief Streamlined SYCL kernel launcher - new structure matching CUDA
 * Eliminates tuple overhead for better multi-GPU performance
 */
template<typename Functor, typename... Args>
Event launch_sycl_kernel(const Resource& resource,
						 idx_t thread_count,
						 const KernelConfig& config,
						 Functor kernel_func,
						 Args... args) {
	
	// Auto-configure if needed
	KernelConfig local_config = config;
	local_config.auto_configure(thread_count, resource);

	// Calculate SYCL execution ranges
	sycl::range<1> global_range(local_config.grid_size.x * local_config.block_size.x);
	sycl::range<1> local_range(local_config.block_size.x);
	sycl::nd_range<1> execution_range(global_range, local_range);

	// Get queue from config or resource
	auto* queue_wrapper_ptr = static_cast<ARBD::SYCL::Queue*>(local_config.get_queue(resource));
	sycl::queue& queue = queue_wrapper_ptr->get();

	// Pre-extract all buffer pointers to avoid capture issues
	auto extracted_ptr_args = [&](){
		return std::make_tuple(get_buffer_pointer(args)...);
	}();

	// Submit kernel with dependency handling
	auto sycl_event = queue.submit([&](sycl::handler& h) {
		// Handle dependencies
		if (!config.dependencies.empty()) {
			h.depends_on(config.dependencies.get_sycl_events());
		}

		// Launch kernel with pre-extracted pointers
		h.parallel_for(execution_range, [=](sycl::nd_item<1> item) {
			idx_t i = item.get_global_id(0);
			if (i < thread_count) {
				// Apply pre-extracted arguments - minimal overhead
				std::apply([&](auto... ptrs) {
					kernel_func(i, ptrs...);
				}, extracted_ptr_args);
			}
		});
	});

	// Sync if requested
	if (config.sync) {
		sycl_event.wait();
	}

	return Event(sycl_event, resource);
}

/**
 * @brief Legacy tuple-based interface for backward compatibility
 * @deprecated Use the streamlined version above for better performance
 */
template<typename InputTuple, typename OutputTuple, typename Functor, typename... Args>
Event launch_sycl_kernel(const Resource& resource,
						 idx_t thread_count,
						 const InputTuple& inputs,
						 const OutputTuple& outputs,
						 const KernelConfig& config,
						 Functor&& kernel_func,
						 Args&&... args) {

	// Auto-configure if needed
	KernelConfig local_config = config;
	local_config.auto_configure(thread_count, resource);

	// Calculate SYCL execution ranges
	sycl::range<1> global_range(local_config.grid_size.x * local_config.block_size.x);
	sycl::range<1> local_range(local_config.block_size.x);
	sycl::nd_range<1> execution_range(global_range, local_range);

	// Get queue
	auto* queue_wrapper_ptr = static_cast<ARBD::SYCL::Queue*>(local_config.get_queue(resource));
	sycl::queue& queue = queue_wrapper_ptr->get();

	// Submit kernel
	auto sycl_event = queue.submit([&](sycl::handler& h) {
		// Handle dependencies
		if (!config.dependencies.empty()) {
			h.depends_on(config.dependencies.get_sycl_events());
		}

		// Extract buffer pointers from tuples
		auto input_ptrs = get_buffer_tuples(inputs);
		auto output_ptrs = get_buffer_tuples(outputs);

		// Combine all arguments
		auto all_args = std::tuple_cat(input_ptrs, output_ptrs, std::make_tuple(std::forward<Args>(args)...));

		h.parallel_for(execution_range, [=](sycl::nd_item<1> item) {
			idx_t i = item.get_global_id(0);
			if (i < thread_count) {
				std::apply([&](auto&&... unpacked_args) { 
					kernel_func(i, unpacked_args...);
				}, all_args);
			}
		});
	});

	if (config.sync) {
		sycl_event.wait();
	}

	return Event(sycl_event, resource);
}

} // namespace ARBD
#endif
