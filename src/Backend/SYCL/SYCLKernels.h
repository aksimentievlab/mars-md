#pragma once
#include "Header.h"
#ifdef USE_SYCL
#include "../Buffer.h"
#include "../Events.h"
#include "../KernelConfig.h"
#include "../Resource.h"
#include "Header.h"
#include "SYCLManager.h"
#include <sycl/sycl.hpp>

namespace ARBD {

/**
 * @brief Streamlined SYCL kernel launcher - new structure matching CUDA
 * Eliminates tuple overhead for better multi-GPU performance
 */
template<typename Functor, typename... Args>
Event launch_sycl_kernel(const Resource& resource,
						 const KernelConfig& config,
						 Functor kernel_func,
						 Args... args) {

	// Auto-configure if needed
	KernelConfig local_config = config;
	// Auto-configuration should be done by caller before calling this function

	// Calculate SYCL execution ranges (3D) - map to maintain x=fastest varying like CUDA/Metal
	// SYCL: last dimension (index 2) is fastest, so map x->2, y->1, z->0
	sycl::range<3> global_range(local_config.grid_size.z * local_config.block_size.z,
								local_config.grid_size.y * local_config.block_size.y,
								local_config.grid_size.x * local_config.block_size.x);
	sycl::range<3> local_range(local_config.block_size.z,
							   local_config.block_size.y,
							   local_config.block_size.x);
	sycl::nd_range<3> execution_range(global_range, local_range);

	// Get queue from config or resource
	auto* queue_wrapper_ptr = static_cast<ARBD::SYCL::Queue*>(resource.get_stream());
	sycl::queue& queue = queue_wrapper_ptr->get();

	// Pre-extract all buffer pointers to avoid capture issues
	auto extracted_ptr_args = [&]() { return std::make_tuple(get_buffer_pointer(args)...); }();

	// Submit kernel with dependency handling
	auto sycl_event = queue.submit([&](sycl::handler& h) {
		// Handle dependencies
		if (!config.dependencies.empty()) {
			h.depends_on(config.dependencies.get_sycl_events());
		}

		// Launch kernel with pre-extracted pointers
		h.parallel_for(execution_range, [=](sycl::nd_item<3> item) {
			// Extract coordinates: SYCL maps z->0, y->1, x->2 to keep x fastest varying
			idx_t gx = static_cast<idx_t>(item.get_global_id(2));
			idx_t gy = static_cast<idx_t>(item.get_global_id(1));
			idx_t gz = static_cast<idx_t>(item.get_global_id(0));

			if (gx < local_config.problem_size.x && gy < local_config.problem_size.y &&
				gz < local_config.problem_size.z) {
				idx_t i =
					(gz * local_config.problem_size.y + gy) * local_config.problem_size.x + gx;
				std::apply([&](auto... ptrs) { kernel_func(i, ptrs...); }, extracted_ptr_args);
			}
		});
	});

	// Sync if requested
	if (config.sync) {
		sycl_event.wait();
	}

	return Event(sycl_event, resource);
}
} // namespace ARBD
#endif
