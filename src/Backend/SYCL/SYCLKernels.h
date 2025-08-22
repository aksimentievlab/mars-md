#pragma once
#ifdef USE_SYCL
#include "../Events.h"
#include "../Header.h"
#include "../KernelConfig.h"
#include "../Resource.h"
#include "SYCLManager.h"
#include <sycl/sycl.hpp>

namespace ARBD {
template<typename InputTuple, typename OutputTuple, typename Functor, typename... Args>
Event launch_sycl_kernel(const Resource& resource,
						 idx_t thread_count,
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

	auto& queue = SYCL::Manager::get_device(resource.id).get_next_queue();

	auto sycl_event = queue.get().submit([&](sycl::handler& h) {
		h.depends_on(config.dependencies.get_sycl_events());

		auto input_pointers = get_buffer_pointers(inputs);
		auto output_pointers = get_buffer_pointers(outputs);

		auto kernel_args = std::tuple_cat(input_pointers,
										  output_pointers,
										  std::make_tuple(std::forward<Args>(args)...));

		h.parallel_for(execution_range, [=](sycl::nd_item<1> item) {
			idx_t i = item.get_global_id(0);
			if (i < thread_count) {
				std::apply([&](auto&&... unpacked_args) { kernel_func(i, unpacked_args...); },
						   kernel_args);
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
