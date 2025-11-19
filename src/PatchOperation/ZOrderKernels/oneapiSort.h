#pragma once
#ifdef USE_SYCL_ICPX
#include "Backend/Buffer.h"
#include "Header.h"

#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/experimental/kernel_templates>
#include <sycl/sycl.hpp>
namespace ARBD {

void sort_morton_codes_oneapi(const DeviceBuffer<morton_t>& morton_codes,
							  const DeviceBuffer<uint32_t>& sorted_indices,
							  size_t num_particles_,
							  Resource& resource) {
	sycl::queue q = resource.get_sycl_queue(0);
	auto policy = oneapi::dpl::execution::make_device_policy(q);

	const auto zip_begin =
		oneapi::dpl::make_zip_iterator(morton_codes.data(), sorted_indices.data());
	const auto zip_end = zip_begin + num_particles_;
	oneapi::dpl::sort(policy, zip_begin, zip_end, [](const auto& a, const auto& b) {
		return std::get<0>(a) < std::get<0>(b);
	});
}

} // namespace ARBD
#endif
