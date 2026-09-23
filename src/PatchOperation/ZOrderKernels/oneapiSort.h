#pragma once
#if defined(PROJECT_USES_SYCL_ICPX) || defined(USE_SYCL_ICPX)
#include "Backend/Buffer.h"
#include "Header.h"
#include "DeviceRadix.h"

#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <sycl/sycl.hpp>
namespace MARS {

inline void sort_morton_codes_oneapi(const Resource& resource,
									 morton_t* morton_codes,
									 uint32_t* sorted_indices,
									 size_t num_particles) {
	sycl::queue& q = resource.get_sycl_queue(0);
	auto policy = oneapi::dpl::execution::make_device_policy(q);

	oneapi::dpl::stable_sort_by_key(policy,
									morton_codes,
									morton_codes + num_particles,
									sorted_indices);
}

} // namespace MARS
#endif
