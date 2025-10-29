#pragma once

#include "Backend/Buffer.h"
#include "Header.h"
#include "Types/Types.h"

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>

namespace ARBD {

void sort_morton_codes_cuda(const DeviceBuffer<morton_t>& morton_codes_,
							const DeviceBuffer<uint32_t>& sorted_indices_,
							size_t num_particles_) {

	// Use Thrust for sorting Morton codes along with indices
	thrust::device_ptr<morton_t> morton_ptr(morton_codes_.data());
	thrust::device_ptr<uint32_t> indices_ptr(sorted_indices_.data());

	// Create zip iterator to sort both arrays together
	auto zip_begin = thrust::make_zip_iterator(thrust::make_tuple(morton_ptr, indices_ptr));
	auto zip_end = thrust::make_zip_iterator(
		thrust::make_tuple(morton_ptr + num_particles_, indices_ptr + num_particles_));

	// Sort by Morton code (first element of tuple)
	thrust::sort(
		zip_begin,
		zip_end,
		[](const thrust::tuple<morton_t, uint32_t>& a, const thrust::tuple<morton_t, uint32_t>& b) {
			return thrust::get<0>(a) < thrust::get<0>(b);
		});
}

} // namespace ARBD
