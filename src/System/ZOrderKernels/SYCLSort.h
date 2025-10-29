#pragma once

#include "Backend/Buffer.h"
#include "Header.h"

#include <sycl/sycl.hpp>

namespace ARBD {

/**
 * @brief Complete radix sort implementation for Morton codes using SYCL
 *
 * This implements a multi-pass radix sort where each pass sorts on 8 bits.
 * For 64-bit Morton codes, this requires 8 passes (bits 0-7, 8-15, ..., 56-63).
 *
 * Each pass consists of:
 * 1. Local histogram computation per work-group
 * 2. Global histogram collection and prefix sum
 * 3. Scatter phase to reorder elements
 */
inline void sort_morton_codes_sycl(DeviceBuffer<morton_t>& morton_codes,
								   DeviceBuffer<uint32_t>& sorted_indices,
								   size_t num_particles,
								   const Resource& resource) {

	if (num_particles == 0)
		return;

	sycl::queue q = resource.get_sycl_queue(0);

	// Initialize sorted_indices with sequential values [0, 1, 2, ...]
	q.submit([&](sycl::handler& cgh) {
		 auto indices_ptr = sorted_indices.data();
		 cgh.parallel_for(sycl::range<1>(num_particles),
						  [=](sycl::id<1> idx) { indices_ptr[idx] = static_cast<uint32_t>(idx); });
	 }).wait();

	const size_t local_size = 256;
	const size_t num_bins = 256; // 2^8 = 256 bins per pass
	const size_t num_passes = 8; // 64 bits / 8 bits per pass

	// Calculate number of work-groups
	size_t num_groups = (num_particles + local_size - 1) / local_size;

	// Temporary buffers for double buffering
	DeviceBuffer<morton_t> temp_morton_codes(num_particles, resource);
	DeviceBuffer<uint32_t> temp_indices(num_particles, resource);

	// Global histogram buffer: num_groups * num_bins
	DeviceBuffer<uint32_t> global_histogram(num_groups * num_bins, resource);
	DeviceBuffer<uint32_t> histogram_scan(num_bins, resource);

	for (size_t pass = 0; pass < num_passes; ++pass) {
		const int bit_shift = static_cast<int>(pass * 8);

		// Select input/output buffers for ping-pong
		auto& input_morton = (pass % 2 == 0) ? morton_codes : temp_morton_codes;
		auto& output_morton = (pass % 2 == 0) ? temp_morton_codes : morton_codes;
		auto& input_indices = (pass % 2 == 0) ? sorted_indices : temp_indices;
		auto& output_indices = (pass % 2 == 0) ? temp_indices : sorted_indices;

		// Phase 1: Local histogram computation
		sycl::nd_range<1> hist_range(num_groups * local_size, local_size);
		q.submit([&](sycl::handler& cgh) {
			 auto local_histogram = sycl::local_accessor<uint32_t, 1>(num_bins, cgh);
			 auto input_morton_ptr = input_morton.data();
			 auto global_hist_ptr = global_histogram.data();

			 cgh.parallel_for(hist_range, [=](sycl::nd_item<1> item) {
				 size_t global_id = item.get_global_id(0);
				 size_t local_id = item.get_local_id(0);
				 size_t group_id = item.get_group(0);

				 // Initialize local histogram
				 for (size_t i = local_id; i < num_bins; i += local_size) {
					 local_histogram[i] = 0;
				 }
				 item.barrier(sycl::access::fence_space::local_space);

				 // Compute histogram for this work-group
				 if (global_id < num_particles) {
					 morton_t key = input_morton_ptr[global_id];
					 uint32_t digit = (key >> bit_shift) & 0xFF;

					 auto atomic_bin = sycl::atomic_ref<uint32_t,
														sycl::memory_order::relaxed,
														sycl::memory_scope::work_group,
														sycl::access::address_space::local_space>(
						 local_histogram[digit]);
					 atomic_bin.fetch_add(1);
				 }
				 item.barrier(sycl::access::fence_space::local_space);

				 // Write local histogram to global memory
				 for (size_t i = local_id; i < num_bins; i += local_size) {
					 global_hist_ptr[group_id * num_bins + i] = local_histogram[i];
				 }
			 });
		 }).wait();

		// Phase 2: Global histogram reduction and prefix sum
		// Phase 2a: Global histogram reduction and prefix sum (unchanged)
		// This kernel populates 'global_hist_ptr' with local prefix sums
		// and 'scan_ptr' with the total count for each bin.
		q.submit([&](sycl::handler& cgh) {
			 auto global_hist_ptr = global_histogram.data();
			 auto scan_ptr = histogram_scan.data();

			 cgh.parallel_for(sycl::range<1>(num_bins), [=](sycl::id<1> bin) {
				 uint32_t sum = 0;
				 for (size_t g = 0; g < num_groups; ++g) {
					 uint32_t count = global_hist_ptr[g * num_bins + bin];
					 global_hist_ptr[g * num_bins + bin] = sum; // Convert to prefix sum
					 sum += count;
				 }
				 scan_ptr[bin] = sum; // Total count for this bin
			 });
		 }).wait();

		// Phase 2b: Scan the 256-element bin totals (fast, serial is fine)
		q.submit([&](sycl::handler& cgh) {
			 auto scan_ptr = histogram_scan.data();
			 cgh.single_task([=]() {
				 uint32_t running_sum = 0;
				 for (size_t bin = 0; bin < num_bins; ++bin) {
					 uint32_t bin_total = scan_ptr[bin];
					 scan_ptr[bin] = running_sum; // 'scan_ptr' now holds exclusive scan
					 running_sum += bin_total;
				 }
			 });
		 }).wait();

		// Phase 2c: Add global bin offsets to local offsets (in parallel)
		sycl::range<2> add_range(num_groups, num_bins);
		q.submit([&](sycl::handler& cgh) {
			 auto scan_ptr = histogram_scan.data();
			 auto global_hist_ptr = global_histogram.data();

			 cgh.parallel_for(add_range, [=](sycl::id<2> idx) {
				 size_t g = idx.get(0);	  // Work-group ID
				 size_t bin = idx.get(1); // Bin ID

				 // Add the global offset for 'bin' to the local
				 // offset for 'g' in that 'bin'.
				 global_hist_ptr[g * num_bins + bin] += scan_ptr[bin];
			 });
		 }).wait();

		// Phase 3: Scatter elements to sorted positions
		q.submit([&](sycl::handler& cgh) {
			 auto local_offsets = sycl::local_accessor<uint32_t, 1>(num_bins, cgh);
			 auto input_morton_ptr = input_morton.data();
			 auto input_indices_ptr = input_indices.data();
			 auto output_morton_ptr = output_morton.data();
			 auto output_indices_ptr = output_indices.data();
			 auto global_hist_ptr = global_histogram.data();

			 cgh.parallel_for(hist_range, [=](sycl::nd_item<1> item) {
				 size_t global_id = item.get_global_id(0);
				 size_t local_id = item.get_local_id(0);
				 size_t group_id = item.get_group(0);

				 // Load group's starting positions for each bin
				 for (size_t i = local_id; i < num_bins; i += local_size) {
					 local_offsets[i] = global_hist_ptr[group_id * num_bins + i];
				 }
				 item.barrier(sycl::access::fence_space::local_space);

				 if (global_id < num_particles) {
					 morton_t key = input_morton_ptr[global_id];
					 uint32_t index = input_indices_ptr[global_id];
					 uint32_t digit = (key >> bit_shift) & 0xFF;

					 // Atomically get position and increment
					 auto atomic_offset =
						 sycl::atomic_ref<uint32_t,
										  sycl::memory_order::relaxed,
										  sycl::memory_scope::work_group,
										  sycl::access::address_space::local_space>(
							 local_offsets[digit]);
					 uint32_t pos = atomic_offset.fetch_add(1);

					 // Write to sorted position
					 output_morton_ptr[pos] = key;
					 output_indices_ptr[pos] = index;
				 }
			 });
		 }).wait();
	}

	// If we ended with data in temp buffers, copy back to original buffers
	if (num_passes % 2 == 1) {
		q.submit([&](sycl::handler& cgh) {
			 auto temp_morton_ptr = temp_morton_codes.data();
			 auto temp_indices_ptr = temp_indices.data();
			 auto morton_ptr = morton_codes.data();
			 auto indices_ptr = sorted_indices.data();

			 cgh.parallel_for(sycl::range<1>(num_particles), [=](sycl::id<1> idx) {
				 morton_ptr[idx] = temp_morton_ptr[idx];
				 indices_ptr[idx] = temp_indices_ptr[idx];
			 });
		 }).wait();
	}
}

} // namespace ARBD
