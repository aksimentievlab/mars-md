#pragma once

#include "../MortonCode.h"
#include "Header.h"
#include "Types/Types.h"
#include "Types/Vector3.h"

namespace ARBD {

/**
 * @brief Kernel for computing optimal search range per particle based on local density
 *
 * Analyzes the local particle density around each particle to determine
 * an optimal search range for neighbor finding, improving efficiency.
 */
struct ComputeSearchRangeKernel {
	const Vector3* sorted_positions;
	const morton_t* sorted_morton_codes;
	uint32_t* search_ranges;
	float cutoff_squared;
	size_t num_particles;
	size_t default_search_range;
	float density_factor; // Scales search range based on density

	HOST DEVICE void operator()(idx_t i) const {
		if (i >= num_particles)
			return;

		Vector3 pos_i = sorted_positions[i];
		morton_t code_i = sorted_morton_codes[i];

		// Sample local density by counting neighbors in a small fixed range
		const size_t sample_range = 16; // Small fixed range for density estimation
		size_t local_neighbors = 0;

		size_t start = (i >= sample_range) ? i - sample_range : 0;
		size_t end = (i + sample_range < num_particles) ? i + sample_range : num_particles;

		for (size_t j = start; j < end; ++j) {
			if (i == j)
				continue;

			Vector3 pos_j = sorted_positions[j];
			Vector3 dr = pos_j - pos_i;
			float dist_squared = dr.length2();

			if (dist_squared <= cutoff_squared) {
				local_neighbors++;
			}
		}

		// Compute adaptive search range based on local density
		// Higher density -> smaller search range needed
		// Lower density -> larger search range needed
		float density_ratio = static_cast<float>(local_neighbors) / (2.0f * sample_range);
		float adaptive_factor =
			1.0f / (density_ratio * density_factor + 0.1f); // Avoid division by zero

		size_t adaptive_range = static_cast<size_t>(default_search_range * adaptive_factor);

		// Clamp to reasonable bounds
		adaptive_range = (adaptive_range < 16) ? 16 : adaptive_range;
		adaptive_range = (adaptive_range > 256) ? 256 : adaptive_range;

		search_ranges[i] = static_cast<uint32_t>(adaptive_range);
	}
};

/**
 * @brief Enhanced Z-order neighbor kernel with adaptive search ranges and optimizations
 *
 * Uses per-particle adaptive search ranges and improved early termination
 * to optimize neighbor finding performance.
 */
struct AdaptiveZOrderNeighborKernel {
	const Vector3* sorted_positions;
	const morton_t* sorted_morton_codes;
	const uint32_t* sorted_to_original;
	const uint32_t* search_ranges; // Per-particle adaptive search ranges
	int2* neighbor_pairs;
	uint32_t* pair_count;
	float cutoff_squared;
	size_t num_particles;
	size_t max_pairs;

	HOST DEVICE void operator()(idx_t i) const {
		if (i >= num_particles)
			return;

		Vector3 pos_i = sorted_positions[i];
		morton_t code_i = sorted_morton_codes[i];
		uint32_t search_range = search_ranges[i];

		// Adaptive search range for this particle
		size_t start = (i >= search_range) ? i - search_range : 0;
		size_t end = (i + search_range < num_particles) ? i + search_range : num_particles;

		// Use local pair counting to reduce atomic contention
		uint32_t local_pairs = 0;
		int2 local_pair_buffer[32]; // Local buffer to batch pair writes

		for (size_t j = start; j < end; ++j) {
			if (i >= j)
				continue; // Avoid double counting and self-interaction

			Vector3 pos_j = sorted_positions[j];
			Vector3 dr = pos_j - pos_i;
			float dist_squared = dr.length2();

			if (dist_squared <= cutoff_squared) {
				// Found a neighbor pair
				uint32_t original_i = sorted_to_original[i];
				uint32_t original_j = sorted_to_original[j];

				// Buffer pairs locally to reduce global memory access
				if (local_pairs < 32) {
					local_pair_buffer[local_pairs] = int2(original_i, original_j);
					local_pairs++;
				} else {
					// Flush local buffer when full
					flush_local_pairs(local_pair_buffer,
									  local_pairs,
									  pair_count,
									  neighbor_pairs,
									  max_pairs);
					local_pair_buffer[0] = int2(original_i, original_j);
					local_pairs = 1;
				}
			}

			// Enhanced early termination using Morton code hierarchy
			morton_t code_j = sorted_morton_codes[j];
			if (j > i) {
				morton_t code_diff = code_j - code_i;

				// Use multi-level Morton code distance for better early termination
				if (code_diff > get_morton_threshold(search_range)) {
					break;
				}
			}
		}

		// Flush remaining local pairs
		if (local_pairs > 0) {
			flush_local_pairs(local_pair_buffer,
							  local_pairs,
							  pair_count,
							  neighbor_pairs,
							  max_pairs);
		}
	}

  private:
	HOST DEVICE void flush_local_pairs(const int2* local_buffer,
									   uint32_t count,
									   uint32_t* global_count,
									   int2* global_pairs,
									   size_t max_pairs) const {
#ifdef __CUDA_ARCH__
		uint32_t start_idx = atomicAdd(global_count, count);
#elif defined(__SYCL_DEVICE_ONLY__)
		sycl::atomic_ref<uint32_t,
						 sycl::memory_order::relaxed,
						 sycl::memory_scope::device,
						 sycl::access::address_space::global_space>
			atomic_add(*global_count);
		uint32_t start_idx = atomic_add.fetch_add(count);
#else
		uint32_t start_idx = *global_count;
		*global_count += count;
#endif

		// Copy local pairs to global memory
		for (uint32_t k = 0; k < count && (start_idx + k) < max_pairs; ++k) {
			global_pairs[start_idx + k] = local_buffer[k];
		}
	}

	HOST DEVICE morton_t get_morton_threshold(uint32_t search_range) const {
		// Adaptive threshold based on search range
		// Larger search ranges allow larger Morton code differences
		return 0x100000 + (search_range << 12);
	}
};

/**
 * @brief Hierarchical Morton neighbor kernel using multi-level search
 *
 * Uses a hierarchical approach to Morton code searching, first finding
 * coarse-level neighbors then refining to find precise neighbors.
 */
struct HierarchicalMortonNeighborKernel {
	const Vector3* sorted_positions;
	const morton_t* sorted_morton_codes;
	const uint32_t* sorted_to_original;
	int2* neighbor_pairs;
	uint32_t* pair_count;
	float cutoff_squared;
	size_t num_particles;
	size_t max_pairs;

	HOST DEVICE void operator()(idx_t i) const {
		if (i >= num_particles)
			return;

		Vector3 pos_i = sorted_positions[i];
		morton_t code_i = sorted_morton_codes[i];

		// Multi-level search using Morton code hierarchy
		// Level 1: Coarse search using high-order bits
		size_t coarse_range = find_coarse_range(i, code_i);

		// Level 2: Fine search within coarse range
		search_in_range(i, pos_i, code_i, coarse_range);
	}

  private:
	HOST DEVICE size_t find_coarse_range(idx_t i, morton_t code_i) const {
		// Use high-order bits for coarse-level search
		morton_t coarse_code = code_i & 0xFFFFF000; // Keep top 20 bits

		// Binary search for range with similar coarse codes
		size_t range = 64; // Start with default range

		// Expand range if needed based on Morton code distribution
		if (i > 0 && i < num_particles - 1) {
			morton_t prev_coarse = sorted_morton_codes[i - 1] & 0xFFFFF000;
			morton_t next_coarse = sorted_morton_codes[i + 1] & 0xFFFFF000;

			if (prev_coarse != coarse_code || next_coarse != coarse_code) {
				range *= 2; // Expand range at boundaries
			}
		}

		return range;
	}

	HOST DEVICE void
	search_in_range(idx_t i, const Vector3& pos_i, morton_t code_i, size_t range) const {
		size_t start = (i >= range) ? i - range : 0;
		size_t end = (i + range < num_particles) ? i + range : num_particles;

		for (size_t j = start; j < end; ++j) {
			if (i >= j)
				continue;

			Vector3 pos_j = sorted_positions[j];
			Vector3 dr = pos_j - pos_i;
			float dist_squared = dr.length2();

			if (dist_squared <= cutoff_squared) {
				uint32_t original_i = sorted_to_original[i];
				uint32_t original_j = sorted_to_original[j];

#ifdef __CUDA_ARCH__
				uint32_t pair_idx = atomicAdd(pair_count, 1);
#elif defined(__SYCL_DEVICE_ONLY__)
				sycl::atomic_ref<uint32_t,
								 sycl::memory_order::relaxed,
								 sycl::memory_scope::device,
								 sycl::access::address_space::global_space>
					atomic_add(*pair_count);
				uint32_t pair_idx = atomic_add.fetch_add(1);
#else
				uint32_t pair_idx = (*pair_count)++;
#endif

				if (pair_idx < max_pairs) {
					neighbor_pairs[pair_idx] = int2(original_i, original_j);
				}
			}

			// Fine-grained early termination
			morton_t code_j = sorted_morton_codes[j];
			if (j > i && (code_j - code_i) > get_fine_threshold()) {
				break;
			}
		}
	}

	HOST DEVICE morton_t get_fine_threshold() const {
		// Fine-grained threshold for hierarchical search
		return 0x10000; // More restrictive than basic kernel
	}
};

} // namespace ARBD
