#pragma once
/**
 * @file ZOrderNeighbor_FINAL_FIX.h
 * @brief FULLY FIXED version of ZOrderNeighborKernel
 *
 * BUGS FIXED:
 * 1. Changed `if (i >= j) continue` to `if (i == j) continue`
 * 2. Added `if (original_i >= original_j) continue` BEFORE distance check
 * 3. Moved original_i extraction before loop (optimization)
 *
 * KEY INSIGHT: Check original indices BEFORE expensive distance calculation!
 */

#include "../ZOrderKernels/MortonCode.h"
#include "Header.h"
#include "Types/Types.h"
#include "Types/Vector3.h"

namespace ARBD {

/**
 * @brief Kernel for Z-order based neighbor finding
 * Uses the spatial locality of Morton codes to efficiently find neighbors
 */
struct ZOrderNeighborKernel {
	const Vector3* sorted_positions;	 // Positions in Morton-sorted order
	const morton_t* sorted_morton_codes; // Morton codes in sorted order
	const uint32_t* sorted_to_original;	 // Maps sorted index -> original particle ID
	int2* neighbor_pairs;				 // Output: (original_i, original_j) pairs
	uint32_t* pair_count;				 // Atomic counter for pairs
	float cutoff_squared;
	size_t num_particles;
	size_t max_pairs;

	DEVICE void operator()(idx_t i) const {
		if (i >= num_particles)
			return;

		Vector3 pos_i = sorted_positions[i];
		morton_t code_i = sorted_morton_codes[i];
		uint32_t original_i = sorted_to_original[i]; // Get ONCE before loop

		// Search backward and forward in Morton order for neighbors
		// Due to Z-order properties, nearby particles in 3D space
		// are likely to be close in Morton order

		// Search range - could be tuned based on system characteristics
		const size_t search_range = 64; // Search ±64 positions in sorted order

		size_t start = (i >= search_range) ? i - search_range : 0;
		size_t end = (i + search_range < num_particles) ? i + search_range : num_particles;

		for (size_t j = start; j < end; ++j) {
			// FIX #1: Only skip self-interaction, not all backward neighbors
			if (i == j)
				continue; // Avoid self-interaction in sorted space

			// FIX #2: Get original indices EARLY
			uint32_t original_j = sorted_to_original[j];

			// FIX #3: Only store each pair once (i < j in ORIGINAL indices)
			// CRITICAL: Do this BEFORE the expensive distance check!
			// This prevents storing both (i,j) and (j,i)
			if (original_i >= original_j)
				continue;

			// Now do expensive distance calculation
			Vector3 pos_j = sorted_positions[j];
			Vector3 dr = pos_j - pos_i;
			float dist_squared = dr.length2();

			if (dist_squared <= cutoff_squared) {
				// Found a neighbor pair - store it!

#ifdef USE_CUDA
				uint32_t pair_idx = atomicAdd(pair_count, 1);
#elif defined(USE_SYCL)
				sycl::atomic_ref<uint32_t,
								 sycl::memory_order::relaxed,
								 sycl::memory_scope::device,
								 sycl::access::address_space::global_space>
					atomic_ref(*pair_count);
				uint32_t pair_idx = atomic_ref.fetch_add(1);
#else
				uint32_t pair_idx = (*pair_count)++;
#endif

				if (pair_idx < max_pairs) {
					neighbor_pairs[pair_idx] = int2(original_i, original_j);
				}
			}

			// Early termination based on Morton code distance
			// If Morton codes are too far apart, subsequent particles
			// in the sorted order are unlikely to be spatial neighbors
			morton_t code_j = sorted_morton_codes[j];

			// Only check early termination when moving forward in sorted order
			if (j > i && (code_j - code_i) > 0x1000000) { // Heuristic threshold
				break;
			}
		}
	}
};

} // namespace ARBD

#ifdef USE_SYCL
#include <sycl/sycl.hpp>
template<>
struct sycl::is_device_copyable<ARBD::ZOrderNeighborKernel> : std::true_type {};
#endif
