#include "../MortonCode.h"
#include "Header.h"
#include "Types/Types.h"
#include "Types/Vector3.h"
namespace ARBD {
/**
 * @brief Kernel for Z-order based neighbor finding
 * Uses the spatial locality of Morton codes to efficiently find neighbors
 */
struct ZOrderNeighborKernel {
	const Vector3* sorted_positions;
	const morton_t* sorted_morton_codes;
	const uint32_t* sorted_to_original;
	int2* neighbor_pairs;
	uint32_t* pair_count;
	float cutoff_squared;
	size_t num_particles;
	size_t max_pairs;

	DEVICE void operator()(idx_t i) const {
		if (i >= num_particles)
			return;

		Vector3 pos_i = sorted_positions[i];
		morton_t code_i = sorted_morton_codes[i];

		// Search backward and forward in Morton order for neighbors
		// Due to Z-order properties, nearby particles in 3D space
		// are likely to be close in Morton order

		// Search range - could be tuned based on system characteristics
		const size_t search_range = 64; // Search ±64 positions in sorted order

		size_t start = (i >= search_range) ? i - search_range : 0;
		size_t end = (i + search_range < num_particles) ? i + search_range : num_particles;

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

#ifdef __CUDA_ARCH__
				uint32_t pair_idx = atomicAdd(pair_count, 1);
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
			if (j > i && (code_j - code_i) > 0x1000000) { // Heuristic threshold
				break;
			}
		}
	}
};
} // namespace ARBD
