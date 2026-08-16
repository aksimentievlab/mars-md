#pragma once

#include "Header.h"
#include "MortonCode.h"
#include "Types/Types.h"
#include "Types/Vector3.h"

#ifdef __CUDA_ARCH__
#include <cuda/atomic>
#endif

#ifdef __SYCL_DEVICE_ONLY__
#include <sycl/sycl.hpp>
#endif

namespace ARBD {

#ifdef __CUDA_ARCH__
// Helper function for atomic max with float
__device__ inline void atomicMaxFloat(float* address, float val) {
	int* address_as_int = (int*)address;
	int old = *address_as_int, assumed;
	do {
		assumed = old;
		old =
			atomicCAS(address_as_int, assumed, __float_as_int(fmaxf(val, __int_as_float(assumed))));
	} while (assumed != old);
}

// Helper function for atomic min with float
__device__ inline void atomicMinFloat(float* address, float val) {
	int* address_as_int = (int*)address;
	int old = *address_as_int, assumed;
	do {
		assumed = old;
		old =
			atomicCAS(address_as_int, assumed, __float_as_int(fminf(val, __int_as_float(assumed))));
	} while (assumed != old);
}
#endif

/**
 * @brief Kernel for computing maximum particle displacement on device
 *
 * This kernel computes the maximum displacement of particles between two
 * position sets, enabling intelligent decision making for pairlist updates
 * without requiring host-device synchronization.
 *
 * Displacements are measured under the minimum image convention. Integrators
 * wrap positions back into the box every step, so a particle crossing a
 * periodic face differs from its reference by nearly a full box length while
 * having physically moved almost nothing. Taking that raw difference makes
 * every boundary crossing look like a box-sized jump, which trips any skin
 * threshold and forces a spurious pairlist rebuild.
 *
 * `box_len` is per axis: a positive component marks that axis periodic and its
 * displacements are wrapped; a zero or negative component marks it open and the
 * raw difference is used.
 */
struct DisplacementKernel {
	const Vector3* current_positions;
	const Vector3* previous_positions;
	float* max_displacement; // Single value reduction result
	size_t num_particles;
	Vector3 box_len; ///< Per-axis periodic length; <= 0 marks an open axis

	/// Wrap a displacement into [-L/2, L/2]. Both inputs are already wrapped
	/// into the box, so |d| < L and at most one iteration runs.
	KERNEL_FUNC static inline float min_image(float d, float L) {
		if (L <= 0.0f)
			return d;
		while (d > 0.5f * L)
			d -= L;
		while (d < -0.5f * L)
			d += L;
		return d;
	}

	KERNEL_FUNC void operator()(idx_t i) const {
		if (i >= num_particles)
			return;

		Vector3 raw = current_positions[i] - previous_positions[i];
		Vector3 dr{min_image(raw.x, box_len.x),
				   min_image(raw.y, box_len.y),
				   min_image(raw.z, box_len.z)};
		float displacement_sq = dr.length2();

#ifdef __CUDA_ARCH__
		// CUDA: Use atomicMaxFloat for reduction
		atomicMaxFloat(max_displacement, displacement_sq);
#elif defined(__SYCL_DEVICE_ONLY__)
		// SYCL: Use atomic_ref for reduction
		sycl::atomic_ref<float,
						 sycl::memory_order::relaxed,
						 sycl::memory_scope::device,
						 sycl::access::address_space::global_space>
			atomic_max(*max_displacement);

		// Manual atomic max implementation since SYCL doesn't have atomic_max for floats
		float expected = *max_displacement;
		while (displacement_sq > expected) {
			if (atomic_max.compare_exchange_weak(expected, displacement_sq)) {
				break;
			}
		}
#else
		// CPU: Simple comparison
		if (displacement_sq > *max_displacement) {
			*max_displacement = displacement_sq;
		}
#endif
	}
};

/**
 * @brief Kernel for validating Morton code ordering
 *
 * Checks if the current Morton code ordering is still valid given
 * particle movements, enabling smart reuse of sorted data.
 */
struct MortonValidationKernel {
	const Vector3* current_positions;
	const morton_t* morton_codes;
	const uint32_t* sorted_indices;
	Vector3 box_min;
	Vector3 box_max;
	uint32_t* invalid_count;
	size_t num_particles;

	KERNEL_FUNC void operator()(idx_t i) const {
		if (i >= num_particles)
			return;

		// Get current position of particle at sorted index i
		uint32_t particle_id = sorted_indices[i];
		Vector3 pos = current_positions[particle_id];

		// Compute current Morton code for this position
		morton_t current_morton = MortonCode::encode(pos, box_min, box_max);
		morton_t stored_morton = morton_codes[i];

		// Check if Morton code has changed significantly
		// Allow small variations due to floating point precision
		morton_t morton_diff = (current_morton > stored_morton) ? (current_morton - stored_morton)
																: (stored_morton - current_morton);

		// If Morton code changed significantly, mark as invalid
		if (morton_diff > 0x100) { // Threshold for significant change
#ifdef __CUDA_ARCH__
			atomicAdd(invalid_count, 1);
#elif defined(__SYCL_DEVICE_ONLY__)
			sycl::atomic_ref<uint32_t,
							 sycl::memory_order::relaxed,
							 sycl::memory_scope::device,
							 sycl::access::address_space::global_space>
				atomic_add(*invalid_count);
			atomic_add.fetch_add(1);
#else
			(*invalid_count)++;
#endif
		}
	}
};

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

	KERNEL_FUNC void operator()(idx_t i) const {
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
 * @brief Z-order neighbor kernel with adaptive search ranges and optimizations
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

	KERNEL_FUNC void operator()(idx_t i) const {
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
	KERNEL_FUNC void flush_local_pairs(const int2* local_buffer,
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

	KERNEL_FUNC morton_t get_morton_threshold(uint32_t search_range) const {
		// Adaptive threshold based on search range
		// Larger search ranges allow larger Morton code differences
		return 0x100000 + (search_range << 12);
	}
};

} // namespace ARBD
