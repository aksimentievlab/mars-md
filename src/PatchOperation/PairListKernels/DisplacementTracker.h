#pragma once

#include "../ZOrderKernels/MortonCode.h"
#include "Header.h"
#include "PatchOperation/ZOrderKernels/AdaptiveKernels.h"
#include "Types/Types.h"
#include "Types/Vector3.h"

namespace ARBD {

// Use DisplacementKernel and MortonValidationKernel from AdaptiveKernels.h

/**
 * @brief Kernel for computing per-particle displacement and update mask
 *
 * This kernel computes displacement for each particle and creates a mask
 * indicating which particles have moved significantly, enabling selective
 * updates and optimizations.
 */
struct PerParticleDisplacementKernel {
	const Vector3* current_positions;
	const Vector3* previous_positions;
	float* displacements;  // Per-particle displacement values
	uint32_t* update_mask; // Bitmask for particles needing updates
	float displacement_threshold_sq;
	size_t num_particles;

	HOST DEVICE void operator()(idx_t i) const {
		if (i >= num_particles)
			return;

		Vector3 dr = current_positions[i] - previous_positions[i];
		float displacement_sq = dr.length2();

		displacements[i] = displacement_sq;

		// Set bit in update mask if displacement exceeds threshold
		if (displacement_sq > displacement_threshold_sq) {
			uint32_t word_idx = i / 32;
			uint32_t bit_idx = i % 32;
			uint32_t mask = 1u << bit_idx;

#ifdef __CUDA_ARCH__
			atomicOr(&update_mask[word_idx], mask);
#elif defined(__SYCL_DEVICE_ONLY__)
			auto atomic_or =
				sycl::atomic_ref<uint32_t,
								 sycl::memory_order::relaxed,
								 sycl::memory_scope::device,
								 sycl::access::address_space::global_space>(update_mask[word_idx]);
			atomic_or.fetch_or(mask);
#else
			update_mask[word_idx] |= mask;
#endif
		}
	}
};

} // namespace ARBD
