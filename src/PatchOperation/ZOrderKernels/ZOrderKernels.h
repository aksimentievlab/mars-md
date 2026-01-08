#pragma once

/*********************************************************************
 * @file  ZOrderKernels.h
 *
 * @brief GPU kernel functors for Z-order operations
 *
 * This file contains KERNEL_FUNC-side kernel functors for:
 * - Morton code encoding from particle positions
 * - Data reordering based on sorted indices
 * - Neighbor finding using Z-order locality
 *********************************************************************/

#include "Backend/Kernels.h"
#include "MortonCode.h"
#include "Types/Types.h"
#include "Types/Vector3.h"

// Include adaptive kernels for Pairlist mode optimizations
#include "AdaptiveKernels.h"

namespace ARBD {

/**
 * @brief Kernel to encode particle positions into Morton codes
 */
struct MortonEncodeKernel {
	const Vector3* positions;
	morton_t* morton_codes;
	uint32_t* original_indices;
	Vector3 box_min;
	Vector3 box_max;
	size_t num_particles;

	KERNEL_FUNC void operator()(idx_t idx) const {
		if (idx >= num_particles)
			return;

		// Encode position to Morton code
		morton_codes[idx] = MortonCode::encode(positions[idx], box_min, box_max);

		// Store original particle index
		original_indices[idx] = static_cast<uint32_t>(idx);
	}
};

/**
 * @brief Kernel to reorder any data array based on sorted indices
 */
template<typename T>
struct ReorderDataKernel {
	const T* input_data;
	T* output_data;
	const uint32_t* sorted_indices;
	size_t num_elements;

	KERNEL_FUNC void operator()(idx_t idx) const {
		if (idx >= num_elements)
			return;

		output_data[idx] = input_data[sorted_indices[idx]];
	}
};

/**
 * @brief Kernel to create inverse mapping from sorted order back to original
 */
struct InverseIndexKernel {
	const uint32_t* sorted_indices;
	uint32_t* inverse_indices;
	size_t num_elements;

	KERNEL_FUNC void operator()(idx_t idx) const {
		if (idx >= num_elements)
			return;

		// Create inverse mapping: inverse_indices[sorted_indices[i]] = i
		uint32_t original_idx = sorted_indices[idx];
		if (original_idx < num_elements) {
			inverse_indices[original_idx] = static_cast<uint32_t>(idx);
		}
	}
};

/**
 * @brief Kernel to validate Z-order sorting (for debugging)
 */
struct ValidateZOrderKernel {
	const morton_t* morton_codes;
	uint32_t* error_count;
	size_t num_particles;

	KERNEL_FUNC void operator()(idx_t idx) const {
		if (idx >= num_particles - 1)
			return;

		// Check if Morton codes are properly sorted
		if (morton_codes[idx] > morton_codes[idx + 1]) {
#ifdef __CUDA_ARCH__
			atomicAdd(error_count, 1);
#else
			(*error_count)++;
#endif
		}
	}
};

/**
 * @brief Kernel to compute bounding box of particles
 */
struct BoundingBoxKernel {
	const Vector3* positions;
	Vector3* min_bounds;
	Vector3* max_bounds;
	size_t num_particles;

	KERNEL_FUNC void operator()(idx_t idx) const {
		if (idx >= num_particles)
			return;

		Vector3 pos = positions[idx];

		// Use atomic operations to find global min/max
		// Note: This is a simplified version - production code might use reduction
#ifdef __CUDA_ARCH__
		atomicMin(&min_bounds->x, pos.x);
		atomicMin(&min_bounds->y, pos.y);
		atomicMin(&min_bounds->z, pos.z);
		atomicMax(&max_bounds->x, pos.x);
		atomicMax(&max_bounds->y, pos.y);
		atomicMax(&max_bounds->z, pos.z);
#elif defined(__SYCL_KERNEL_FUNC_ONLY__)
		// Create separate atomic refs for each component
		sycl::atomic_ref<float,
						 sycl::memory_order::relaxed,
						 sycl::memory_scope::KERNEL_FUNC,
						 sycl::access::address_space::global_space>
			atomic_min_x(min_bounds->x);
		sycl::atomic_ref<float,
						 sycl::memory_order::relaxed,
						 sycl::memory_scope::KERNEL_FUNC,
						 sycl::access::address_space::global_space>
			atomic_min_y(min_bounds->y);
		sycl::atomic_ref<float,
						 sycl::memory_order::relaxed,
						 sycl::memory_scope::KERNEL_FUNC,
						 sycl::access::address_space::global_space>
			atomic_min_z(min_bounds->z);

		atomic_min_x.fetch_min(pos.x);
		atomic_min_y.fetch_min(pos.y);
		atomic_min_z.fetch_min(pos.z);

		sycl::atomic_ref<float,
						 sycl::memory_order::relaxed,
						 sycl::memory_scope::KERNEL_FUNC,
						 sycl::access::address_space::global_space>
			atomic_max_x(max_bounds->x);
		sycl::atomic_ref<float,
						 sycl::memory_order::relaxed,
						 sycl::memory_scope::KERNEL_FUNC,
						 sycl::access::address_space::global_space>
			atomic_max_y(max_bounds->y);
		sycl::atomic_ref<float,
						 sycl::memory_order::relaxed,
						 sycl::memory_scope::KERNEL_FUNC,
						 sycl::access::address_space::global_space>
			atomic_max_z(max_bounds->z);

		atomic_max_x.fetch_max(pos.x);
		atomic_max_y.fetch_max(pos.y);
		atomic_max_z.fetch_max(pos.z);
#else
		min_bounds->x = fmin(min_bounds->x, pos.x);
		min_bounds->y = fmin(min_bounds->y, pos.y);
		min_bounds->z = fmin(min_bounds->z, pos.z);
		max_bounds->x = fmax(max_bounds->x, pos.x);
		max_bounds->y = fmax(max_bounds->y, pos.y);
		max_bounds->z = fmax(max_bounds->z, pos.z);
#endif
	}
};

} // namespace ARBD

// SYCL device copyability declarations
#ifdef USE_SYCL
template<>
struct sycl::is_device_copyable<ARBD::MortonEncodeKernel> : std::true_type {};

template<typename T>
struct sycl::is_device_copyable<ARBD::ReorderDataKernel<T>> : std::true_type {};

template<>
struct sycl::is_device_copyable<ARBD::InverseIndexKernel> : std::true_type {};

template<>
struct sycl::is_device_copyable<ARBD::ValidateZOrderKernel> : std::true_type {};

template<>
struct sycl::is_device_copyable<ARBD::BoundingBoxKernel> : std::true_type {};

template<>
struct sycl::is_device_copyable<ARBD::DisplacementKernel> : std::true_type {};

template<>
struct sycl::is_device_copyable<ARBD::MortonValidationKernel> : std::true_type {};
#endif
