#pragma once
/**
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

		// WARNING: this fixed-window search is NOT exact and systematically drops
		// neighbors. A Morton curve places most spatial neighbors close in sorted
		// order, but the curve has discontinuities at every octree boundary, so a
		// bounded window can never enumerate the full neighbor set. Measured
		// against a KD-tree ground truth on a 305k-particle cytoplasm at the
		// production pairlist cutoff, this window captured only ~29% of true
		// neighbors, which silently removed ~71% of nonbonded interactions and
		// left the system unable to condense.
		//
		// Retained only for reference/benchmarking. Production builds must use
		// ZOrderCellNeighborKernel below, which is exact.
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
				uint32_t pair_idx = ATOMIC_ADD(pair_count, 1U);
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
		}
	}
};

#ifdef USE_CUDA
extern template Event launch_cuda_kernel(const Resource& resource,
										 const KernelConfig& config,
										 ZOrderNeighborKernel kernel_func);
#endif


/**
 * @brief Build per-cell [begin,end) ranges over the Morton-sorted particle array.
 *
 * Exploits the hierarchical property of Morton codes: taking the top `3*m` bits
 * of a code yields the index of a coarse cell whose side is `1/2^m` of the
 * encoding box, and -- because the interleave preserves that prefix ordering --
 * every particle in such a cell occupies a *contiguous* run of the sorted array.
 * Detecting the run boundaries therefore requires only a linear scan.
 *
 * `cell_begin`/`cell_end` must be zero-filled before launch so that cells with
 * no particles yield an empty range.
 */
struct BuildCellRangesKernel {
	const morton_t* sorted_morton_codes;
	uint32_t* cell_begin;
	uint32_t* cell_end;
	size_t num_particles;
	int shift; ///< 3 * (bits_per_dim - coarse_bits)

	DEVICE void operator()(idx_t i) const {
		if (i >= num_particles)
			return;
		const morton_t c = sorted_morton_codes[i] >> shift;
		if (i == 0 || (sorted_morton_codes[i - 1] >> shift) != c)
			cell_begin[c] = static_cast<uint32_t>(i);
		if (i + 1 == num_particles || (sorted_morton_codes[i + 1] >> shift) != c)
			cell_end[c] = static_cast<uint32_t>(i + 1);
	}
};

/**
 * @brief Exact Z-order neighbor finding via a 27-cell stencil.
 *
 * Replaces the fixed-window heuristic in ZOrderNeighborKernel. Particles remain
 * Morton-sorted -- that is what gives the force kernel its memory locality --
 * but neighbors are enumerated by visiting the 27 coarse cells surrounding each
 * particle, exactly as a conventional cell list does. The coarse cell side is
 * chosen on the host to be at least the pairlist cutoff, so the 27-cell stencil
 * provably covers the cutoff sphere and no interacting pair can be missed.
 *
 * When `periodic` is set, cell indices wrap and displacements use the minimum
 * image convention, so pairs spanning a periodic boundary are found as well.
 */
struct ZOrderCellNeighborKernel {
	const Vector3* sorted_positions;
	const morton_t* sorted_morton_codes;
	const uint32_t* sorted_to_original;
	const uint32_t* cell_begin;
	const uint32_t* cell_end;
	int2* neighbor_pairs;
	uint32_t* pair_count;
	float cutoff_squared;
	size_t num_particles;
	size_t max_pairs;
	int coarse_bits;  ///< m: coarse cells per dim = 2^m
	int shift;        ///< 3 * (bits_per_dim - m)
	Vector3 box_len;  ///< periodic box lengths; ignored unless `periodic`
	bool periodic;

	DEVICE static inline uint32_t compact_by3(uint32_t x) {
		x &= 0x09249249u;
		x = (x ^ (x >> 2)) & 0x030c30c3u;
		x = (x ^ (x >> 4)) & 0x0300f00fu;
		x = (x ^ (x >> 8)) & 0x030000ffu;
		x = (x ^ (x >> 16)) & 0x000003ffu;
		return x;
	}

	DEVICE static inline uint32_t split_by3(uint32_t a) {
		uint32_t x = a & 0x000003ffu;
		x = (x | (x << 16)) & 0x030000ffu;
		x = (x | (x << 8)) & 0x0300f00fu;
		x = (x | (x << 4)) & 0x030c30c3u;
		x = (x | (x << 2)) & 0x09249249u;
		return x;
	}

	DEVICE inline float min_image(float d, float L) const {
		if (!periodic || L <= 0.0f)
			return d;
		while (d > 0.5f * L)
			d -= L;
		while (d < -0.5f * L)
			d += L;
		return d;
	}

	DEVICE void operator()(idx_t i) const {
		if (i >= num_particles)
			return;

		const Vector3 pos_i = sorted_positions[i];
		const uint32_t original_i = sorted_to_original[i];

		// Decode this particle's coarse cell from its Morton prefix. The encode
		// order is z | y<<1 | x<<2 (see MortonCode::encode).
		const morton_t prefix = sorted_morton_codes[i] >> shift;
		const uint32_t cx = compact_by3(prefix >> 2);
		const uint32_t cy = compact_by3(prefix >> 1);
		const uint32_t cz = compact_by3(prefix);

		const int n = 1 << coarse_bits;
		const uint32_t mask = static_cast<uint32_t>(n - 1);

		for (int dx = -1; dx <= 1; ++dx) {
			int nx = static_cast<int>(cx) + dx;
			if (periodic)
				nx = static_cast<int>((static_cast<uint32_t>(nx + n)) & mask);
			else if (nx < 0 || nx >= n)
				continue;

			for (int dy = -1; dy <= 1; ++dy) {
				int ny = static_cast<int>(cy) + dy;
				if (periodic)
					ny = static_cast<int>((static_cast<uint32_t>(ny + n)) & mask);
				else if (ny < 0 || ny >= n)
					continue;

				for (int dz = -1; dz <= 1; ++dz) {
					int nz = static_cast<int>(cz) + dz;
					if (periodic)
						nz = static_cast<int>((static_cast<uint32_t>(nz + n)) & mask);
					else if (nz < 0 || nz >= n)
						continue;

					const uint32_t ncell = split_by3(static_cast<uint32_t>(nz)) |
										   (split_by3(static_cast<uint32_t>(ny)) << 1) |
										   (split_by3(static_cast<uint32_t>(nx)) << 2);

					const uint32_t begin = cell_begin[ncell];
					const uint32_t end = cell_end[ncell];

					for (uint32_t j = begin; j < end; ++j) {
						const uint32_t original_j = sorted_to_original[j];
						// Store each pair once, keyed on original indices.
						if (original_i >= original_j)
							continue;

						const Vector3 pos_j = sorted_positions[j];
						const float ddx = min_image(pos_j.x - pos_i.x, box_len.x);
						const float ddy = min_image(pos_j.y - pos_i.y, box_len.y);
						const float ddz = min_image(pos_j.z - pos_i.z, box_len.z);
						const float d2 = ddx * ddx + ddy * ddy + ddz * ddz;

						if (d2 <= cutoff_squared) {
#ifdef USE_CUDA
							uint32_t pair_idx = ATOMIC_ADD(pair_count, 1U);
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
								neighbor_pairs[pair_idx] =
									int2(static_cast<int>(original_i), static_cast<int>(original_j));
							}
						}
					}
				}
			}
		}
	}
};

#ifdef USE_CUDA
// Both kernels are launched from ZOrderPairlist.cpp, a host-only translation
// unit. Without these declarations the compiler instantiates the non-CUDA stub
// in KernelHelper.cuh and every pairlist build throws NotImplementedError; the
// matching definitions live in ZOrderPairlist.cu.
extern template Event launch_cuda_kernel(const Resource& resource,
										 const KernelConfig& config,
										 BuildCellRangesKernel kernel_func);
extern template Event launch_cuda_kernel(const Resource& resource,
										 const KernelConfig& config,
										 ZOrderCellNeighborKernel kernel_func);
#endif

} // namespace ARBD

#ifdef USE_SYCL
#include <sycl/sycl.hpp>
template<>
struct sycl::is_device_copyable<ARBD::ZOrderNeighborKernel> : std::true_type {};
#endif

#ifdef USE_SYCL
template<>
struct sycl::is_device_copyable<ARBD::ZOrderCellNeighborKernel> : std::true_type {};
template<>
struct sycl::is_device_copyable<ARBD::BuildCellRangesKernel> : std::true_type {};
#endif

