#pragma once
/*********************************************************************
 * @file  ZOrderNeighbor.h
 *
 * @brief Exact neighbor enumeration over Morton-sorted particles.
 *
 * BuildCellRangesKernel indexes the sorted array by coarse cell, and
 * ZOrderCellNeighborKernel walks the 27-cell stencil around each particle.
 *********************************************************************/

#include "../ZOrderKernels/MortonCode.h"
#include "Header.h"
#include "Types/Types.h"
#include "Types/Vector3.h"

namespace ARBD {

/**
 * @brief Build per-cell [begin,end) ranges over the Morton-sorted particle array.
 *
 * Top `3*m` bits of a Morton code index a coarse cell; particles in a cell form
 * a contiguous run of the sorted array, so a linear scan finds the boundaries.
 * Zero-fill `cell_begin`/`cell_end` before launch (empty cells -> empty range).
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

/// Sentinel for an unused / clipped slot in a cell's neighbor table.
inline constexpr uint32_t kInvalidCell = 0xFFFFFFFFu;

/**
 * @brief Precompute each coarse cell's up-to-27 neighbor cell indices.
 *
 * The stencil topology depends only on the grid (coarse_bits, per-axis
 * periodicity), not on particle positions, so it is built once per grid and
 * reused across rebuilds. Slots are padded to MAX_NEIGHBORS with kInvalidCell.
 * See dev_notes.md.
 */
struct BuildCellNeighborsKernel {
	uint32_t* cell_neighbors; ///< [num_cells * MAX_NEIGHBORS] output
	size_t num_cells;
	int coarse_bits; ///< m: coarse cells per dim = 2^m
	Vector3 box_len; ///< per-axis periodic length; <= 0 marks an open axis

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

	DEVICE void operator()(idx_t c) const {
		if (c >= num_cells)
			return;
		const uint32_t cell = static_cast<uint32_t>(c);
		const uint32_t cx = compact_by3(cell >> 2);
		const uint32_t cy = compact_by3(cell >> 1);
		const uint32_t cz = compact_by3(cell);

		const int n = 1 << coarse_bits;
		const uint32_t mask = static_cast<uint32_t>(n - 1);
		const bool per_x = box_len.x > 0.0f;
		const bool per_y = box_len.y > 0.0f;
		const bool per_z = box_len.z > 0.0f;

		// Periodic axes: narrow offsets when n <= 2 so no cell aliases. See dev_notes.md.
		const int p_lo = (n >= 3) ? -1 : 0;
		const int p_hi = (n >= 2) ? 1 : 0;
		const int x_lo = per_x ? p_lo : -1, x_hi = per_x ? p_hi : 1;
		const int y_lo = per_y ? p_lo : -1, y_hi = per_y ? p_hi : 1;
		const int z_lo = per_z ? p_lo : -1, z_hi = per_z ? p_hi : 1;

		uint32_t* out = cell_neighbors + static_cast<size_t>(cell) * MAX_NEIGHBORS;
		int k = 0;
		for (int dx = x_lo; dx <= x_hi; ++dx) {
			int nx = static_cast<int>(cx) + dx;
			if (per_x)
				nx = static_cast<int>((static_cast<uint32_t>(nx + n)) & mask);
			else if (nx < 0 || nx >= n)
				continue;
			const uint32_t mx = split_by3(static_cast<uint32_t>(nx)) << 2;
			for (int dy = y_lo; dy <= y_hi; ++dy) {
				int ny = static_cast<int>(cy) + dy;
				if (per_y)
					ny = static_cast<int>((static_cast<uint32_t>(ny + n)) & mask);
				else if (ny < 0 || ny >= n)
					continue;
				const uint32_t mxy = mx | (split_by3(static_cast<uint32_t>(ny)) << 1);
				for (int dz = z_lo; dz <= z_hi; ++dz) {
					int nz = static_cast<int>(cz) + dz;
					if (per_z)
						nz = static_cast<int>((static_cast<uint32_t>(nz + n)) & mask);
					else if (nz < 0 || nz >= n)
						continue;
					out[k++] = mxy | split_by3(static_cast<uint32_t>(nz));
				}
			}
		}
		for (; k < MAX_NEIGHBORS; ++k)
			out[k] = kInvalidCell;
	}
};

/**
 * @brief Exact Z-order neighbor finding via a 27-cell stencil.
 *
 * Particles stay Morton-sorted (force-kernel locality); neighbors come from the
 * 27 coarse cells around each particle. Cell side >= pairlist cutoff so the
 * stencil covers the cutoff sphere. Periodicity is per axis via `box_len`
 * (positive wraps with minimum image, zero is open). See dev_notes.md.
 */
struct ZOrderCellNeighborKernel {
	const Vector3* sorted_positions;
	const morton_t* sorted_morton_codes;
	const uint32_t* sorted_to_original;
	const uint32_t* cell_begin;
	const uint32_t* cell_end;
	const uint32_t* cell_neighbors; ///< [num_cells * MAX_NEIGHBORS] from BuildCellNeighborsKernel
	int2* neighbor_pairs;
	uint32_t* pair_count;
	float cutoff_squared;
	size_t num_particles;
	size_t max_pairs;
	int shift;		 ///< 3 * (bits_per_dim - m); recovers a cell index from a Morton code
	Vector3 box_len; ///< per-axis periodic length; <= 0 marks an open axis

	DEVICE inline float min_image(float d, float L) const {
		if (L <= 0.0f)
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
		const uint32_t sorted_i = static_cast<uint32_t>(i);
		const uint32_t cell = static_cast<uint32_t>(sorted_morton_codes[i] >> shift);
		const uint32_t* nbrs = cell_neighbors + static_cast<size_t>(cell) * MAX_NEIGHBORS;

		for (int k = 0; k < MAX_NEIGHBORS; ++k) {
			const uint32_t ncell = nbrs[k];
			if (ncell == kInvalidCell)
				continue;

			const uint32_t begin = cell_begin[ncell];
			const uint32_t end = cell_end[ncell];

			// Start at sorted_i+1 so each pair is emitted once. See dev_notes.md.
			const uint32_t j_lo = (begin > sorted_i + 1u) ? begin : sorted_i + 1u;

			for (uint32_t j = j_lo; j < end; ++j) {
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
					// Ordering them keeps the x < y invariant the sorted-index key drops.
					if (pair_idx < max_pairs) {
						const uint32_t a = sorted_to_original[i];
						const uint32_t b = sorted_to_original[j];
						neighbor_pairs[pair_idx] = int2(static_cast<int>(a < b ? a : b),
														static_cast<int>(a < b ? b : a));
					}
				}
			}
		}
	}
};

#ifdef USE_CUDA
extern template Event launch_cuda_kernel(const Resource& resource,
										 const KernelConfig& config,
										 BuildCellRangesKernel kernel_func);
extern template Event launch_cuda_kernel(const Resource& resource,
										 const KernelConfig& config,
										 BuildCellNeighborsKernel kernel_func);
extern template Event launch_cuda_kernel(const Resource& resource,
										 const KernelConfig& config,
										 ZOrderCellNeighborKernel kernel_func);
#endif

} // namespace ARBD

#ifdef USE_SYCL
#include <sycl/sycl.hpp>
template<>
struct sycl::is_device_copyable<ARBD::ZOrderCellNeighborKernel> : std::true_type {};
template<>
struct sycl::is_device_copyable<ARBD::BuildCellRangesKernel> : std::true_type {};
template<>
struct sycl::is_device_copyable<ARBD::BuildCellNeighborsKernel> : std::true_type {};
#endif
