#pragma once
/*********************************************************************
 * @file  ZOrderNeighbor.h
 *
 * @brief Exact neighbor enumeration over Morton-sorted particles.
 *
 * BuildCellRangesKernel indexes the sorted array by coarse cell, and
 * ZOrderCellNeighborKernel walks the per-axis stencil around each particle.
 *********************************************************************/

#include "../ZOrderKernels/MortonCode.h"
#include "Header.h"
#include "Interactions/DeviceExclusions.h"
#include "System/PeriodicBox.h"
#include "Types/Types.h"
#include "Types/Vector3.h"

namespace MARS {

/**
 * @brief Build per-cell [begin,end) ranges over the Morton-sorted particle array.
 *
 * Top `3*m` bits of a Morton code index a coarse cell; particles in a cell form
 * a contiguous run of the sorted array, so a linear scan finds the boundaries.
 * Zero-fill `cell_begin`/`cell_end` before launch (empty cells -> empty range).
 */
struct BuildCellRangesKernel {
	const morton_t* __restrict__ sorted_morton_codes;
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
 * @brief Precompute each coarse cell's neighbor cell indices.
 *
 * Morton splits every axis into the same 2^m cells, so on an anisotropic box the
 * cells are anisotropic too and a fixed +-1 stencil would force the coarsest axis
 * on all three. Each axis therefore carries its own radius. The topology depends
 * only on the grid (coarse_bits, radii, per-axis periodicity), not on particle
 * positions, so it is built once per grid and reused across rebuilds. Rows are
 * padded to neighbors_per_cell with kInvalidCell. See dev_notes.md.
 */
struct BuildCellNeighborsKernel {
	uint32_t* cell_neighbors; ///< [num_cells * neighbors_per_cell] output
	size_t num_cells;
	int coarse_bits; ///< m: coarse cells per dim = 2^m
	int3 z_order_raddi;
	int neighbors_per_cell;
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

	DEVICE static inline void axis_range(bool periodic, int r, int n, int& lo, int& hi) {
		if (periodic && 2 * r + 1 >= n) {
			lo = 0;
			hi = n - 1;
		} else {
			lo = -r;
			hi = r;
		}
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

		int x_lo, x_hi, y_lo, y_hi, z_lo, z_hi;
		axis_range(per_x, z_order_raddi.x, n, x_lo, x_hi);
		axis_range(per_y, z_order_raddi.y, n, y_lo, y_hi);
		axis_range(per_z, z_order_raddi.z, n, z_lo, z_hi);

		uint32_t* out = cell_neighbors + static_cast<size_t>(cell) * neighbors_per_cell;
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
		for (; k < neighbors_per_cell; ++k)
			out[k] = kInvalidCell;
	}
};

/**
 * @brief Exact Z-order neighbor finding over a per-axis cell stencil.
 *
 * Particles stay Morton-sorted (force-kernel locality); neighbors come from the
 * cells BuildCellNeighborsKernel listed for each particle's own cell, whose radii
 * cover the cutoff sphere on every axis independently. Periodicity is per axis
 * via `box_len` (positive wraps with minimum image, zero is open). Excluded pairs
 * are dropped here rather than downstream. See dev_notes.md.
 */
struct ZOrderCellNeighborKernel {
	const Vector3* __restrict__ sorted_positions;
	const morton_t* __restrict__ sorted_morton_codes;
	const uint32_t* __restrict__ sorted_to_original;
	const uint32_t* __restrict__ cell_begin;
	const uint32_t* __restrict__ cell_end;
	const uint32_t* __restrict__ cell_neighbors; ///< [num_cells * neighbors_per_cell] from
												 ///< BuildCellNeighborsKernel
	int2* neighbor_pairs;
	uint32_t* pair_count;
	float cutoff_squared;
	size_t num_particles;
	size_t max_pairs;
	int neighbors_per_cell;
	int shift;				  ///< 3 * (bits_per_dim - m); recovers a cell index from a Morton code
	PeriodicBox box;		  ///< minimum-image periodic box; open axes left unwrapped
	ExclusionView exclusions; ///< excluded pairs are dropped before emission

	DEVICE void operator()(idx_t i) const {
		if (i >= num_particles)
			return;

		const Vector3 pos_i = sorted_positions[i];
		const uint32_t sorted_i = static_cast<uint32_t>(i);
		const uint32_t cell = static_cast<uint32_t>(sorted_morton_codes[i] >> shift);
		const uint32_t* nbrs = cell_neighbors + static_cast<size_t>(cell) * neighbors_per_cell;

		// Both endpoint's original index and its exclusion row depend only on i,
		// so they are loaded once rather than per candidate. See dev_notes.md.
		const int a = static_cast<int>(sorted_to_original[i]);
		const int excl_begin = exclusions.row_begin(a);
		const int excl_end = exclusions.row_end(a);
		const int excl_body = exclusions.body_of(a);

		for (int k = 0; k < neighbors_per_cell; ++k) {
			const uint32_t ncell = nbrs[k];
			if (ncell == kInvalidCell)
				continue;

			const uint32_t begin = cell_begin[ncell];
			const uint32_t end = cell_end[ncell];

			// Start at sorted_i+1 so each pair is emitted once. See dev_notes.md.
			const uint32_t j_lo = (begin > sorted_i + 1u) ? begin : sorted_i + 1u;

			for (uint32_t j = j_lo; j < end; ++j) {
				const Vector3 pos_j = sorted_positions[j];
				const Vector3 dr = box.wrap_diff(pos_j - pos_i);
				const float d2 = dr.length2();

				if (d2 <= cutoff_squared) {
					const int b = static_cast<int>(sorted_to_original[j]);
					if (exclusions.same_body(excl_body, b) ||
						exclusions.row_contains(excl_begin, excl_end, b))
						continue;
					// Per-hit atomic on purpose: the interleaved slot order it
					// produces is load-bearing for the force kernel. See dev_notes.md.
					const uint32_t pair_idx = ATOMIC_ADD(pair_count, 1U);
					// Ordering them keeps the x < y invariant the sorted-index key drops.
					if (pair_idx < max_pairs) {
						neighbor_pairs[pair_idx] = int2(a < b ? a : b, a < b ? b : a);
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

} // namespace MARS

#ifdef USE_SYCL
#include <sycl/sycl.hpp>
template<>
struct sycl::is_device_copyable<MARS::ZOrderCellNeighborKernel> : std::true_type {};
template<>
struct sycl::is_device_copyable<MARS::BuildCellRangesKernel> : std::true_type {};
template<>
struct sycl::is_device_copyable<MARS::BuildCellNeighborsKernel> : std::true_type {};
#endif
