#pragma once
/*********************************************************************
 * @file  ZOrderNeighbor.h
 *
 * @brief Exact neighbor enumeration over Morton-sorted particles.
 *
 * BuildCellRangesKernel indexes the sorted array by coarse cell, and
 * ZOrderCellNeighborKernel walks the 27-cell stencil around each particle.
 *
 * A fixed-window search over the sorted order used to live here as well. It is
 * gone, not disabled: a Morton curve is discontinuous at every octree boundary,
 * so no bounded window can enumerate the full neighbor set, and no choice of
 * window size fixes that. Measured against a KD-tree ground truth on a
 * 305k-particle cytoplasm at the production pairlist cutoff, a +/-64 window
 * found ~29% of true neighbors, silently dropping ~71% of nonbonded
 * interactions and leaving the system unable to condense.
 *********************************************************************/

#include "../ZOrderKernels/MortonCode.h"
#include "Header.h"
#include "Types/Types.h"
#include "Types/Vector3.h"

namespace ARBD {

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
 * Particles remain
 * Morton-sorted -- that is what gives the force kernel its memory locality --
 * but neighbors are enumerated by visiting the 27 coarse cells surrounding each
 * particle, exactly as a conventional cell list does. The coarse cell side is
 * chosen on the host to be at least the pairlist cutoff, so the 27-cell stencil
 * provably covers the cutoff sphere and no interacting pair can be missed.
 *
 * Periodicity is per axis and is carried entirely by `box_len`: a positive
 * component means that axis is periodic, so its cell indices wrap and its
 * displacements use the minimum image convention; a zero component means the
 * axis is open and the stencil is simply clipped there. Mixed boundary
 * conditions therefore work, instead of degrading the whole search to open.
 *
 * The Morton encoding box must equal the simulation box on every periodic
 * axis -- see ZOrderPairlist::build_pairlist, which enforces that. Wrapping a
 * cell index modulo the grid asserts that cell n-1 is physically adjacent to
 * cell 0, which only holds when the encoded extent is the periodic extent.
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
	int coarse_bits; ///< m: coarse cells per dim = 2^m
	int shift;		 ///< 3 * (bits_per_dim - m)
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

		// Decode this particle's coarse cell from its Morton prefix. The encode
		// order is z | y<<1 | x<<2 (see MortonCode::encode).
		const morton_t prefix = sorted_morton_codes[i] >> shift;
		const uint32_t cx = compact_by3(prefix >> 2);
		const uint32_t cy = compact_by3(prefix >> 1);
		const uint32_t cz = compact_by3(prefix);

		const int n = 1 << coarse_bits;
		const uint32_t mask = static_cast<uint32_t>(n - 1);

		const bool per_x = box_len.x > 0.0f;
		const bool per_y = box_len.y > 0.0f;
		const bool per_z = box_len.z > 0.0f;

		// Offset range for a *periodic* axis. With fewer than three cells along
		// it the wrapped offsets -1/0/+1 alias onto the same cell -- all three
		// when n == 1, and -1 with +1 when n == 2 -- so the naive -1..1 loop
		// visits that cell repeatedly and emits each pair in it up to 27 times,
		// multiplying its force by the same factor. Narrowing the range keeps
		// every distinct cell visited exactly once, and it stays complete: when
		// n <= 2 the stencil covers the entire grid either way.
		//
		// Open axes are unaffected -- out-of-range indices are skipped rather
		// than wrapped, so they cannot alias -- and must keep the full range to
		// reach the cell below.
		const int p_lo = (n >= 3) ? -1 : 0;
		const int p_hi = (n >= 2) ? 1 : 0;
		const int x_lo = per_x ? p_lo : -1, x_hi = per_x ? p_hi : 1;
		const int y_lo = per_y ? p_lo : -1, y_hi = per_y ? p_hi : 1;
		const int z_lo = per_z ? p_lo : -1, z_hi = per_z ? p_hi : 1;

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

					const uint32_t ncell = mxy | split_by3(static_cast<uint32_t>(nz));

					const uint32_t begin = cell_begin[ncell];
					const uint32_t end = cell_end[ncell];

					// Emit each pair once, keyed on the *sorted* index. Because
					// a cell occupies a contiguous run of the sorted array, a
					// whole cell lying before this particle collapses to an
					// empty loop, and the particle's own cell is entered at
					// i+1 -- so roughly half the stencil is skipped outright
					// rather than enumerated and rejected pairwise. The stencil
					// relation is symmetric under wrapping, so a pair dropped
					// here is always emitted by the other particle's thread.
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
							if (pair_idx < max_pairs) {
								// Original indices are only needed for pairs
								// that survive; ordering them keeps the
								// x < y invariant the sorted-index key drops.
								const uint32_t a = sorted_to_original[i];
								const uint32_t b = sorted_to_original[j];
								neighbor_pairs[pair_idx] = int2(static_cast<int>(a < b ? a : b),
																static_cast<int>(a < b ? b : a));
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
struct sycl::is_device_copyable<ARBD::ZOrderCellNeighborKernel> : std::true_type {};
template<>
struct sycl::is_device_copyable<ARBD::BuildCellRangesKernel> : std::true_type {};
#endif
