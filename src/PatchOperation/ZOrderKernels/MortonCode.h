#pragma once

/*********************************************************************
 * @file  MortonCode.h
 *
 * @brief Morton (Z-order) code utilities for 3D spatial indexing
 *
 * This file contains utilities for encoding 3D coordinates into
 * Morton codes for spatial sorting and neighbor finding optimizations.
 * @todo make the 32-bit be able to reorganized with different axis resolution.
 *********************************************************************/

#include "Header.h"
#include "Types/Types.h"
#include "Types/Vector3.h"

namespace ARBD {

struct MortonConfig {
	Vector3 box_min;
	Vector3 box_max;
	float cutoff_inv;
	int3 grid_cells; // Number of cells (Nx, Ny, Nz)
	float max_coord; // For Morton encoding (e.g. 1023.0f)
};

/**
 * @brief Morton code utilities with adaptive precision
 */
class MortonCode {
  public:
	// Runtime-configurable bits per dimension (set during initialization)
	static int max_coord_bits; // will be 10
	static coord_t max_coord;  // Will be (1 << max_coord_bits) - 1
	static float actual_spacing;
	static morton_t morton_threshold;

	// Device-safe constant for SYCL kernels (use max possible value)
	static constexpr coord_t max_coord_device = (1u << 10) - 1; // 1023 for 10 bits

	/**
	 * @brief Initialize Morton code precision based on system parameters
	 * @param box_size Largest dimension of the simulation box
	 * @param cutoff Interaction cutoff distance
	 *
	 * Automatically selects bits per dimension to ensure:
	 *   grid_spacing = box_size / (2^bits) < 0.1 * cutoff
	 */
	HOST static void initialize(float box_size, float cutoff) {
		// We want: box_size / (2^bits) < 0.1 * cutoff
		// So: 2^bits > box_size / (0.1 * cutoff)
		// bits > log2(box_size / (0.1 * cutoff))

		float required_resolution = box_size / (0.1f * cutoff);
		int required_bits = static_cast<int>(std::ceil(std::log2(required_resolution)));

		// Clamp to valid range for 32-bit Morton codes
		// 3 * bits_per_dim must be <= 32, so bits_per_dim <= 10
		if (required_bits <= 8) {
			max_coord_bits = 8; // Conservative: 256 levels per dimension
			morton_threshold = 0x40000;
		} else if (required_bits <= 9) {
			max_coord_bits = 9; // Conservative: 512 levels per dimension
			morton_threshold = 0x80000;
		} else if (required_bits <= 10) {
			max_coord_bits = 10; // Standard: 1024 levels per dimension
			morton_threshold = 0x100000;
		} else {
			// Box is too large relative to cutoff for 32-bit Morton codes
			LOGWARN("Box size {} is very large relative to cutoff {}. "
					"Morton code resolution may be insufficient. "
					"Consider using 64-bit Morton codes or smaller patches.",
					box_size,
					cutoff);
			max_coord_bits = 10; // Use maximum available
			morton_threshold = 0x100000;
		}

		max_coord = (1u << max_coord_bits) - 1;

		actual_spacing = box_size / (1u << max_coord_bits);
		LOGINFO("MortonCode: Using {} bits/dim ({} levels), "
				"grid spacing = {:.4f} nm (cutoff = {:.4f} nm, ratio = {:.3f})",
				max_coord_bits,
				(1u << max_coord_bits),
				actual_spacing,
				cutoff,
				actual_spacing / cutoff);
	}

	HOST DEVICE static morton_t encode(coord_t x, coord_t y, coord_t z) {
		return splitBy3(z) | (splitBy3(y) << 1) | (splitBy3(x) << 2);
	}

	HOST DEVICE static morton_t
	encode(const Vector3& pos, const Vector3& box_min, const Vector3& box_max) {
		Vector3 range = box_max - box_min;
		Vector3 offset = pos - box_min;
		Vector3 normalized = Vector3(offset.x / range.x, offset.y / range.y, offset.z / range.z);

		// Use device-safe constant in device code
		constexpr coord_t max_c = max_coord_device;

#ifdef USE_SYCL
		coord_t x = static_cast<coord_t>(
			sycl::fmin(sycl::fmax(normalized.x * max_c, 0.0f), static_cast<float>(max_c)));
		coord_t y = static_cast<coord_t>(
			sycl::fmin(sycl::fmax(normalized.y * max_c, 0.0f), static_cast<float>(max_c)));
		coord_t z = static_cast<coord_t>(
			sycl::fmin(sycl::fmax(normalized.z * max_c, 0.0f), static_cast<float>(max_c)));
#else
		coord_t x = static_cast<coord_t>(
			fminf(fmaxf(normalized.x * max_c, 0.0f), static_cast<float>(max_c)));
		coord_t y = static_cast<coord_t>(
			fminf(fmaxf(normalized.y * max_c, 0.0f), static_cast<float>(max_c)));
		coord_t z = static_cast<coord_t>(
			fminf(fmaxf(normalized.z * max_c, 0.0f), static_cast<float>(max_c)));
#endif

		return encode(x, y, z);
	}

	HOST DEVICE static void decode(morton_t code, coord_t& x, coord_t& y, coord_t& z) {
		x = compactBy3(code >> 2);
		y = compactBy3(code >> 1);
		z = compactBy3(code);
	}

	HOST DEVICE static Vector3
	decode(morton_t code, const Vector3& box_min, const Vector3& box_max) {
		coord_t x, y, z;
		decode(code, x, y, z);

		constexpr coord_t max_c = max_coord_device;
		Vector3 normalized(static_cast<float>(x) / max_c,
						   static_cast<float>(y) / max_c,
						   static_cast<float>(z) / max_c);

		return box_min + normalized.element_mult(box_max - box_min);
	}

  private:
	/**
	 * @brief Split bits by 3 - supports up to 11 bits input
	 * Works for 9, 10, or 11 bit inputs
	 */
	HOST DEVICE static morton_t splitBy3(coord_t a) {
		// Mask based on max_coord_bits (use device-safe constant)
		constexpr coord_t max_c = max_coord_device;
		morton_t x = a & max_c; // Keep only valid bits

		// Generic bit-splitting that works for 9-11 bits
		x = (x | x << 16) & 0x30000ff;
		x = (x | x << 8) & 0x300f00f;
		x = (x | x << 4) & 0x30c30c3;
		x = (x | x << 2) & 0x9249249;

		return x;
	}

	HOST DEVICE static coord_t compactBy3(morton_t x) {
		constexpr coord_t max_c = max_coord_device;
		x &= 0x9249249;
		x = (x ^ (x >> 2)) & 0x30c30c3;
		x = (x ^ (x >> 4)) & 0x300f00f;
		x = (x ^ (x >> 8)) & 0x30000ff;
		x = (x ^ (x >> 16)) & max_c;

		return static_cast<coord_t>(x);
	}
};

} // namespace ARBD
