#pragma once

/*********************************************************************
 * @file  MortonCode.h
 *
 * @brief Morton (Z-order) code utilities for 3D spatial indexing
 *
 * This file contains utilities for encoding 3D coordinates into
 * Morton codes for spatial sorting and neighbor finding optimizations.
 *********************************************************************/

#include "Types/Types.h"
#include "Types/Vector3.h"
#include "Header.h"
#include <cstdint>

namespace ARBD {

/**
 * @brief Morton code utilities for 3D space-filling curves
 *
 * Provides functions to convert 3D coordinates to Morton codes
 * for cache-friendly spatial data organization.
 */
class MortonCode {
  public:

	static constexpr int MAX_COORD_BITS = 21; // 21 bits per dimension = 63 bits total
	static constexpr coord_t MAX_COORD = (1u << MAX_COORD_BITS) - 1;

	/**
	 * @brief Encode 3D coordinates into a Morton code
	 * @param x X coordinate (normalized to [0, MAX_COORD])
	 * @param y Y coordinate (normalized to [0, MAX_COORD])
	 * @param z Z coordinate (normalized to [0, MAX_COORD])
	 * @return 64-bit Morton code
	 */
	HOST DEVICE static morton_t encode(coord_t x, coord_t y, coord_t z) {
		return splitBy3(z) | (splitBy3(y) << 1) | (splitBy3(x) << 2);
	}

	/**
	 * @brief Encode 3D position vector into Morton code
	 * @param pos 3D position
	 * @param box_min Minimum bounds of the domain
	 * @param box_max Maximum bounds of the domain
	 * @return 64-bit Morton code
	 */
	HOST DEVICE static morton_t
	encode(const Vector3& pos, const Vector3& box_min, const Vector3& box_max) {
		// Normalize coordinates to [0, MAX_COORD]
		Vector3 normalized = (pos - box_min).element_mult(1.0f / (box_max - box_min));

		// Clamp to valid range and convert to integer coordinates
		coord_t x = static_cast<coord_t>(
			fmin(fmax(normalized.x * MAX_COORD, 0.0f), static_cast<float>(MAX_COORD)));
		coord_t y = static_cast<coord_t>(
			fmin(fmax(normalized.y * MAX_COORD, 0.0f), static_cast<float>(MAX_COORD)));
		coord_t z = static_cast<coord_t>(
			fmin(fmax(normalized.z * MAX_COORD, 0.0f), static_cast<float>(MAX_COORD)));

		return encode(x, y, z);
	}

	/**
	 * @brief Decode Morton code back to 3D coordinates
	 * @param code Morton code to decode
	 * @return Tuple of (x, y, z) coordinates
	 */
	HOST DEVICE static void decode(morton_t code, coord_t& x, coord_t& y, coord_t& z) {
		x = compactBy3(code >> 2);
		y = compactBy3(code >> 1);
		z = compactBy3(code);
	}

	/**
	 * @brief Decode Morton code back to 3D position
	 * @param code Morton code to decode
	 * @param box_min Minimum bounds of the domain
	 * @param box_max Maximum bounds of the domain
	 * @return 3D position vector
	 */
	HOST DEVICE static Vector3
	decode(morton_t code, const Vector3& box_min, const Vector3& box_max) {
		coord_t x, y, z;
		decode(code, x, y, z);

		Vector3 normalized(static_cast<float>(x) / MAX_COORD,
						   static_cast<float>(y) / MAX_COORD,
						   static_cast<float>(z) / MAX_COORD);

		return box_min + normalized.element_mult(box_max - box_min);
	}

  private:
	/**
	 * @brief Split bits by 3 (insert 2 zeros between each bit)
	 * Used for Morton encoding: 0b00000abc becomes 0b000a000b000c
	 */
	HOST DEVICE static morton_t splitBy3(coord_t a) {
		morton_t x = a & 0x1fffff; // Only keep lower 21 bits

		x = (x | x << 32) & 0x1f00000000ffff;
		x = (x | x << 16) & 0x1f0000ff0000ff;
		x = (x | x << 8) & 0x100f00f00f00f00f;
		x = (x | x << 4) & 0x10c30c30c30c30c3;
		x = (x | x << 2) & 0x1249249249249249;

		return x;
	}

	/**
	 * @brief Compact bits by 3 (remove 2 zeros between each bit)
	 * Used for Morton decoding: reverse of splitBy3
	 */
	HOST DEVICE static coord_t compactBy3(morton_t x) {
		x &= 0x1249249249249249;
		x = (x ^ (x >> 2)) & 0x10c30c30c30c30c3;
		x = (x ^ (x >> 4)) & 0x100f00f00f00f00f;
		x = (x ^ (x >> 8)) & 0x1f0000ff0000ff;
		x = (x ^ (x >> 16)) & 0x1f00000000ffff;
		x = (x ^ (x >> 32)) & 0x1fffff;

		return static_cast<coord_t>(x);
	}
};

} // namespace ARBD
