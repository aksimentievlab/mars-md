#pragma once

#include "Backend/Buffer.h"
#include "Header.h"

namespace ARBD {

// Constants
constexpr uint32_t DRS_RADIX = 256;
constexpr uint32_t DRS_RADIX_MASK = 255;
constexpr uint32_t DRS_RADIX_LOG = 8;
constexpr uint32_t DRS_PART_SIZE = 7680;
constexpr uint32_t DRS_VEC_PART_SIZE = 1920;
constexpr uint32_t DRS_BIN_PART_SIZE = 7680;
constexpr uint32_t DRS_BIN_HISTS_SIZE = 4096;
constexpr uint32_t DRS_BIN_SUB_PART_SIZE = 480;
constexpr uint32_t DRS_BIN_WARPS = 16;
constexpr uint32_t DRS_BIN_KEYS_PER_THREAD = 15;
constexpr uint32_t DRS_BIN_THREADS = DRS_BIN_WARPS * 32; // 512
constexpr uint32_t DRS_SCAN_THREADS = 128;
constexpr uint32_t DRS_UPSWEEP_THREADS = 512;
/**
 * @brief Complete radix sort implementation for Morton codes using SYCL
 *
 * This implements a multi-pass radix sort where each pass sorts on 8 bits.
 * For 64-bit Morton codes, this requires 8 passes (bits 0-7, 8-15, ..., 56-63).
 *
 * Each pass consists of:
 * 1. Local histogram computation per work-group
 * 2. Global histogram collection and prefix sum
 * 3. Scatter phase to reorder elements
 */
void sort_morton_codes_sycl(size_t num_particles,
							DeviceBuffer<morton_t>& morton_codes_in,
							DeviceBuffer<uint32_t>& indices_in,
							DeviceBuffer<morton_t>& morton_codes_out,
							DeviceBuffer<uint32_t>& indices_out,
							DeviceBuffer<uint8_t>& temp_storage,
							const Resource& resource);

} // namespace ARBD
