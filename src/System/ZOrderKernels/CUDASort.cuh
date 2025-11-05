#pragma once

// Temporarily undefine Debug macro to avoid conflicts with CUB
#ifdef Debug
#undef Debug
#include <cub/cub.cuh>
#define Debug(x) static_cast<void>(0)
#else
#include <cub/cub.cuh>
#endif

#include "Backend/Buffer.h"
#include "Header.h"
#include "Types/Types.h"

namespace ARBD {

void sort_morton_codes_cuda(DeviceBuffer<morton_t>& morton_codes_in,
                           DeviceBuffer<uint32_t>& indices_in,
                           DeviceBuffer<morton_t>& morton_codes_out,
                           DeviceBuffer<uint32_t>& indices_out,
                           DeviceBuffer<uint8_t>& temp_storage,
                           size_t num_particles) {

    size_t temp_storage_bytes = 0;

    // Determine temporary storage requirements
    cub::DeviceRadixSort::SortPairs(nullptr, temp_storage_bytes,
                                   morton_codes_in.data(), morton_codes_out.data(),
                                   indices_in.data(), indices_out.data(),
                                   num_particles);

    // Resize temp storage if needed
    if (temp_storage.size() < temp_storage_bytes) {
        temp_storage.resize(temp_storage_bytes);
    }

    // Perform the sort
    cub::DeviceRadixSort::SortPairs(temp_storage.data(), temp_storage_bytes,
                                   morton_codes_in.data(), morton_codes_out.data(),
                                   indices_in.data(), indices_out.data(),
                                   num_particles);
}

} // namespace ARBD
