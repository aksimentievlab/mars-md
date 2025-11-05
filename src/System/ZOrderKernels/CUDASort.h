#pragma once

#include "Types/Types.h"
#include "Backend/Buffer.h"

#ifdef USE_CUDA
namespace ARBD {

// Forward declarations for CUDA sorting functions
void sort_morton_codes_cuda(DeviceBuffer<morton_t>& morton_codes_in,
                           DeviceBuffer<uint32_t>& indices_in,
                           DeviceBuffer<morton_t>& morton_codes_out,
                           DeviceBuffer<uint32_t>& indices_out,
                           DeviceBuffer<uint8_t>& temp_storage,
                           size_t num_particles);

size_t get_sort_temp_storage_size_cuda(size_t num_particles);

} // namespace ARBD
#endif