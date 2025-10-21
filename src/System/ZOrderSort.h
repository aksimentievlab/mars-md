#pragma once

/*********************************************************************
 * @file  ZOrderSort.h
 *
 * @brief Z-order (Morton) sorting for spatial data organization
 *
 * This class provides Morton code based sorting for particles to
 * improve spatial locality and cache performance in neighbor finding
 * and force computation operations.
 *********************************************************************/

#include "Backend/Buffer.h"
#include "Backend/Kernels.h"
#include "Backend/Resource.h"
#include "System/MortonCode.h"
#include "System/ZOrderKernels.h"
#include "Types/Types.h"
#include "Types/Vector3.h"

#ifdef USE_CUDA
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>
#include <thrust/zip_iterator.h>
#endif

namespace ARBD {

/**
 * @brief Z-order spatial sorting for improved memory locality
 *
 * Sorts particles by Morton codes to improve spatial locality for
 * neighbor finding and force computation. Works within individual
 * GPU patches for optimal multi-device performance.
 */
class ZOrderSort {
public:
    using morton_t = MortonCode::morton_t;

    /**
     * @brief Constructor
     * @param resource Computing resource for this sorter
     * @param max_particles Maximum number of particles to handle
     */
    ZOrderSort(const Resource& resource, size_t max_particles);

    /**
     * @brief Destructor
     */
    ~ZOrderSort() = default;

    /**
     * @brief Sort particles by Morton code
     * @param positions Input particle positions
     * @param num_particles Number of particles to sort
     * @param box_min Minimum bounds of the domain
     * @param box_max Maximum bounds of the domain
     */
    void sort_particles(const DeviceBuffer<Vector3>& positions,
                       size_t num_particles,
                       const Vector3& box_min,
                       const Vector3& box_max);

    /**
     * @brief Reorder any data array according to the last sort
     * @param input_data Input data to reorder
     * @param output_data Output buffer for reordered data
     * @param num_elements Number of elements to reorder
     */
    template<typename T>
    void reorder_data(const DeviceBuffer<T>& input_data,
                     DeviceBuffer<T>& output_data,
                     size_t num_elements);

    /**
     * @brief Get sorted Morton codes
     */
    const DeviceBuffer<morton_t>& get_morton_codes() const { return morton_codes_; }

    /**
     * @brief Get sorted particle indices (original particle IDs)
     */
    const DeviceBuffer<uint32_t>& get_sorted_indices() const { return sorted_indices_; }

    /**
     * @brief Get inverse mapping (sorted position → original index)
     */
    const DeviceBuffer<uint32_t>& get_inverse_indices() const { return inverse_indices_; }

    /**
     * @brief Get current number of sorted particles
     */
    size_t get_num_particles() const { return num_particles_; }

    /**
     * @brief Validate sorting (for debugging)
     * @return Number of sorting errors found
     */
    uint32_t validate_sorting();

    /**
     * @brief Resize buffers for different particle counts
     * @param new_max_particles New maximum capacity
     */
    void resize(size_t new_max_particles);

private:
    Resource resource_;
    size_t max_particles_;
    size_t num_particles_;

    // Core sorting data
    DeviceBuffer<morton_t> morton_codes_;      ///< Morton codes for particles
    DeviceBuffer<uint32_t> sorted_indices_;    ///< Original particle indices after sorting
    DeviceBuffer<uint32_t> inverse_indices_;   ///< Mapping from original to sorted position

    // Temporary buffers for sorting
    DeviceBuffer<morton_t> temp_morton_codes_;
    DeviceBuffer<uint32_t> temp_indices_;

    // Bounding box computation
    DeviceBuffer<Vector3> bbox_min_;
    DeviceBuffer<Vector3> bbox_max_;

    // Validation
    DeviceBuffer<uint32_t> error_count_;

    /**
     * @brief Compute bounding box of particles
     * @param positions Input positions
     * @param num_particles Number of particles
     * @param box_min Output minimum bounds
     * @param box_max Output maximum bounds
     */
    void compute_bounding_box(const DeviceBuffer<Vector3>& positions,
                             size_t num_particles,
                             Vector3& box_min,
                             Vector3& box_max);

    /**
     * @brief Encode positions to Morton codes
     * @param positions Input positions
     * @param num_particles Number of particles
     * @param box_min Domain minimum bounds
     * @param box_max Domain maximum bounds
     */
    void encode_morton_codes(const DeviceBuffer<Vector3>& positions,
                            size_t num_particles,
                            const Vector3& box_min,
                            const Vector3& box_max);

    /**
     * @brief Sort Morton codes and indices
     */
    void sort_morton_codes();

    /**
     * @brief Create inverse index mapping
     */
    void create_inverse_mapping();
};

// Template implementation
template<typename T>
void ZOrderSort::reorder_data(const DeviceBuffer<T>& input_data,
                             DeviceBuffer<T>& output_data,
                             size_t num_elements) {
    if (num_elements > max_particles_) {
        ARBD_Exception(ExceptionType::ValueError,
                       "Cannot reorder {} elements, maximum is {}",
                       num_elements, max_particles_);
    }

    if (output_data.size() < num_elements) {
        output_data.resize(num_elements);
    }

    ReorderDataKernel<T> kernel{
        input_data.data(),
        output_data.data(),
        sorted_indices_.data(),
        num_elements
    };

    KernelConfig config = KernelConfig::for_1d(num_elements, resource_);
    launch_kernel(resource_, config, kernel);
}

} // namespace ARBD