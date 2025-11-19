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

// Forward declarations for CUDA/SYCL specific functions
#ifdef USE_CUDA
namespace ARBD {
void sort_morton_codes_cuda(size_t num_particles,
							DeviceBuffer<morton_t>& morton_codes_in,
							DeviceBuffer<uint32_t>& indices_in,
							DeviceBuffer<morton_t>& morton_codes_out,
							DeviceBuffer<uint32_t>& indices_out,
							DeviceBuffer<uint8_t>& temp_storage,
							const Resource& resource);
}
#elif defined(USE_SYCL)
#include "System/ZOrderKernels/SYCLSort.h"
#endif

namespace ARBD {

/**
 * @brief Optimization modes for different Z-order sorting use cases
 */
enum class ZOrderOptimizationMode {
	System,	 ///< Optimized for large, infrequent sorts (decomposition)
	Pairlist ///< Optimized for small, frequent sorts with smart updates
};

/**
 * @brief Z-order spatial sorting for improved memory locality
 *
 * Unified sorting class that supports different optimization modes:
 * - System mode: Maximum throughput for global decomposition
 * - Pairlist mode: Low latency with smart updates for neighbor finding
 *
 * Sorts particles by Morton codes to improve spatial locality for
 * neighbor finding and force computation. Works within individual
 * GPU patches for optimal multi-device performance.
 */
class ZOrderSort {
  public:
	/**
	 * @brief Constructor
	 * @param resource Computing resource for this sorter
	 * @param max_particles Maximum number of particles to handle
	 * @param mode Optimization mode for this sorter
	 */
	ZOrderSort(const Resource& resource,
			   size_t max_particles,
			   ZOrderOptimizationMode mode = ZOrderOptimizationMode::Pairlist);

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
	const DeviceBuffer<morton_t>& get_morton_codes() const {
		return morton_codes_;
	}

	/**
	 * @brief Get sorted particle indices (original particle IDs)
	 */
	const DeviceBuffer<uint32_t>& get_sorted_indices() const {
		return sorted_indices_;
	}

	/**
	 * @brief Get inverse mapping (sorted position → original index)
	 */
	const DeviceBuffer<uint32_t>& get_inverse_indices() const {
		return inverse_indices_;
	}

	/**
	 * @brief Get current number of sorted particles
	 */
	size_t get_num_particles() const {
		return num_particles_;
	}

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

	/**
	 * @brief Set optimization mode
	 * @param mode New optimization mode
	 */
	void set_optimization_mode(ZOrderOptimizationMode mode) {
		optimization_mode_ = mode;
	}

	/**
	 * @brief Get current optimization mode
	 */
	ZOrderOptimizationMode get_optimization_mode() const {
		return optimization_mode_;
	}

	/**
	 * @brief Enable/disable smart updates (Pairlist mode only)
	 * @param enable Whether to enable smart displacement-based updates
	 */
	void enable_smart_updates(bool enable) {
		smart_updates_enabled_ = enable && (optimization_mode_ == ZOrderOptimizationMode::Pairlist);
	}

	/**
	 * @brief Set displacement thresholds for smart updates (Pairlist mode only)
	 * @param validation_threshold Threshold for Morton code validation
	 * @param update_threshold Threshold for full rebuild
	 */
	void set_displacement_thresholds(float validation_threshold, float update_threshold) {
		validation_threshold_ = validation_threshold;
		update_threshold_ = update_threshold;
	}

	/**
	 * @brief Compute maximum displacement between current and previous positions (Pairlist mode)
	 * @param current_positions Current particle positions
	 * @param num_particles Number of particles
	 * @return Maximum displacement
	 */
	float compute_max_displacement(const DeviceBuffer<Vector3>& current_positions,
								   size_t num_particles) const;

  private:
	Resource resource_;
	size_t max_particles_;
	size_t num_particles_;
	ZOrderOptimizationMode optimization_mode_;

	// Core sorting data
	DeviceBuffer<morton_t> morton_codes_;	 ///< Morton codes for particles
	DeviceBuffer<uint32_t> sorted_indices_;	 ///< Original particle indices after sorting
	DeviceBuffer<uint32_t> inverse_indices_; ///< Mapping from original to sorted position

	// Temporary buffers for sorting
	DeviceBuffer<morton_t> temp_morton_codes_;
	DeviceBuffer<uint32_t> temp_indices_;

	// CUB temporary storage for device-side sorting
	DeviceBuffer<uint8_t> cub_temp_storage_;

	// Bounding box computation
	DeviceBuffer<Vector3> bbox_min_;
	DeviceBuffer<Vector3> bbox_max_;

	// Validation
	DeviceBuffer<uint32_t> error_count_;

	// Pairlist mode specific features
	bool smart_updates_enabled_{false};
	float validation_threshold_{0.05f};
	float update_threshold_{0.1f};
	mutable bool morton_codes_valid_{false};

	// Displacement tracking buffers (Pairlist mode only)
	DeviceBuffer<Vector3> old_positions_;		   ///< Previous positions for displacement tracking
	mutable DeviceBuffer<float> max_displacement_; ///< Device-side maximum displacement
	mutable DeviceBuffer<uint32_t> invalid_count_; ///< Count of invalid Morton codes

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
	void sort_morton_codes() {
#ifdef USE_CUDA
		sort_morton_codes_cuda(morton_codes_,
							   sorted_indices_,
							   temp_morton_codes_,
							   temp_indices_,
							   cub_temp_storage_,
							   num_particles_);
// #elif defined(USE_SYCL_ICPX)
//	sort_morton_codes_oneapi(morton_codes_, sorted_indices_, num_particles_, resource_);
#elif defined(USE_SYCL)
		sort_morton_codes_sycl(morton_codes_, sorted_indices_, num_particles_, resource_);
#else
		ARBD_Exception(ExceptionType::NotImplementedError,
					   "Z-order sorting not implemented for non-CUDA or SYCL backends");
#endif
	};

	/**
	 * @brief Create inverse index mapping
	 */
	void create_inverse_mapping();

	/**
	 * @brief Validate if current Morton code ordering is still valid (Pairlist mode)
	 * @param current_positions Current particle positions
	 * @param num_particles Number of particles
	 * @param box_min Current bounding box minimum
	 * @param box_max Current bounding box maximum
	 * @return True if Morton codes are still valid
	 */
	bool validate_morton_codes(const DeviceBuffer<Vector3>& current_positions,
							   size_t num_particles,
							   const Vector3& box_min,
							   const Vector3& box_max);

	/**
	 * @brief Update positions without full resort (Pairlist mode)
	 * @param positions New particle positions
	 * @param num_particles Number of particles
	 */
	void update_positions_incremental(const DeviceBuffer<Vector3>& positions, size_t num_particles);
};

// Template implementation
template<typename T>
void ZOrderSort::reorder_data(const DeviceBuffer<T>& input_data,
							  DeviceBuffer<T>& output_data,
							  size_t num_elements) {
	if (num_elements > max_particles_) {
		ARBD_Exception(ExceptionType::ValueError,
					   "Cannot reorder {} elements, maximum is {}",
					   num_elements,
					   max_particles_);
	}

	if (output_data.size() < num_elements) {
		output_data.resize(num_elements);
	}

	ReorderDataKernel<T> kernel{input_data.data(),
								output_data.data(),
								sorted_indices_.data(),
								num_elements};

	KernelConfig config = KernelConfig::for_1d(num_elements, resource_);
	launch_kernel(resource_, config, kernel);
}

} // namespace ARBD
