#include "System/ZOrderSort.h"
#include "ARBDException.h"
#include "ARBDLogger.h"
#ifdef USE_CUDA
#include "System/ZOrderKernels/CUDASort.h"
#elif defined(USE_SYCL)
#include "System/ZOrderKernels/SYCLSort.h"
#endif

namespace ARBD {

ZOrderSort::ZOrderSort(const Resource& resource, size_t max_particles, ZOrderOptimizationMode mode)
	: resource_(resource), max_particles_(max_particles), num_particles_(0),
	  optimization_mode_(mode), morton_codes_(max_particles, resource),
	  sorted_indices_(max_particles, resource), inverse_indices_(max_particles, resource),
	  temp_morton_codes_(max_particles, resource), temp_indices_(max_particles, resource),
	  cub_temp_storage_(0, resource), bbox_min_(1, resource), bbox_max_(1, resource),
	  error_count_(1, resource),
	  // Initialize Pairlist mode buffers only if needed
	  old_positions_(mode == ZOrderOptimizationMode::Pairlist ? max_particles : 0, resource),
	  max_displacement_(mode == ZOrderOptimizationMode::Pairlist ? 1 : 0, resource),
	  invalid_count_(mode == ZOrderOptimizationMode::Pairlist ? 1 : 0, resource) {

	// Configure mode-specific settings
	if (optimization_mode_ == ZOrderOptimizationMode::Pairlist) {
		smart_updates_enabled_ = true;
		max_displacement_.fill(0.0f);
		invalid_count_.fill(0);
		LOGINFO("Created ZOrderSort (Pairlist mode) with capacity for {} particles on {}",
				max_particles_,
				resource_.toString());
	} else {
		smart_updates_enabled_ = false;
		LOGINFO("Created ZOrderSort (System mode) with capacity for {} particles on {}",
				max_particles_,
				resource_.toString());
	}
}

void ZOrderSort::sort_particles(const DeviceBuffer<Vector3>& positions,
								size_t num_particles,
								const Vector3& box_min,
								const Vector3& box_max) {
	if (num_particles > max_particles_) {
		ARBD_Exception(ExceptionType::ValueError,
					   "Cannot sort {} particles, maximum is {}",
					   num_particles,
					   max_particles_);
	}

	num_particles_ = num_particles;

	LOGTRACE("Sorting {} particles using Z-order", num_particles_);

	// Step 1: Encode positions to Morton codes
	encode_morton_codes(positions, num_particles_, box_min, box_max);

	// Step 2: Sort by Morton codes
	sort_morton_codes();

	// Step 3: Create inverse mapping
	create_inverse_mapping();

	LOGTRACE("Z-order sorting completed");
}

void ZOrderSort::compute_bounding_box(const DeviceBuffer<Vector3>& positions,
									  size_t num_particles,
									  Vector3& box_min,
									  Vector3& box_max) {
	// Initialize bounds
	Vector3 init_min((std::numeric_limits<float>::max()));
	Vector3 init_max(std::numeric_limits<float>::lowest());

	bbox_min_.copy_from_host(&init_min, 1);
	bbox_max_.copy_from_host(&init_max, 1);

	// Launch kernel to find bounds
	BoundingBoxKernel kernel{positions.data(), bbox_min_.data(), bbox_max_.data(), num_particles};

	KernelConfig config = KernelConfig::for_1d(num_particles, resource_);
	launch_kernel(resource_, config, kernel);

	// Copy results back
	bbox_min_.copy_to_host(&box_min, 1, true);
	bbox_max_.copy_to_host(&box_max, 1, true);
}

void ZOrderSort::encode_morton_codes(const DeviceBuffer<Vector3>& positions,
									 size_t num_particles,
									 const Vector3& box_min,
									 const Vector3& box_max) {

	KernelConfig config = KernelConfig::for_1d(num_particles, resource_);
	launch_kernel(resource_,
				  config,
				  MortonEncodeKernel{},
				  positions.data(),
				  morton_codes_.data(),
				  sorted_indices_.data(),
				  box_min,
				  box_max,
				  num_particles);
}

void ZOrderSort::create_inverse_mapping() {
	KernelConfig config = KernelConfig::for_1d(num_particles_, resource_);
	launch_kernel(resource_,
				  config,
				  InverseIndexKernel{},
				  sorted_indices_.data(),
				  inverse_indices_.data(),
				  num_particles_);
}

uint32_t ZOrderSort::validate_sorting() {
	// Reset error count
	uint32_t zero = 0;
	error_count_.copy_from_host(&zero, 1);

	// Launch validation kernel
	KernelConfig config = KernelConfig::for_1d(num_particles_, resource_);
	launch_kernel(resource_,
				  config,
				  ValidateZOrderKernel{},
				  morton_codes_.data(),
				  error_count_.data(),
				  num_particles_);

	// Get result
	uint32_t errors;
	error_count_.copy_to_host(&errors, 1, true);

	if (errors > 0) {
		LOGWARN("Z-order validation found {} sorting errors", errors);
	} else {
		LOGTRACE("Z-order validation passed");
	}

	return errors;
}

void ZOrderSort::resize(size_t new_max_particles) {
	if (new_max_particles == max_particles_) {
		return;
	}

	LOGINFO("Resizing ZOrderSort from {} to {} particles", max_particles_, new_max_particles);

	max_particles_ = new_max_particles;

	// Resize all buffers
	morton_codes_.resize(max_particles_);
	sorted_indices_.resize(max_particles_);
	inverse_indices_.resize(max_particles_);
	temp_morton_codes_.resize(max_particles_);
	temp_indices_.resize(max_particles_);
	// CUB temp storage will be resized dynamically as needed

	// Resize Pairlist mode buffers if enabled
	if (optimization_mode_ == ZOrderOptimizationMode::Pairlist) {
		old_positions_.resize(max_particles_);
		morton_codes_valid_ = false; // Invalidate after resize
	}
}

float ZOrderSort::compute_max_displacement(const DeviceBuffer<Vector3>& current_positions,
										   size_t num_particles) const {
	if (optimization_mode_ != ZOrderOptimizationMode::Pairlist) {
		ARBD_Exception(ExceptionType::RuntimeError,
					   "compute_max_displacement only available in Pairlist mode");
	}

	// Reset displacement buffer
	max_displacement_.fill(0.0f);

	KernelConfig config = KernelConfig::for_1d(num_particles, resource_);
	launch_kernel(resource_,
				  config,
				  DisplacementKernel{},
				  current_positions.data(),
				  old_positions_.data(),
				  max_displacement_.data(),
				  num_particles);

	// Get result from device
	float max_disp_sq;
	max_displacement_.copy_to_host(&max_disp_sq, 1, true);

	return sqrtf(max_disp_sq);
}

bool ZOrderSort::validate_morton_codes(const DeviceBuffer<Vector3>& current_positions,
									   size_t num_particles,
									   const Vector3& box_min,
									   const Vector3& box_max) {
	if (optimization_mode_ != ZOrderOptimizationMode::Pairlist) {
		ARBD_Exception(ExceptionType::RuntimeError,
					   "validate_morton_codes only available in Pairlist mode");
	}

	// Reset invalid count
	invalid_count_.fill(0);
	
	KernelConfig config = KernelConfig::for_1d(num_particles, resource_);
	launch_kernel(resource_,
				  config,
				  MortonValidationKernel{},
				  current_positions.data(),
				  morton_codes_.data(),
				  sorted_indices_.data(),
				  box_min,
				  box_max,
				  invalid_count_.data(),
				  num_particles);

	// Get result from device
	uint32_t invalid_count;
	invalid_count_.copy_to_host(&invalid_count, 1, true);

	// Consider Morton codes valid if less than 5% of particles have invalid codes
	float invalid_fraction = static_cast<float>(invalid_count) / num_particles;
	return invalid_fraction < 0.05f;
}

void ZOrderSort::update_positions_incremental(const DeviceBuffer<Vector3>& positions,
											  size_t num_particles) {
	if (optimization_mode_ != ZOrderOptimizationMode::Pairlist) {
		ARBD_Exception(ExceptionType::RuntimeError,
					   "update_positions_incremental only available in Pairlist mode");
	}

	// Store new positions for next update detection without host roundtrip
	old_positions_.copy_device_to_device_sync(positions, num_particles);
}

} // namespace ARBD
