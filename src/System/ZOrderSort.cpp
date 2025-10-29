#include "System/ZOrderSort.h"
#include "ARBDException.h"
#include "ARBDLogger.h"
#ifdef USE_CUDA
#include "ZOrderKernels/CUDASort.h"
#include <limits>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>
#include <thrust/zip_iterator.h>
#endif
#if defined(USE_SYCL)
#include "System/ZOrderKernels/SYCLSort.h"
#endif

namespace ARBD {

ZOrderSort::ZOrderSort(const Resource& resource, size_t max_particles)
	: resource_(resource), max_particles_(max_particles), num_particles_(0),
	  morton_codes_(max_particles, resource), sorted_indices_(max_particles, resource),
	  inverse_indices_(max_particles, resource), temp_morton_codes_(max_particles, resource),
	  temp_indices_(max_particles, resource), bbox_min_(1, resource), bbox_max_(1, resource),
	  error_count_(1, resource) {

	LOGINFO("Created ZOrderSort with capacity for {} particles on {}",
			max_particles_,
			resource_.toString());
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
	MortonEncodeKernel kernel{positions.data(),
							  morton_codes_.data(),
							  sorted_indices_.data(),
							  box_min,
							  box_max,
							  num_particles};

	KernelConfig config = KernelConfig::for_1d(num_particles, resource_);
	launch_kernel(resource_, config, kernel);
}

void ZOrderSort::create_inverse_mapping() {
	InverseIndexKernel kernel{sorted_indices_.data(), inverse_indices_.data(), num_particles_};

	KernelConfig config = KernelConfig::for_1d(num_particles_, resource_);
	launch_kernel(resource_, config, kernel);
}

uint32_t ZOrderSort::validate_sorting() {
	// Reset error count
	uint32_t zero = 0;
	error_count_.copy_from_host(&zero, 1);

	// Launch validation kernel
	ValidateZOrderKernel kernel{morton_codes_.data(), error_count_.data(), num_particles_};

	KernelConfig config = KernelConfig::for_1d(num_particles_, resource_);
	launch_kernel(resource_, config, kernel);

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
}

} // namespace ARBD
