#include "Compute/ZOrderPairlist.h"
#include "ARBDException.h"
#include "ARBDLogger.h"
#include "Backend/Kernels.h"
#include "Compute/PairListKernels/ZOrderNeighbor.h"
#include <chrono>

namespace ARBD {

ZOrderPairlist::ZOrderPairlist(const Resource& resource, size_t max_particles, size_t max_pairs)
	: Pairlist(resource, max_particles, max_pairs), sorter_(resource, max_particles),
	  sorted_positions_(max_particles, resource), old_positions_(max_particles, resource),
	  search_range_(64), // Default search range
	  auto_bbox_(true), manual_box_min_(0.0f), manual_box_max_(1.0f), last_build_time_ms_(0.0),
	  last_max_neighbors_(0) {

	LOGINFO("Created ZOrderPairlist with capacity {} particles, {} pairs on {}",
			max_particles_,
			max_pairs_,
			resource_.toString());
}

void ZOrderPairlist::build_pairlist(const DeviceBuffer<Vector3>& positions,
									size_t num_particles,
									float cutoff) {
	auto start_time = std::chrono::high_resolution_clock::now();

	if (num_particles > max_particles_) {
		ARBD_Exception(ExceptionType::ValueError,
					   "Cannot build pairlist for {} particles, maximum is {}",
					   num_particles,
					   max_particles_);
	}

	LOGTRACE("Building Z-order pairlist for {} particles with cutoff {}", num_particles, cutoff);

	// Step 1: Determine bounding box
	Vector3 box_min, box_max;
	get_bounding_box(positions, num_particles, box_min, box_max);

	// Step 2: Sort particles by Morton code
	sorter_.sort_particles(positions, num_particles, box_min, box_max);

	// Step 3: Reorder positions for cache-friendly access
	sorter_.reorder_data(positions, sorted_positions_, num_particles);

	// Step 4: Find neighbors using sorted order
	reset_pair_count();
	find_neighbors_zorder(num_particles);

	// Step 5: Update internal state
	update_state(num_particles, cutoff);

	// Store positions for update detection
	positions.copy_to_host(old_positions_.data(), num_particles);

	auto end_time = std::chrono::high_resolution_clock::now();
	last_build_time_ms_ = std::chrono::duration<double, std::milli>(end_time - start_time).count();

	LOGTRACE("Z-order pairlist built: {} pairs in {:.2f} ms", num_pairs_, last_build_time_ms_);
}

void ZOrderPairlist::update_pairlist(const DeviceBuffer<Vector3>& positions, size_t num_particles) {
	// For now, always rebuild
	// Future optimization: check if particles have moved significantly
	// and reuse existing sort if displacement is small
	build_pairlist(positions, num_particles, cutoff_);
}

bool ZOrderPairlist::needs_update(const DeviceBuffer<Vector3>& positions,
								  const DeviceBuffer<Vector3>& old_positions,
								  size_t num_particles,
								  float skin_distance) const {
	// Simple criterion: maximum displacement > skin_distance/2
	// More sophisticated implementations could track average displacement
	// or use other criteria

	// For now, always return true (conservative approach)
	// TODO: Implement displacement-based update criterion
	return true;
}

void ZOrderPairlist::resize(size_t new_max_particles, size_t new_max_pairs) {
	Pairlist::resize(new_max_particles, new_max_pairs);

	sorter_.resize(new_max_particles);
	sorted_positions_.resize(new_max_particles);
	old_positions_.resize(new_max_particles);

	LOGINFO("Resized ZOrderPairlist to {} particles, {} pairs", new_max_particles, new_max_pairs);
}

Pairlist::Statistics ZOrderPairlist::get_statistics() const {
	Statistics stats = Pairlist::get_statistics();
	stats.max_neighbors_per_particle = last_max_neighbors_;
	stats.build_time_ms = last_build_time_ms_;
	return stats;
}

void ZOrderPairlist::find_neighbors_zorder(size_t num_particles) {
	ZOrderNeighborKernel kernel{sorted_positions_.data(),
								sorter_.get_morton_codes().data(),
								sorter_.get_sorted_indices().data(),
								neighbor_pairs_.data(),
								pair_count_.data(),
								cutoff_squared_,
								num_particles,
								max_pairs_};

	// Update search range in kernel based on current setting
	// Note: This requires modifying the kernel structure to accept search_range
	// For now, the kernel uses a fixed range of 64

	KernelConfig config = KernelConfig::for_1d(num_particles, resource_);
	launch_kernel(resource_, config, kernel);
}

void ZOrderPairlist::get_bounding_box(const DeviceBuffer<Vector3>& positions,
									  size_t num_particles,
									  Vector3& box_min,
									  Vector3& box_max) const {
	if (auto_bbox_) {
		// Compute bounding box from particle positions
		// This is a simplified version - production code should use reduction
		BoundingBoxKernel kernel{positions.data(), &box_min, &box_max, num_particles};

		// Initialize bounds
		box_min = Vector3(std::numeric_limits<float>::max());
		box_max = Vector3(std::numeric_limits<float>::lowest());

		// Create temporary buffers for reduction
		DeviceBuffer<Vector3> temp_min(1, resource_);
		DeviceBuffer<Vector3> temp_max(1, resource_);

		temp_min.copy_from_host(&box_min, 1);
		temp_max.copy_from_host(&box_max, 1);

		BoundingBoxKernel bbox_kernel{positions.data(),
									  temp_min.data(),
									  temp_max.data(),
									  num_particles};

		KernelConfig config = KernelConfig::for_1d(num_particles, resource_);
		launch_kernel(resource_, config, bbox_kernel);

		temp_min.copy_to_host(&box_min, 1, true);
		temp_max.copy_to_host(&box_max, 1, true);

		// Add small margin to avoid boundary issues
		Vector3 margin = (box_max - box_min) * 0.01f;
		box_min -= margin;
		box_max += margin;
	} else {
		// Use manually specified bounds
		box_min = manual_box_min_;
		box_max = manual_box_max_;
	}
}

} // namespace ARBD
