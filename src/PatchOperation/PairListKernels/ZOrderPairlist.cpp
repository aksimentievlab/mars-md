#include "ZOrderPairlist.h"
#include "ARBDException.h"
#include "ARBDLogger.h"

#include "ZOrderNeighbor.h"
#include <algorithm>
#include <chrono>
#include <cmath>

namespace ARBD {

ZOrderPairlist::ZOrderPairlist(const Resource& resource, size_t max_particles, size_t max_pairs)
	: Pairlist(resource, max_particles, max_pairs),
	  sorter_(resource, max_particles, ZOrderOptimizationMode::Pairlist), // Use Pairlist mode
	  sorted_positions_(max_particles, resource), persistent_bbox_min_(1, resource),
	  persistent_bbox_max_(1, resource), cell_begin_(kInitialCoarseCells, resource),
	  cell_end_(kInitialCoarseCells, resource),
	  cell_neighbors_(kInitialCoarseCells * MAX_NEIGHBORS, resource), auto_bbox_(true),
	  manual_box_min_(0.0f), manual_box_max_(1.0f), last_build_time_ms_(0.0),
	  last_max_neighbors_(0) {

	// Configure smart updates for Pairlist mode
	sorter_.enable_smart_updates(true);
	sorter_.set_displacement_thresholds(0.05f, 0.1f); // validation_threshold, update_threshold

	LOGDEBUG("Created ZOrderPairlist with capacity {} particles, {} pairs on {}",
			 max_particles_,
			 max_pairs_,
			 resource_.toString());
}

void ZOrderPairlist::build_pairlist(const DeviceBuffer<Vector3>& positions,
									size_t num_particles,
									float pairlist_cutoff) {
	auto start_time = std::chrono::high_resolution_clock::now();

	if (num_particles > max_particles_) {
		ARBD_Exception(ExceptionType::ValueError,
					   "Cannot build pairlist for {} particles, maximum is {}",
					   num_particles,
					   max_particles_);
	}

	LOGTRACE("Building Z-order pairlist for {} particles with cutoff {}",
			 num_particles,
			 pairlist_cutoff);

	// Step 1: Determine bounding box
	Vector3 box_min, box_max;
	get_bounding_box(positions, num_particles, box_min, box_max);

	// Periodic axes: encode against the simulation box, not the bbox. See dev_notes.
	if (box_len_.x > 0.0f) {
		box_min.x = box_origin_.x;
		box_max.x = box_origin_.x + box_len_.x;
	}
	if (box_len_.y > 0.0f) {
		box_min.y = box_origin_.y;
		box_max.y = box_origin_.y + box_len_.y;
	}
	if (box_len_.z > 0.0f) {
		box_min.z = box_origin_.z;
		box_max.z = box_origin_.z + box_len_.z;
	}

	LOGTRACE("Bounding box: [{:.6f}, {:.6f}, {:.6f}] to [{:.6f}, {:.6f}, {:.6f}]",
			 box_min.x,
			 box_min.y,
			 box_min.z,
			 box_max.x,
			 box_max.y,
			 box_max.z);
	last_box_extent_ = box_max - box_min;
	update_state(num_particles, pairlist_cutoff);
	sorter_.sort_particles(positions, num_particles, box_min, box_max);
	LOGTRACE("Sorted particles by Morton code");
	resource_.synchronize_streams();
	sorter_.reorder_data(positions, sorted_positions_, num_particles);
	LOGTRACE("Reordered positions for cache-friendly access");

	resource_.synchronize_streams();
	// Step 4: Find neighbors using sorted order
	uint32_t zero = 0;
	pair_count_.copy_from_host(&zero, 1, true);
	resource_.synchronize_streams();
	find_neighbors_zorder(num_particles);
	LOGTRACE("Found neighbors using sorted order");
	resource_.synchronize_streams();

	// Snapshot positions so needs_update() can measure drift. See dev_notes.md.
	sorter_.update_positions_incremental(positions, num_particles);

	auto end_time = std::chrono::high_resolution_clock::now();
	last_build_time_ms_ = std::chrono::duration<double, std::milli>(end_time - start_time).count();
	LOGTRACE("Z-order pairlist built in {:.2f} ms", last_build_time_ms_);
}

void ZOrderPairlist::update_pairlist(const DeviceBuffer<Vector3>& positions, size_t num_particles) {
	build_pairlist(positions, num_particles, cutoff_);
}

bool ZOrderPairlist::needs_update(const DeviceBuffer<Vector3>& positions,
								  const DeviceBuffer<Vector3>& old_positions,
								  size_t num_particles,
								  float skin_distance) const {
	float max_disp = sorter_.compute_max_displacement(positions, num_particles, box_len_);
	return max_disp > (skin_distance * 0.5f);
}

void ZOrderPairlist::resize(size_t new_max_particles, size_t new_max_pairs) {
	Pairlist::resize(new_max_particles, new_max_pairs);

	sorter_.resize(new_max_particles);
	sorted_positions_.resize(new_max_particles);

	LOGINFO("Resized ZOrderPairlist to {} particles, {} pairs", new_max_particles, new_max_pairs);
}

Pairlist::Statistics ZOrderPairlist::get_statistics() const {
	Statistics stats = Pairlist::get_statistics();
	stats.max_neighbors_per_particle = last_max_neighbors_;
	stats.build_time_ms = last_build_time_ms_;
	return stats;
}

void ZOrderPairlist::find_neighbors_zorder(size_t num_particles) {
	LOGTRACE("find_neighbors_zorder: {} particles, max_pairs={}, cutoff_squared={}",
			 num_particles,
			 max_pairs_,
			 cutoff_squared_);

	// Largest m whose cell is still >= cutoff (27-cell stencil must cover it). See dev_notes.md.
	const float cutoff = std::sqrt(cutoff_squared_);
	// max_bits from max_coord_device (compile-time), matching MortonCode::encode. See dev_notes.md.
	constexpr int max_bits = []() constexpr {
		int b = 0;
		for (auto v = MortonCode::max_coord_device; v; v >>= 1)
			++b;
		return b;
	}();
	int m = 0;
	if (cutoff > 0.0f) {
		// Size cells against the encoded extent, not the box. See dev_notes.md.
		const float min_extent =
			std::min({last_box_extent_.x, last_box_extent_.y, last_box_extent_.z});
		if (min_extent > 0.0f) {
			m = static_cast<int>(std::floor(std::log2(min_extent / cutoff)));
		}
	}
	m = std::max(0, std::min({m, max_bits, kMaxCoarseBits}));
	coarse_bits_ = m;

	const size_t num_cells = static_cast<size_t>(1) << (3 * m);
	if (cell_begin_.size() < num_cells) {
		cell_begin_.resize(num_cells);
		cell_end_.resize(num_cells);
	}
	cell_begin_.fill(0u, true);
	cell_end_.fill(0u, true);

	// Per-cell 27-neighbor table: topology depends only on the grid, so rebuild it
	// only when coarse_bits_/periodicity change (~once, at patch init). See dev_notes.md.
	const int per_mask =
		(box_len_.x > 0.0f ? 1 : 0) | (box_len_.y > 0.0f ? 2 : 0) | (box_len_.z > 0.0f ? 4 : 0);
	if (cell_neighbors_.size() < num_cells * static_cast<size_t>(MAX_NEIGHBORS))
		cell_neighbors_.resize(num_cells * static_cast<size_t>(MAX_NEIGHBORS));
	if (m != cell_neighbors_bits_ || per_mask != cell_neighbors_permask_) {
		BuildCellNeighborsKernel nbr_table{cell_neighbors_.data(), num_cells, m, box_len_};
		launch_kernel(resource_, KernelConfig::for_1d(num_cells, resource_), nbr_table).wait();
		cell_neighbors_bits_ = m;
		cell_neighbors_permask_ = per_mask;
	}

	const int shift = 3 * (max_bits - m);

	KernelConfig config = KernelConfig::for_1d(num_particles, resource_);

	BuildCellRangesKernel range_kernel{sorter_.get_morton_codes().data(),
									   cell_begin_.data(),
									   cell_end_.data(),
									   num_particles,
									   shift};
	launch_kernel(resource_, config, range_kernel).wait();

	PeriodicBox box(box_len_, box_len_.x > 0.0f, box_len_.y > 0.0f, box_len_.z > 0.0f);
	box.set_origin(box_origin_);

	ZOrderCellNeighborKernel kernel{sorted_positions_.data(),
									sorter_.get_morton_codes().data(),
									sorter_.get_sorted_indices().data(),
									cell_begin_.data(),
									cell_end_.data(),
									cell_neighbors_.data(),
									neighbor_pairs_.data(),
									pair_count_.data(),
									cutoff_squared_,
									num_particles,
									max_pairs_,
									shift,
									box};

	Event launch_event = launch_kernel(resource_, config, kernel);
	launch_event.wait();

	// Overflow is fatal: kept pairs are a nondeterministic subset. See dev_notes.md.
	uint32_t num_pairs;
	pair_count_.copy_to_host(&num_pairs, 1, true);
	if (num_pairs > max_pairs_) {
		ARBD_Exception(ExceptionType::ValueError,
					   "Pairlist capacity exceeded: %u pairs found for %zu particles (%.1f per "
					   "particle) but the device-memory budget holds only %u. Raise GPU_MEM in "
					   "CMake or shorten the pairlist cutoff; continuing would silently simulate "
					   "a different system.",
					   num_pairs,
					   num_particles,
					   num_particles ? static_cast<double>(num_pairs) / num_particles : 0.0,
					   max_pairs_);
	}
	num_pairs_ = num_pairs;
	LOGDEBUG("pair_count AFTER kernel: {}", num_pairs);
}

void ZOrderPairlist::get_bounding_box(const DeviceBuffer<Vector3>& positions,
									  size_t num_particles,
									  Vector3& box_min,
									  Vector3& box_max) const {
	if (auto_bbox_) {
		// Use persistent buffers to avoid recreation
		box_min = Vector3(std::numeric_limits<float>::max());
		box_max = Vector3(std::numeric_limits<float>::lowest());

		persistent_bbox_min_.copy_from_host(&box_min, 1);
		persistent_bbox_max_.copy_from_host(&box_max, 1);

		BoundingBoxKernel bbox_kernel{positions.data(),
									  persistent_bbox_min_.data(),
									  persistent_bbox_max_.data(),
									  num_particles};

		KernelConfig config = KernelConfig::for_1d(num_particles, resource_);
		Event evt = launch_kernel(resource_, config, bbox_kernel);
		evt.wait();
		persistent_bbox_min_.copy_to_host(&box_min, 1, true);
		persistent_bbox_max_.copy_to_host(&box_max, 1, true);

		// Add small margin to avoid boundary issues
		Vector3 margin = (box_max - box_min) * 0.01f;
		box_min -= margin;
		box_max += margin;
		constexpr arbd_real MIN_EXTENT = 1e-4;
		Vector3 range = box_max - box_min;

		if (range.x < MIN_EXTENT) {
			arbd_real center = box_min.x + range.x * 0.5;
			box_min.x = center - MIN_EXTENT * 0.5;
			box_max.x = center + MIN_EXTENT * 0.5;
		}
		if (range.y < MIN_EXTENT) {
			arbd_real center = box_min.y + range.y * 0.5;
			box_min.y = center - MIN_EXTENT * 0.5;
			box_max.y = center + MIN_EXTENT * 0.5;
		}
		if (range.z < MIN_EXTENT) {
			arbd_real center = box_min.z + range.z * 0.5;
			box_min.z = center - MIN_EXTENT * 0.5;
			box_max.z = center + MIN_EXTENT * 0.5;
		}
	} else {
		// Use manually specified bounds
		box_min = manual_box_min_;
		box_max = manual_box_max_;
	}
}

} // namespace ARBD
