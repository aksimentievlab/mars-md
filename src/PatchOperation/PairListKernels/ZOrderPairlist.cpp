#include "ZOrderPairlist.h"
#include "MARSException.h"
#include "MARSLogger.h"

#include "ZOrderNeighbor.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <limits>

namespace MARS {

namespace {

constexpr int kMortonBits = []() constexpr {
	int b = 0;
	for (auto v = MortonCode::max_coord_device; v; v >>= 1)
		++b;
	return b;
}();

int env_int(const char* name, int fallback) {
	const char* raw = std::getenv(name);
	if (!raw || !*raw)
		return fallback;
	char* end = nullptr;
	const long v = std::strtol(raw, &end, 10);
	return (end && *end == '\0') ? static_cast<int>(v) : fallback;
}

} // namespace

void ZOrderPairlist::select_coarse_grid(float cutoff, size_t num_particles) {
	coarse_bits_ = 0;
	cell_radii_ = int3(1, 1, 1);
	neighbors_per_cell_ = 1;

	const double extent[3] = {static_cast<double>(last_box_extent_.x),
							  static_cast<double>(last_box_extent_.y),
							  static_cast<double>(last_box_extent_.z)};
	if (cutoff <= 0.0f || extent[0] <= 0.0 || extent[1] <= 0.0 || extent[2] <= 0.0 ||
		num_particles == 0)
		return;

	const size_t byte_cap =
		static_cast<size_t>(std::max(
			1, env_int("MARS_ZORDER_TABLE_MB", static_cast<int>(kCellNeighborBytesCap >> 20))))
		<< 20;
	const int forced_bits = env_int("MARS_ZORDER_BITS", -1);
	const double density = static_cast<double>(num_particles) / (extent[0] * extent[1] * extent[2]);

	int best_m = 0;
	int best_r[3] = {1, 1, 1};
	long long best_slots = 1;
	double best_cost = 0.0;
	bool have_best = false;

	for (int m = 0; m <= std::min(kMortonBits, kMaxCoarseBits); ++m) {
		const long long n = 1LL << m;
		const size_t num_cells = static_cast<size_t>(1) << (3 * m);

		int r[3];
		long long span[3];
		double volume = 1.0;
		for (int a = 0; a < 3; ++a) {
			const double side = extent[a] / static_cast<double>(n);
			r[a] = static_cast<int>(std::ceil(static_cast<double>(cutoff) / side));
			span[a] = std::min(2LL * r[a] + 1LL, n);
			volume *= static_cast<double>(span[a]) * side;
		}
		const long long slots = span[0] * span[1] * span[2];

		const size_t bytes =
			num_cells * (static_cast<size_t>(slots) * sizeof(uint32_t) + 2 * sizeof(uint32_t));
		const bool over_cap = (m > 0 && bytes > byte_cap);

		const double cost = static_cast<double>(slots) + volume * density;
		const bool take = (forced_bits >= 0) ? (m == forced_bits)
											 : (!over_cap && (!have_best || cost < best_cost));
		if (take) {
			have_best = true;
			best_cost = cost;
			best_m = m;
			best_r[0] = r[0];
			best_r[1] = r[1];
			best_r[2] = r[2];
			best_slots = slots;
		}
		if (forced_bits >= 0 ? (m >= forced_bits) : over_cap)
			break;
	}

	coarse_bits_ = best_m;
	cell_radii_ = int3(best_r[0], best_r[1], best_r[2]);
	neighbors_per_cell_ = static_cast<int>(best_slots);
}

ZOrderPairlist::ZOrderPairlist(const Resource& resource, size_t max_particles, size_t max_pairs)
	: Pairlist(resource, max_particles, max_pairs),
	  sorter_(resource, max_particles, ZOrderOptimizationMode::Pairlist), // Use Pairlist mode
	  sorted_positions_(max_particles, resource), persistent_bbox_min_(1, resource),
	  persistent_bbox_max_(1, resource), cell_begin_(kInitialCoarseCells, resource),
	  cell_end_(kInitialCoarseCells, resource),
	  cell_neighbors_(kInitialCoarseCells * MAX_NEIGHBORS, resource), last_build_time_ms_(0.0),
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
		MARS_Exception(ExceptionType::ValueError,
					   "Cannot build pairlist for {} particles, maximum is {}",
					   num_particles,
					   max_particles_);
	}

	LOGTRACE("Building Z-order pairlist for {} particles with cutoff {}",
			 num_particles,
			 pairlist_cutoff);

	const Vector3& box_size = box_.get_box_size();
	if (box_size.x <= 0.0f || box_size.y <= 0.0f || box_size.z <= 0.0f) {
		MARS_Exception(ExceptionType::ValueError,
					   "ZOrderPairlist has no simulation box (size %.3f x %.3f x %.3f). Call "
					   "set_periodic_box() before building.",
					   box_size.x,
					   box_size.y,
					   box_size.z);
	}

	// Periodic axes must encode against the simulation box, since wrapping is
	// defined against it. Open axes use the particle extent, which is tighter and
	// packs the coarse cells better. See dev_notes.md.
	Vector3 box_min, box_max;
	compute_particle_extent(positions, num_particles, box_min, box_max);
	const Vector3& origin = box_.get_origin();
	if (box_.is_periodic(0)) {
		box_min.x = origin.x;
		box_max.x = origin.x + box_size.x;
	}
	if (box_.is_periodic(1)) {
		box_min.y = origin.y;
		box_max.y = origin.y + box_size.y;
	}
	if (box_.is_periodic(2)) {
		box_min.z = origin.z;
		box_max.z = origin.z + box_size.z;
	}
	last_box_extent_ = box_max - box_min;

	LOGTRACE("Morton domain: [{:.6f}, {:.6f}, {:.6f}] to [{:.6f}, {:.6f}, {:.6f}]",
			 box_min.x,
			 box_min.y,
			 box_min.z,
			 box_max.x,
			 box_max.y,
			 box_max.z);
	update_state(num_particles, pairlist_cutoff);
	sorter_.sort_particles(positions, num_particles, box_min, box_max);
	LOGTRACE("Sorted particles by Morton code");
	resource_.synchronize_streams();
	LOGDEBUG("PAIRLIST DIAG: sort errors = {}", sorter_.validate_sorting());
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
	float max_disp = sorter_.compute_max_displacement(positions, num_particles, periodic_lengths());
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

	// Cells are sized against the encoded extent, not the box. See dev_notes.md.
	const float cutoff = std::sqrt(cutoff_squared_);
	constexpr int max_bits = kMortonBits;
	select_coarse_grid(cutoff, num_particles);
	const int m = coarse_bits_;

	const size_t num_cells = static_cast<size_t>(1) << (3 * m);
	if (cell_begin_.size() < num_cells) {
		cell_begin_.resize(num_cells);
		cell_end_.resize(num_cells);
	}
	cell_begin_.fill(0u, true);
	cell_end_.fill(0u, true);
	// Buffers fill on the Memory stream, kernels run on Compute, and fill() only
	// self-synchronizes under SYCL. Without this the range kernels read stale
	// cell bounds. See dev_notes.md.
	resource_.synchronize_streams();

	// Per-cell neighbor table: topology depends only on the grid, so rebuild it only
	// when bits/radii/periodicity change (~once, at patch init). See dev_notes.md.
	const Vector3 per_len = periodic_lengths();
	const int per_mask =
		(per_len.x > 0.0f ? 1 : 0) | (per_len.y > 0.0f ? 2 : 0) | (per_len.z > 0.0f ? 4 : 0);
	const size_t table_size = num_cells * static_cast<size_t>(neighbors_per_cell_);
	if (cell_neighbors_.size() < table_size)
		cell_neighbors_.resize(table_size);
	if (m != cell_neighbors_bits_ || per_mask != cell_neighbors_permask_ ||
		cell_radii_.x != cell_neighbors_radii_.x || cell_radii_.y != cell_neighbors_radii_.y ||
		cell_radii_.z != cell_neighbors_radii_.z) {
		BuildCellNeighborsKernel nbr_table{cell_neighbors_.data(),
										   num_cells,
										   m,
										   cell_radii_,
										   neighbors_per_cell_,
										   per_len};
		launch_kernel(resource_, KernelConfig::for_1d(num_cells, resource_), nbr_table).wait();
		cell_neighbors_bits_ = m;
		cell_neighbors_permask_ = per_mask;
		cell_neighbors_radii_ = cell_radii_;
	}

	const int shift = 3 * (max_bits - m);

	KernelConfig config = KernelConfig::for_1d(num_particles, resource_);

	BuildCellRangesKernel range_kernel{sorter_.get_morton_codes().data(),
									   cell_begin_.data(),
									   cell_end_.data(),
									   num_particles,
									   shift};
	launch_kernel(resource_, config, range_kernel).wait();

	const PeriodicBox& box = box_;

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
									neighbors_per_cell_,
									shift,
									box,
									exclusions_};

	Event launch_event = launch_kernel(resource_, config, kernel);
	launch_event.wait();

	// Overflow is fatal: kept pairs are a nondeterministic subset. See dev_notes.md.
	uint32_t num_pairs;
	pair_count_.copy_to_host(&num_pairs, 1, true);
	if (num_pairs > max_pairs_) {
		MARS_Exception(ExceptionType::ValueError,
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
	LOGDEBUG("PLDIAG n={} cut={:.1f} m={} cells={} stencil=({},{},{})x{} "
			 "cell=({:.1f},{:.1f},{:.1f}) extent=({:.1f},{:.1f},{:.1f}) "
			 "per=({},{},{}) boxsz=({:.1f},{:.1f},{:.1f}) org=({:.1f},{:.1f},{:.1f}) pairs={}",
			 num_particles,
			 std::sqrt(cutoff_squared_),
			 m,
			 num_cells,
			 cell_radii_.x,
			 cell_radii_.y,
			 cell_radii_.z,
			 neighbors_per_cell_,
			 last_box_extent_.x / (1 << m),
			 last_box_extent_.y / (1 << m),
			 last_box_extent_.z / (1 << m),
			 last_box_extent_.x,
			 last_box_extent_.y,
			 last_box_extent_.z,
			 box_.is_periodic(0),
			 box_.is_periodic(1),
			 box_.is_periodic(2),
			 box_.get_box_size().x,
			 box_.get_box_size().y,
			 box_.get_box_size().z,
			 box_.get_origin().x,
			 box_.get_origin().y,
			 box_.get_origin().z,
			 num_pairs);
}

void ZOrderPairlist::compute_particle_extent(const DeviceBuffer<Vector3>& positions,
											 size_t num_particles,
											 Vector3& box_min,
											 Vector3& box_max) const {
	box_min = Vector3(std::numeric_limits<float>::max());
	box_max = Vector3(std::numeric_limits<float>::lowest());

	persistent_bbox_min_.copy_from_host(&box_min, 1);
	persistent_bbox_max_.copy_from_host(&box_max, 1);

	BoundingBoxKernel bbox_kernel{positions.data(),
								  persistent_bbox_min_.data(),
								  persistent_bbox_max_.data(),
								  num_particles};

	KernelConfig config = KernelConfig::for_1d(num_particles, resource_);
	launch_kernel(resource_, config, bbox_kernel).wait();
	persistent_bbox_min_.copy_to_host(&box_min, 1, true);
	persistent_bbox_max_.copy_to_host(&box_max, 1, true);

	// Margin keeps particles off the domain boundary; MIN_EXTENT keeps a
	// degenerate axis (a line or plane of particles) from collapsing the encode.
	const Vector3 margin = (box_max - box_min) * 0.01f;
	box_min -= margin;
	box_max += margin;
	constexpr mars_real MIN_EXTENT = 1e-4;
	const Vector3 range = box_max - box_min;

	if (range.x < MIN_EXTENT) {
		const mars_real center = box_min.x + range.x * 0.5;
		box_min.x = center - MIN_EXTENT * 0.5;
		box_max.x = center + MIN_EXTENT * 0.5;
	}
	if (range.y < MIN_EXTENT) {
		const mars_real center = box_min.y + range.y * 0.5;
		box_min.y = center - MIN_EXTENT * 0.5;
		box_max.y = center + MIN_EXTENT * 0.5;
	}
	if (range.z < MIN_EXTENT) {
		const mars_real center = box_min.z + range.z * 0.5;
		box_min.z = center - MIN_EXTENT * 0.5;
		box_max.z = center + MIN_EXTENT * 0.5;
	}
}

} // namespace MARS
