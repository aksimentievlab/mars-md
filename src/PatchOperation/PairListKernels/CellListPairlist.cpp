#include "CellListPairlist.h"
#include "ARBDException.h"
#include "ARBDLogger.h"
#include "Backend/Kernels.h"
#include <chrono>

#ifdef USE_CUDA
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <thrust/system/cuda/execution_policy.h>
#endif

namespace ARBD {

CellListPairlist::CellListPairlist(const Resource& resource,
								   size_t max_particles,
								   size_t max_pairs,
								   int num_replicas)
	: Pairlist(resource, max_particles, max_pairs), num_replicas_(num_replicas), origin_(0.0f),
	  box_size_(1.0f), n_cells_(1, 1, 1), cell_size_(1.0f), total_cells_(1),
	  cells_(max_particles * num_replicas, resource), cell_ranges_(max_particles, resource),
	  temp_ranges_(max_particles, resource), last_build_time_ms_(0.0), cells_used_(0),
	  average_particles_per_cell_(0.0f) {

	LOGINFO("Created CellListPairlist with capacity {} particles, {} pairs, {} replicas on {}",
			max_particles_,
			max_pairs_,
			num_replicas_,
			resource_.toString());
}

void CellListPairlist::build_pairlist(const DeviceBuffer<Vector3>& positions,
									  size_t num_particles,
									  float cutoff) {
	auto start_time = std::chrono::high_resolution_clock::now();

	if (num_particles > max_particles_) {
		ARBD_Exception(ExceptionType::ValueError,
					   "Cannot build pairlist for {} particles, maximum is {}",
					   num_particles,
					   max_particles_);
	}

	LOGTRACE("Building cell list pairlist for {} particles with cutoff {}", num_particles, cutoff);

	// Step 1: Compute optimal cell grid
	compute_cell_grid(cutoff);

	// Step 2: Decompose particles into cells
	decompose_particles(positions, num_particles);

	// Step 3: Sort particles by cell ID
	sort_particles_by_cell();

	// Step 4: Create cell ranges
	create_cell_ranges();

	// Step 5: Find neighbors using cell decomposition
	reset_pair_count();
	find_neighbors_celllist(positions, num_particles);

	// Step 6: Update internal state
	update_state(num_particles, cutoff);

	auto end_time = std::chrono::high_resolution_clock::now();
	last_build_time_ms_ = std::chrono::duration<double, std::milli>(end_time - start_time).count();

	LOGTRACE("Cell list pairlist built: {} pairs in {:.2f} ms", num_pairs_, last_build_time_ms_);
}

void CellListPairlist::update_pairlist(const DeviceBuffer<Vector3>& positions,
									   size_t num_particles) {
	// For now, always rebuild
	// Future optimization: check if particles have moved significantly
	build_pairlist(positions, num_particles, cutoff_);
}

bool CellListPairlist::needs_update(const DeviceBuffer<Vector3>& positions,
									const DeviceBuffer<Vector3>& old_positions,
									size_t num_particles,
									float skin_distance) const {
	// Simple criterion: maximum displacement > skin_distance/2
	// More sophisticated implementations could track average displacement
	// For now, always return true (conservative approach)
	return true;
}

void CellListPairlist::resize(size_t new_max_particles, size_t new_max_pairs) {
	Pairlist::resize(new_max_particles, new_max_pairs);

	cells_.resize(new_max_particles * num_replicas_);
	cell_ranges_.resize(new_max_particles);
	temp_ranges_.resize(new_max_particles);

	LOGINFO("Resized CellListPairlist to {} particles, {} pairs", new_max_particles, new_max_pairs);
}

Pairlist::Statistics CellListPairlist::get_statistics() const {
	Statistics stats = Pairlist::get_statistics();
	stats.build_time_ms = last_build_time_ms_;
	return stats;
}

void CellListPairlist::compute_cell_grid(float cutoff) {
	// Set cell size to cutoff distance for optimal neighbor finding
	cell_size_ = cutoff;

	// Compute number of cells in each dimension
	n_cells_.x = std::max(1, int(floorf(box_size_.x / cell_size_)));
	n_cells_.y = std::max(1, int(floorf(box_size_.y / cell_size_)));
	n_cells_.z = std::max(1, int(floorf(box_size_.z / cell_size_)));

	total_cells_ = n_cells_.x * n_cells_.y * n_cells_.z;

	// Ensure we have enough space for cell ranges
	if (total_cells_ * num_replicas_ > cell_ranges_.size()) {
		cell_ranges_.resize(total_cells_ * num_replicas_);
		temp_ranges_.resize(total_cells_ * num_replicas_);
	}

	LOGTRACE("Cell grid: {}x{}x{} = {} cells, cell size = {:.3f}",
			 n_cells_.x,
			 n_cells_.y,
			 n_cells_.z,
			 total_cells_,
			 cell_size_);
}

void CellListPairlist::decompose_particles(const DeviceBuffer<Vector3>& positions,
										   size_t num_particles) {
	DecomposeKernel kernel{origin_, cell_size_, n_cells_, num_replicas_};

	KernelConfig config = KernelConfig::for_1d(num_particles * num_replicas_, resource_);
	launch_kernel(resource_, config, kernel, positions.data(), cells_.data());
}

void CellListPairlist::sort_particles_by_cell() {
#ifdef USE_CUDA
	// Use Thrust to sort particles by cell ID
	// Use raw pointer instead of device_ptr to avoid compatibility issues
	DecomposeKernel::cell_t* cells_raw = cells_.data();
	size_t total_elements = num_particles_ * num_replicas_;
	cudaStream_t stream = resource_.get_cuda_stream();
	thrust::cuda::par.on(stream).sort(
		cells_raw,
		cells_raw + total_elements,
		[](const DecomposeKernel::cell_t& a, const DecomposeKernel::cell_t& b) {
			if (a.id != b.id)
				return a.id < b.id;
			return a.repID < b.repID;
		});
#else
	ARBD_Exception(ExceptionType::NotImplementedError,
				   "Cell sorting not implemented for non-CUDA backends");
#endif
}

void CellListPairlist::create_cell_ranges() {
	// Step 1: Create range markers
	MakeRangesKernel make_kernel{static_cast<int>(total_cells_), num_replicas_};

	size_t total_elements = num_particles_ * num_replicas_;
	KernelConfig config1 = KernelConfig::for_1d(total_elements, resource_);
	launch_kernel(resource_,
				  config1,
				  make_kernel,
				  cells_.data(),
				  temp_ranges_.data(),
				  total_elements);

	// Step 2: Bind ranges from temporary array
	BindRangesKernel bind_kernel{static_cast<int>(total_cells_), num_replicas_};

	KernelConfig config2 = KernelConfig::for_1d(total_cells_ * num_replicas_, resource_);
	launch_kernel(resource_,
				  config2,
				  bind_kernel,
				  cell_ranges_.data(),
				  temp_ranges_.data(),
				  cells_.data(),
				  total_elements);
}

void CellListPairlist::find_neighbors_celllist(const DeviceBuffer<Vector3>& positions,
											   size_t num_particles) {
	CellNeighborKernel kernel{positions.data(),
							  cells_.data(),
							  cell_ranges_.data(),
							  neighbor_pairs_.data(),
							  pair_count_.data(),
							  cutoff_squared_,
							  origin_,
							  cell_size_,
							  n_cells_,
							  num_particles,
							  max_pairs_,
							  num_replicas_};

	KernelConfig config = KernelConfig::for_1d(num_particles * num_replicas_, resource_);
	launch_kernel(resource_, config, kernel);
}

} // namespace ARBD
