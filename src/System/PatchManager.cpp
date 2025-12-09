// src/System/PatchManager.cpp
#include "System/PatchManager.h"
#include <stdexcept>

#ifdef USE_MPI
#include "Backend/MPIManager.h"
#endif

namespace ARBD {

// Constructor
PatchManager::PatchManager(SimSystem& sys)
#ifdef USE_MPI
	: sys_(sys), mpi_manager_(MPI::Manager::instance()), global_seed_(0), max_ghost_particles_(0)
#else
	: sys_(sys), global_seed_(0), max_ghost_particles_(0)
#endif
{
}

// Initialize patch manager from decomposition plan
void PatchManager::initialize_from_plan(const DecompositionPlan& plan,
										idx_t estimated_particles_per_patch,
										const Length& cutoff) {
	if (!plan.is_valid()) {
		throw std::runtime_error("Invalid decomposition plan provided to PatchManager");
	}

#ifdef USE_MPI
	if (!mpi_manager_.is_initialized()) {
		throw std::runtime_error("MPI Manager must be initialized before PatchManager");
	}

	int mpi_size = mpi_manager_.get_size();
	size_t total_patches = plan.total_patches();

	if (static_cast<size_t>(mpi_size) != total_patches) {
		throw std::runtime_error("MPI size (" + std::to_string(mpi_size) +
								 ") must match total patches (" + std::to_string(total_patches) +
								 ")");
	}
#endif

	// Setup spatial grid from plan
	setup_spatial_grid(plan.grid_dimensions[0],
					   plan.grid_dimensions[1],
					   plan.grid_dimensions[2],
					   plan.periodicity[0],
					   plan.periodicity[1],
					   plan.periodicity[2]);

	// Initialize local patch using the decomposition plan
	initialize_local_patch(estimated_particles_per_patch, cutoff);

#ifdef USE_MPI
	// Set local patch bounds from plan
	int my_rank = mpi_manager_.get_rank();
	if (my_rank >= 0 && static_cast<size_t>(my_rank) < plan.patch_min_bounds.size()) {
		update_local_bounds(plan.patch_min_bounds[my_rank], plan.patch_max_bounds[my_rank]);

		// Set resource for local patch from plan
		if (static_cast<size_t>(my_rank) < plan.patch_resources.size()) {
			local_metadata_.resource = plan.patch_resources[my_rank];
			if (!patches_.empty()) {
				patches_[0].set_resource(plan.patch_resources[my_rank]);
			}
		}
	}
#else
	// Non-MPI: initialize all patches from plan
	patches_.clear();
	patches_.reserve(plan.total_patches());

	for (size_t idx = 0; idx < plan.total_patches(); ++idx) {
		Patch patch(static_cast<patch_t>(idx), estimated_particles_per_patch);
		patch.set_bounds(plan.patch_min_bounds[idx], plan.patch_max_bounds[idx]);
		patch.set_resource(plan.patch_resources[idx]);
		patches_.push_back(std::move(patch));
	}

	// For non-MPI, local metadata refers to first patch (if exists)
	if (!patches_.empty()) {
		local_metadata_.min = patches_[0].get_bounds_min();
		local_metadata_.max = patches_[0].get_bounds_max();
		local_metadata_.resource = patches_[0].get_resource();
	}
#endif

	// Exchange metadata to populate all_metadata_
	exchange_metadata();
}

// Initialize patch manager with MPI integration (deprecated)
void PatchManager::initialize(int grid_nx,
							  int grid_ny,
							  int grid_nz,
							  bool periodic_x,
							  bool periodic_y,
							  bool periodic_z) {
#ifdef USE_MPI
	if (!mpi_manager_.is_initialized()) {
		throw std::runtime_error("MPI Manager must be initialized before PatchManager");
	}

	int mpi_size = mpi_manager_.get_size();
	int grid_size = grid_nx * grid_ny * grid_nz;

	if (mpi_size != grid_size) {
		throw std::runtime_error("MPI size (" + std::to_string(mpi_size) +
								 ") must match grid size (" + std::to_string(grid_size) + ")");
	}
#endif

	setup_spatial_grid(grid_nx, grid_ny, grid_nz, periodic_x, periodic_y, periodic_z);
}

// Setup spatial grid and determine neighbors
void PatchManager::setup_spatial_grid(int nx, int ny, int nz, bool px, bool py, bool pz) {
	grid_info_.dimensions = {nx, ny, nz};
	grid_info_.periodic = {px, py, pz};

#ifdef USE_MPI
	int my_rank = mpi_manager_.get_rank();
	auto my_coords = rank_to_grid_coords(my_rank);

	local_metadata_.grid_coords = {my_coords[0], my_coords[1], my_coords[2]};
	local_metadata_.mpi_rank = my_rank;
#endif

	discover_neighbors();
}

// Initialize local patch for this MPI rank
void PatchManager::initialize_local_patch(idx_t estimated_particles, const Length& cutoff) {
#ifdef USE_MPI
	// Create local patch
	Patch local_patch(static_cast<patch_t>(mpi_manager_.get_rank()), estimated_particles);
	local_patch.set_bounds(local_metadata_.min, local_metadata_.max);
	local_patch.set_resource(local_metadata_.resource);

	patches_.clear();
	patches_.push_back(std::move(local_patch));

	// Update metadata
	local_metadata_.capacity = estimated_particles;
	local_metadata_.num = 0;

	// Estimate max ghost particles (worst case: all particles in boundary region)
	// Assuming cutoff-based halo
	float cutoff_val = cutoff;
	float patch_volume = (local_metadata_.max.x - local_metadata_.min.x) *
						 (local_metadata_.max.y - local_metadata_.min.y) *
						 (local_metadata_.max.z - local_metadata_.min.z);
	float halo_volume = 2.0f * cutoff_val *
						((local_metadata_.max.x - local_metadata_.min.x) *
							 (local_metadata_.max.y - local_metadata_.min.y) +
						 (local_metadata_.max.x - local_metadata_.min.x) *
							 (local_metadata_.max.z - local_metadata_.min.z) +
						 (local_metadata_.max.y - local_metadata_.min.y) *
							 (local_metadata_.max.z - local_metadata_.min.z));
	float density = static_cast<float>(estimated_particles) / patch_volume;
	max_ghost_particles_ = static_cast<idx_t>(halo_volume * density * 1.5f); // 1.5x safety factor
#else
	// Non-MPI case - create single patch
	Patch local_patch(0, estimated_particles);
	local_patch.set_bounds(local_metadata_.min, local_metadata_.max);
	patches_.clear();
	patches_.push_back(std::move(local_patch));
	local_metadata_.capacity = estimated_particles;
	local_metadata_.num = 0;
	max_ghost_particles_ = estimated_particles / 10; // Conservative estimate
#endif
}

// Find and setup neighbor relationships
void PatchManager::discover_neighbors() {
#ifdef USE_MPI
	int my_rank = mpi_manager_.get_rank();
	auto my_coords = rank_to_grid_coords(my_rank);

	// Reset neighbors
	for (auto& neighbor : neighbors_) {
		neighbor.active = false;
	}

	// Only support 2D grid for now (as noted in header: just 2 neighbors in 2D grid)
	// Check left and right neighbors (x-direction)
	for (int dir = 0; dir < 2; ++dir) {
		std::array<int, 3> neighbor_coords = my_coords;
		neighbor_coords[0] += (dir == 0) ? -1 : +1;

		// Handle periodic boundaries
		if (grid_info_.periodic[0]) {
			if (neighbor_coords[0] < 0) {
				neighbor_coords[0] = grid_info_.dimensions[0] - 1;
			} else if (neighbor_coords[0] >= grid_info_.dimensions[0]) {
				neighbor_coords[0] = 0;
			}
		}

		// Check if neighbor is valid
		if (neighbor_coords[0] >= 0 && neighbor_coords[0] < grid_info_.dimensions[0]) {
			int neighbor_rank =
				grid_coords_to_rank(neighbor_coords[0], neighbor_coords[1], neighbor_coords[2]);
			if (neighbor_rank >= 0 && neighbor_rank != my_rank) {
				neighbors_[dir].neighbor_rank = neighbor_rank;
				neighbors_[dir].direction = {(dir == 0) ? -1 : 1, 0, 0};
				neighbors_[dir].active = true;
			}
		}
	}

	// Initialize neighbor communication buffers
	const Resource& resource = local_metadata_.resource;
	for (int dir = 0; dir < 2; ++dir) {
		if (neighbors_[dir].active) {
			int saved_rank = neighbors_[dir].neighbor_rank;
			std::array<int, 3> saved_dir = neighbors_[dir].direction;
			neighbors_[dir] = NeighborComm(resource, max_ghost_particles_);
			neighbors_[dir].neighbor_rank = saved_rank;
			neighbors_[dir].direction = saved_dir;
			neighbors_[dir].active = true;
		}
	}
#else
	// Non-MPI: no neighbors
	for (auto& neighbor : neighbors_) {
		neighbor.active = false;
	}
#endif
}

// Exchange metadata with all ranks
void PatchManager::exchange_metadata() {
#ifdef USE_MPI
	int mpi_size = mpi_manager_.get_size();
	all_metadata_.resize(mpi_size);

	// Serialize metadata for exchange
	// Pack: mpi_rank, num, capacity, min (3 floats), max (3 floats)
	constexpr int metadata_size = 1 + 1 + 1 + 3 + 3; // int + idx_t + idx_t + 3 floats + 3 floats
	DeviceBuffer<float> send_buffer(metadata_size, local_metadata_.resource);
	DeviceBuffer<float> recv_buffer(metadata_size * mpi_size, local_metadata_.resource);

	// Pack local metadata
	std::vector<float> send_host(metadata_size);
	send_host[0] = static_cast<float>(local_metadata_.mpi_rank);
	send_host[1] = static_cast<float>(local_metadata_.num);
	send_host[2] = static_cast<float>(local_metadata_.capacity);
	send_host[3] = local_metadata_.min.x;
	send_host[4] = local_metadata_.min.y;
	send_host[5] = local_metadata_.min.z;
	send_host[6] = local_metadata_.max.x;
	send_host[7] = local_metadata_.max.y;
	send_host[8] = local_metadata_.max.z;

	send_buffer.copy_from_host(send_host.data(), metadata_size);
	mpi_manager_.allGather(send_buffer, recv_buffer, metadata_size, local_metadata_.resource);

	// Unpack received metadata
	std::vector<float> recv_host(metadata_size * mpi_size);
	recv_buffer.copy_to_host(recv_host.data(), metadata_size * mpi_size);

	for (int i = 0; i < mpi_size; ++i) {
		int offset = i * metadata_size;
		all_metadata_[i].mpi_rank = static_cast<int>(recv_host[offset]);
		all_metadata_[i].num = static_cast<idx_t>(recv_host[offset + 1]);
		all_metadata_[i].capacity = static_cast<idx_t>(recv_host[offset + 2]);
		all_metadata_[i].min.x = recv_host[offset + 3];
		all_metadata_[i].min.y = recv_host[offset + 4];
		all_metadata_[i].min.z = recv_host[offset + 5];
		all_metadata_[i].max.x = recv_host[offset + 6];
		all_metadata_[i].max.y = recv_host[offset + 7];
		all_metadata_[i].max.z = recv_host[offset + 8];
	}
#else
	// Non-MPI: just store local metadata
	all_metadata_.clear();
	all_metadata_.push_back(local_metadata_);
#endif
}

// Update local patch bounds
void PatchManager::update_local_bounds(const Vector3& new_min, const Vector3& new_max) {
	local_metadata_.min = new_min;
	local_metadata_.max = new_max;

	if (!patches_.empty()) {
		patches_[0].set_bounds(new_min, new_max);
	}
}

// Exchange halo particles with all neighbors
std::vector<Event> PatchManager::exchange_halo_particles(DeviceBuffer<float>& positions,
														 DeviceBuffer<float>& velocities,
														 const Length& cutoff) {
#ifdef USE_MPI
	std::vector<Event> events;
	const Resource& resource = local_metadata_.resource;

	// Prepare send and receive lists
	std::vector<std::pair<DeviceBuffer<float>*, int>> send_list;
	std::vector<std::pair<DeviceBuffer<float>*, int>> recv_list;

	std::vector<idx_t> send_counts(2, 0);

	// For each active neighbor, pack and send boundary particles
	for (int dir = 0; dir < 2; ++dir) {
		if (!has_neighbor(dir)) {
			continue;
		}

		int neighbor_rank = get_neighbor_rank(dir);

		// Pack boundary particles for this direction from local patch
		Patch& local_patch = get_local_patch();
		idx_t packed_count =
			local_patch.pack_boundary_particles(neighbors_[dir].send_buffer, dir, cutoff);
		send_counts[dir] = packed_count;

		if (packed_count > 0) {
			// Send count first (using separate communication for metadata)
			// Then send actual data
			send_list.push_back({&neighbors_[dir].send_buffer, neighbor_rank});
		}
	}

	// Post receives for all neighbors
	for (int dir = 0; dir < 2; ++dir) {
		if (!has_neighbor(dir)) {
			continue;
		}

		int neighbor_rank = get_neighbor_rank(dir);
		recv_list.push_back({&neighbors_[dir].recv_buffer, neighbor_rank});
	}

	// Exchange particle counts first (for proper buffer allocation)

	// Exchange counts with neighbors using point-to-point communication
	for (int dir = 0; dir < 2; ++dir) {
		if (!has_neighbor(dir)) {
			continue;
		}

		int neighbor_rank = get_neighbor_rank(dir);
		int opposite_dir = (dir == 0) ? 1 : 0; // Opposite direction

		// Send count
		DeviceBuffer<int> send_single(1, resource);
		DeviceBuffer<int> recv_single(1, resource);
		std::vector<int> send_val = {static_cast<int>(send_counts[dir])};
		send_single.copy_from_host(send_val.data(), 1);

		events.push_back(mpi_manager_.send_boundary(send_single,
													1,
													neighbor_rank,
													METADATA_TAG + dir,
													resource));
		events.push_back(mpi_manager_.recv_boundary(recv_single,
													1,
													neighbor_rank,
													METADATA_TAG + opposite_dir,
													resource));

		// Wait for count exchange to complete
		events.back().wait();

		// Get received count
		std::vector<int> recv_val(1);
		recv_single.copy_to_host(recv_val.data(), 1);
		neighbors_[opposite_dir].halo_size = static_cast<idx_t>(recv_val[0]);
	}

	// Exchange actual particle data using MPI Manager
	if (!send_list.empty() || !recv_list.empty()) {
		// For positions (6 floats per particle: 3 pos + 3 vel)
		for (size_t i = 0; i < send_list.size(); ++i) {
			int dir = static_cast<int>(i);
			int neighbor_rank = get_neighbor_rank(dir);
			idx_t count = send_counts[dir] * 6; // 6 floats per particle

			if (count > 0) {
				events.push_back(mpi_manager_.send_boundary(*send_list[i].first,
															count,
															neighbor_rank,
															POSITIONS_TAG + dir,
															resource));
			}
		}

		for (size_t i = 0; i < recv_list.size(); ++i) {
			int dir = static_cast<int>(i);
			int opposite_dir = (dir == 0) ? 1 : 0; // Opposite direction
			int neighbor_rank = get_neighbor_rank(dir);
			idx_t count = neighbors_[dir].halo_size * 6; // 6 floats per particle

			if (count > 0) {
				events.push_back(mpi_manager_.recv_boundary(*recv_list[i].first,
															count,
															neighbor_rank,
															POSITIONS_TAG + opposite_dir,
															resource));
			}
		}
	}

	return events;
#else
	// Non-MPI: return empty events
	return std::vector<Event>();
#endif
}

// Send boundary particles to specific neighbor
Event PatchManager::send_boundary_particles(DeviceBuffer<float>& particles,
											idx_t count,
											int neighbor_direction,
											int tag) {
#ifdef USE_MPI
	if (!has_neighbor(neighbor_direction)) {
		throw std::invalid_argument("No neighbor in specified direction");
	}

	int neighbor_rank = get_neighbor_rank(neighbor_direction);
	const Resource& resource = local_metadata_.resource;

	return mpi_manager_.send_boundary(particles, count, neighbor_rank, tag, resource);
#else
	return Event(nullptr, local_metadata_.resource);
#endif
}

// Receive boundary particles from specific neighbor
Event PatchManager::receive_boundary_particles(DeviceBuffer<float>& particles,
											   idx_t max_count,
											   int neighbor_direction,
											   int tag) {
#ifdef USE_MPI
	if (!has_neighbor(neighbor_direction)) {
		throw std::invalid_argument("No neighbor in specified direction");
	}

	int neighbor_rank = get_neighbor_rank(neighbor_direction);
	const Resource& resource = local_metadata_.resource;

	return mpi_manager_.recv_boundary(particles, max_count, neighbor_rank, tag, resource);
#else
	return Event(nullptr, local_metadata_.resource);
#endif
}

// Reduce forces across overlapping regions (if needed)
Event PatchManager::reduce_overlapping_forces(DeviceBuffer<float>& forces, idx_t particle_count) {
#ifdef USE_MPI
	const Resource& resource = local_metadata_.resource;
	// This would typically use MPI_Allreduce for overlapping force regions
	// For now, return empty event
	return mpi_manager_.allReduce(forces, particle_count, resource, MPI_SUM);
#else
	return Event(nullptr, local_metadata_.resource);
#endif
}

// Broadcast global data to all patches
Event PatchManager::broadcast_global_data(DeviceBuffer<float>& data, idx_t count, int root_rank) {
#ifdef USE_MPI
	const Resource& resource = local_metadata_.resource;
	return mpi_manager_.broadcast(data, count, root_rank, resource);
#else
	return Event(nullptr, local_metadata_.resource);
#endif
}

} // namespace ARBD
