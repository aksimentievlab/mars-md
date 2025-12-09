/*********************************************************************
 * @file  Patch.h
 *
 * @brief Declaration of PatchManagerOp class. Is a Singleton class that manages the patches for the
 * system. It:
 * 1. Manages the patches for the system.
 * 2. Manages the metadata for the patches.
 * 3. Manages the communication between the patches.
 * 4. Manages the computation for the patches.
 * 5. Manages the neighbor list for the patches.
 * 6. Manages the pair list for the patches.
 *
 * @details This file contains the declaration of the abstract base
 *          class PatchManagerOp, which operates on Patch data. It also
 *          includes headers of derived classes for convenient access
 *          to factory methods. The virtual method
 *          `PatchManagerOp::compute(Patch* patch)` is called by
 *          `patch->compute()` by any Patch to which the PatchOp has
 *          been added.'
 * @todo: Implement neighbour list (v1 cell decomposition)
 *********************************************************************/
// src/System/PatchManager.h
#pragma once

#ifdef USE_MPI
#include "Backend/MPIManager.h"
#endif
#include "Backend/Resource.h"
#include "PatchOperation/Patch.h"
#include "System/Decomposer.h"
#include "System/PeriodicBox.h"
#include "Types/Types.h"
#include <array>
#include <numeric>
#include <unordered_set>
namespace ARBD {

// Forward declarations

class Patch;
class Decomposer;

/**
 * @brief Enhanced PatchManager with MPI communication support
 *
 * Manages patches in a spatial decomposition with:
 * - One patch per GPU/MPI rank
 * - Neighbor detection and communication
 * - Halo particle exchange
 * - Load balancing capabilities
 */
class PatchManager {
	friend Decomposer;

  public:
	PatchManager(SimSystem& sys);

	/**
	 * @brief Metadata for each patch including spatial and communication info
	 */
	struct Metadata {
		idx_t num = 0;				 ///< Current number of particles
		idx_t capacity = 0;			 ///< Allocated capacity
		Vector3 min = Vector3(0.0f); ///< Spatial minimum bounds
		Vector3 max = Vector3(0.0f); ///< Spatial maximum bounds

		// MPI/Communication specific
		int mpi_rank = -1;			///< MPI rank owning this patch
		Resource resource;			///< Compute resource for this patch
		std::vector<int> neighbors; ///< MPI ranks of neighboring patches

		// Grid coordinates for spatial decomposition
		std::array<int, 3> grid_coords = {-1, -1, -1}; ///< Grid coordinates (i,j,k)

		Metadata() = default;
	};

	/**
	 * @brief Neighbor communication structure
	 */
	struct NeighborComm {
		int neighbor_rank = -1;					  ///< MPI rank of neighbor
		std::array<int, 3> direction = {0, 0, 0}; ///< Direction vector to neighbor
		DeviceBuffer<float> send_buffer;		  ///< Buffer for outgoing halo data
		DeviceBuffer<float> recv_buffer;		  ///< Buffer for incoming halo data
		idx_t halo_size = 0;					  ///< Number of halo particles
		bool active = false;					  ///< Whether this neighbor exists

		NeighborComm() = default;
		NeighborComm(const Resource& resource, idx_t max_halo_size)
			: send_buffer(max_halo_size * 6, resource), // 6 = 3 position + 3 velocity
			  recv_buffer(max_halo_size * 6, resource), halo_size(0) {}
	};
	struct GridInfo {
		std::array<int, 3> dimensions; ///< Grid dimensions (nx, ny, nz)
		std::array<bool, 3> periodic;  ///< Periodic boundary conditions
		Vector3 grid_spacing;		   ///< Physical spacing between grid cells
		Vector3 system_origin;		   ///< Origin of the simulation domain
	};

  private:
	SimSystem& sys_;
	MPI::Manager& mpi_manager_;
	size_t global_seed_;
	std::unordered_map<const Resource*, size_t> rng_seeds_;

	// Patch management
	std::vector<Patch> patches_;
	Metadata local_metadata_;			 ///< Metadata for local patch (one per rank)
	std::vector<Metadata> all_metadata_; ///< Metadata for all patches (gathered from all ranks)

	// Spatial grid information
	GridInfo grid_info_;

	// Communication structures
	std::array<NeighborComm, 2> neighbors_; ///< just 2 neighbors in 2D grid
	idx_t max_ghost_particles_;

	// Communication tags for different data types
	enum CommTags {
		POSITIONS_TAG = 1000,
		VELOCITIES_TAG = 2000,
		FORCES_TAG = 3000,
		METADATA_TAG = 4000
	};

  public:
	// ==================== Initialization ====================

	/**
	 * @brief Initialize patch manager from a decomposition plan
	 *
	 * This is the primary initialization method that should be called once
	 * during system setup after decomposition. It creates patches based on
	 * the provided decomposition plan.
	 *
	 * @param plan The decomposition plan computed by a PatchDecomposer
	 * @param estimated_particles_per_patch Estimated number of particles per patch
	 * @param cutoff Cutoff radius for halo region calculation
	 */
	void initialize_from_plan(const struct DecompositionPlan& plan,
							  idx_t estimated_particles_per_patch,
							  const Length& cutoff);

	/**
	 * @brief Initialize patch manager with MPI integration
	 * @deprecated Use initialize_from_plan() instead. This method is kept for backward
	 * compatibility.
	 */
	void initialize(int grid_nx,
					int grid_ny,
					int grid_nz,
					bool periodic_x = true,
					bool periodic_y = true,
					bool periodic_z = true);

	/**
	 * @brief Setup spatial grid and determine neighbors
	 */
	void setup_spatial_grid(int nx, int ny, int nz, bool px, bool py, bool pz);

	/**
	 * @brief Initialize local patch for this MPI rank
	 */
	void initialize_local_patch(idx_t estimated_particles, const Length& cutoff);

	// ==================== Spatial Bounds ====================

	const Vector3 get_min() const {
		return !patches_.empty() ? patches_[0].get_bounds_min() : local_metadata_.min;
	}
	const Vector3 get_max() const {
		return !patches_.empty() ? patches_[0].get_bounds_max() : local_metadata_.max;
	}

	const Vector3 get_global_min() const {
		Vector3 min(Vector3::highest());
		for (const auto& meta : all_metadata_) {
			min.x = std::min(min.x, meta.min.x);
			min.y = std::min(min.y, meta.min.y);
			min.z = std::min(min.z, meta.min.z);
		}
		return min;
	}

	const Vector3 get_global_max() const {
		Vector3 max(Vector3::lowest());
		for (const auto& meta : all_metadata_) {
			max.x = std::max(max.x, meta.max.x);
			max.y = std::max(max.y, meta.max.y);
			max.z = std::max(max.z, meta.max.z);
		}
		return max;
	}

	// ==================== Neighbor Management ====================

	/**
	 * @brief Find and setup neighbor relationships
	 */
	void discover_neighbors();

	/**
	 * @brief Get neighbor in specific direction
	 * @param direction 0=left, 1=right
	 */
	int get_neighbor_rank(int direction) const {
		return (direction >= 0 && direction < 2) ? neighbors_[direction].neighbor_rank : -1;
	}

	/**
	 * @brief Check if patch has neighbor in direction
	 */
	bool has_neighbor(int direction) const {
		return direction >= 0 && direction < 2 && neighbors_[direction].active;
	}

	/**
	 * @brief Get all active neighbor ranks
	 */
	std::vector<int> get_all_neighbors() const {
		std::vector<int> neighbor_ranks;
		for (const auto& neighbor : neighbors_) {
			if (neighbor.active) {
				neighbor_ranks.push_back(neighbor.neighbor_rank);
			}
		}
		return neighbor_ranks;
	}

	// ==================== Communication Operations ====================

	/**
	 * @brief Exchange halo particles with all neighbors
	 */
	std::vector<Event> exchange_halo_particles(DeviceBuffer<float>& positions,
											   DeviceBuffer<float>& velocities,
											   const Length& cutoff);

	/**
	 * @brief Send boundary particles to specific neighbor
	 */
	Event send_boundary_particles(DeviceBuffer<float>& particles,
								  idx_t count,
								  int neighbor_direction,
								  int tag);

	/**
	 * @brief Receive boundary particles from specific neighbor
	 */
	Event receive_boundary_particles(DeviceBuffer<float>& particles,
									 idx_t max_count,
									 int neighbor_direction,
									 int tag);

	/**
	 * @brief Reduce forces across overlapping regions (if needed)
	 */
	Event reduce_overlapping_forces(DeviceBuffer<float>& forces, idx_t particle_count);

	/**
	 * @brief Broadcast global parameters to all patches
	 */
	Event broadcast_global_data(DeviceBuffer<float>& data, idx_t count, int root_rank = 0);

	/**
	 * @brief Gather local statistics for global reduction
	 */
	template<typename T>
	Event gather_local_statistics(T local_value, T& global_result, MPI_Op operation = MPI_SUM) {
		DeviceBuffer<T> local_buffer(1, local_metadata_.resource);
		DeviceBuffer<T> global_buffer(1, local_metadata_.resource);

		local_buffer.copy_from_host(&local_value, 1);
		auto event = mpi_manager_.allReduce(local_buffer, 1, local_metadata_.resource, operation);
		event.wait();

		local_buffer.copy_to_host(&global_result, 1);
		return event;
	}

	// ==================== Metadata Exchange ====================

	/**
	 * @brief Exchange metadata with all ranks
	 */
	void exchange_metadata();

	/**
	 * @brief Update local patch bounds
	 */
	void update_local_bounds(const Vector3& new_min, const Vector3& new_max);

	/**
	 * @brief Get metadata for specific rank
	 */
	const Metadata& get_patch_metadata(int rank) const {
		if (rank >= 0 && rank < static_cast<int>(all_metadata_.size())) {
			return all_metadata_[rank];
		}
		throw std::out_of_range("Invalid rank for patch metadata");
	}

	// ==================== Utility Functions ====================

	/**
	 * @brief Convert grid coordinates to rank
	 */
	int grid_coords_to_rank(int i, int j, int k) const {
		if (i < 0 || i >= grid_info_.dimensions[0] || j < 0 || j >= grid_info_.dimensions[1] ||
			k < 0 || k >= grid_info_.dimensions[2]) {
			return -1; // Invalid coordinates
		}
		return i + j * grid_info_.dimensions[0] +
			   k * grid_info_.dimensions[0] * grid_info_.dimensions[1];
	}

	/**
	 * @brief Convert rank to grid coordinates
	 */
	std::array<int, 3> rank_to_grid_coords(int rank) const {
		int nx = grid_info_.dimensions[0];
		int ny = grid_info_.dimensions[1];

		int k = rank / (nx * ny);
		int remainder = rank % (nx * ny);
		int j = remainder / nx;
		int i = remainder % nx;

		return {i, j, k};
	}

	/**
	 * @brief Check if two patches are neighbors
	 */
	bool are_neighbors(const std::array<int, 3>& coords1, const std::array<int, 3>& coords2) const {
		int dx = std::abs(coords1[0] - coords2[0]);
		int dy = std::abs(coords1[1] - coords2[1]);
		int dz = std::abs(coords1[2] - coords2[2]);

		// Handle periodic boundaries
		if (grid_info_.periodic[0]) {
			dx = std::min(dx, grid_info_.dimensions[0] - dx);
		}
		if (grid_info_.periodic[1]) {
			dy = std::min(dy, grid_info_.dimensions[1] - dy);
		}
		if (grid_info_.periodic[2]) {
			dz = std::min(dz, grid_info_.dimensions[2] - dz);
		}

		// Neighbors are adjacent in exactly one dimension
		int adjacent_dims = (dx <= 1 ? 1 : 0) + (dy <= 1 ? 1 : 0) + (dz <= 1 ? 1 : 0);
		int zero_dims = (dx == 0 ? 1 : 0) + (dy == 0 ? 1 : 0) + (dz == 0 ? 1 : 0);

		return adjacent_dims == 3 && zero_dims == 2; // Adjacent in one dimension, same in two
	}

	/**
	 * @brief Get direction index for neighbor relationship
	 */
	int get_neighbor_direction(const std::array<int, 3>& my_coords,
							   const std::array<int, 3>& neighbor_coords) const {
		int dx = neighbor_coords[0] - my_coords[0];
		int dy = neighbor_coords[1] - my_coords[1];
		int dz = neighbor_coords[2] - my_coords[2];

		// Handle periodic wrapping
		if (grid_info_.periodic[0] && std::abs(dx) > 1) {
			dx = -dx / std::abs(dx); // Flip direction for wrap-around
		}
		if (grid_info_.periodic[1] && std::abs(dy) > 1) {
			dy = -dy / std::abs(dy);
		}
		if (grid_info_.periodic[2] && std::abs(dz) > 1) {
			dz = -dz / std::abs(dz);
		}

		// Convert to direction index
		if (dx == -1 && dy == 0 && dz == 0)
			return 0; // x-
		if (dx == +1 && dy == 0 && dz == 0)
			return 1; // x+
		if (dx == 0 && dy == -1 && dz == 0)
			return 2; // y-
		if (dx == 0 && dy == +1 && dz == 0)
			return 3; // y+
		if (dx == 0 && dy == 0 && dz == -1)
			return 4; // z-
		if (dx == 0 && dy == 0 && dz == +1)
			return 5; // z+

		return -1; // Not a face neighbor
	}

	// ==================== Load Balancing ====================

	/**
	 * @brief Calculate load for local patch
	 */
	float calculate_local_load() const {
		// Simple load metric based on particle count
		// Could be enhanced with computation time, memory usage, etc.
		return static_cast<float>(local_metadata_.num);
	}

	/**
	 * @brief Gather load information from all patches
	 */
	std::vector<float> gather_all_loads() const {
		float local_load = calculate_local_load();
		std::vector<float> all_loads(mpi_manager_.get_size());

		MPI_Allgather(&local_load, 1, MPI_FLOAT, all_loads.data(), 1, MPI_FLOAT, MPI_COMM_WORLD);

		return all_loads;
	}

	/**
	 * @brief Check if load balancing is needed
	 */
	bool needs_rebalancing(float imbalance_threshold = 0.1f) const {
		auto loads = gather_all_loads();

		float min_load = *std::min_element(loads.begin(), loads.end());
		float max_load = *std::max_element(loads.begin(), loads.end());
		float avg_load = std::accumulate(loads.begin(), loads.end(), 0.0f) / loads.size();

		float imbalance = (max_load - min_load) / avg_load;

		if (mpi_manager_.get_rank() == 0) {
			LOGINFO("Load balance check: min={:.1f}, max={:.1f}, avg={:.1f}, imbalance={:.3f}",
					min_load,
					max_load,
					avg_load,
					imbalance);
		}

		return imbalance > imbalance_threshold;
	}

	// ==================== Particle Management ====================

	// ==================== Synchronization ====================

	/**
	 * @brief Wait for all communication to complete
	 */
	void wait_all_communication() {
		mpi_manager_.wait_all();
	}

	/**
	 * @brief Barrier synchronization across all patches
	 */
	void synchronize_all_patches() {
		mpi_manager_.barrier();
	}

	const std::vector<Patch>& get_all_patches() {
		return patches_;
	}

	/**
	 * @brief Get the local patch for this MPI rank
	 * @return Reference to the local patch
	 */
	Patch& get_local_patch() {
		if (patches_.empty()) {
			throw std::runtime_error("No patches initialized");
		}
		return patches_[0]; // In MPI mode, each rank has one local patch
	}

	const Patch& get_local_patch() const {
		if (patches_.empty()) {
			throw std::runtime_error("No patches initialized");
		}
		return patches_[0]; // In MPI mode, each rank has one local patch
	}

	/**
	 * @brief Test for completed communications without blocking
	 */
	void test_communications() {
		mpi_manager_.test_completions();
	}

	// ==================== Information and Debugging ====================

	/**
	 * @brief Print patch manager status
	 */
	void print_status() const {
		if (mpi_manager_.get_rank() == 0) {
			LOGINFO("=== PatchManager Status ===");
			LOGINFO("Grid dimensions: {}x{}x{}",
					grid_info_.dimensions[0],
					grid_info_.dimensions[1],
					grid_info_.dimensions[2]);
			LOGINFO("Periodic boundaries: x={}, y={}, z={}",
					grid_info_.periodic[0],
					grid_info_.periodic[1],
					grid_info_.periodic[2]);
			LOGINFO("Total patches: {}", mpi_manager_.get_size());
		}

		// Print local patch info
		auto coords = rank_to_grid_coords(mpi_manager_.get_rank());
		int num_neighbors = std::count_if(neighbors_.begin(),
										  neighbors_.end(),
										  [](const NeighborComm& n) { return n.active; });

		LOGINFO("Rank {}: grid_coords=({},{},{}), bounds=({:.2f},{:.2f},{:.2f}) to "
				"({:.2f},{:.2f},{:.2f}), neighbors={}",
				mpi_manager_.get_rank(),
				coords[0],
				coords[1],
				coords[2],
				local_metadata_.min.x,
				local_metadata_.min.y,
				local_metadata_.min.z,
				local_metadata_.max.x,
				local_metadata_.max.y,
				local_metadata_.max.z,
				num_neighbors);
	}

	/**
	 * @brief Get local patch metadata
	 */
	const Metadata& get_local_metadata() const {
		return local_metadata_;
	}

	/**
	 * @brief Get all patch metadata (collected from all ranks)
	 */
	const std::vector<Metadata>& get_all_metadata() const {
		return all_metadata_;
	}

	/**
	 * @brief Get grid information
	 */
	const GridInfo& get_grid_info() const {
		return grid_info_;
	}

	/**
	 * @brief Validate patch setup
	 */
	bool validate_patch_setup() const {
		// Check that grid is complete
		if (mpi_manager_.get_size() !=
			grid_info_.dimensions[0] * grid_info_.dimensions[1] * grid_info_.dimensions[2]) {
			LOGERROR("Mismatch: MPI size ({}) != grid size ({})",
					 mpi_manager_.get_size(),
					 grid_info_.dimensions[0] * grid_info_.dimensions[1] *
						 grid_info_.dimensions[2]);
			return false;
		}

		// Check neighbor consistency
		for (int dir = 0; dir < 6; ++dir) {
			if (neighbors_[dir].active) {
				int neighbor_rank = neighbors_[dir].neighbor_rank;
				if (neighbor_rank < 0 || neighbor_rank >= mpi_manager_.get_size()) {
					LOGERROR("Invalid neighbor rank {} in direction {}", neighbor_rank, dir);
					return false;
				}
			}
		}

		return true;
	}
};

} // namespace ARBD
