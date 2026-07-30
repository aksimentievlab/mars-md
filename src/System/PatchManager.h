/**
 * @file System/PatchManager.h
 * @author Pin-Yi Li <pinyili2@illinois.edu>
 * @brief Modern patch manager for distributed particle simulation
 * @version 2.0
 * @date 2025-09-09
 *
 * Redesigned to work with DeviceParticle system and SystemState/SimSystem architecture.
 * Manages a collection of patches with proper resource allocation and communication.
 */

#pragma once

#ifdef USE_MPI
#include "Backend/MPIManager.h"
#endif
#include "Backend/Events.h"
#include "Backend/Resource.h"
#include "Objects/DeviceParticleManager.h"
#include "Objects/Grid.h"
#include "PatchOperation/Patch.h"
#include "System/PatchDecomposer.h"
#include "System/SystemForward.h"
#include "Types/Types.h"
#include <array>
#include <memory>
#include <unordered_set>
#include <vector>

namespace ARBD {

// Forward declarations
class SimSystem;
class SystemState;

/**
 * @brief Modern patch manager with DeviceParticle integration
 *
 * Manages spatial decomposition with:
 * - Direct integration with DeviceParticle data structures
 * - Proper resource binding for GPU acceleration
 * - MPI-aware communication for distributed systems
 * - Load balancing and dynamic redecomposition
 * - Seamless integration with SystemState/SimSystem
 */
class PatchManager {
  public:
	/**
	 * @brief Neighbor communication structure for halo exchange
	 */
	struct NeighborComm {
		int neighbor_rank = -1;					  ///< MPI rank of neighbor patch
		int neighbor_patch_id = -1;				  ///< Patch ID of neighbor
		std::array<int, 3> direction = {0, 0, 0}; ///< Direction vector to neighbor
		DeviceBuffer<float> send_buffer;		  ///< Buffer for outgoing data
		DeviceBuffer<float> recv_buffer;		  ///< Buffer for incoming data
		idx_t max_halo_particles = 0;			  ///< Maximum expected halo particles
		bool active = false;					  ///< Whether this neighbor connection is active

		NeighborComm() = default;
		NeighborComm(const Resource& resource, idx_t max_particles)
			: send_buffer(max_particles * 8, resource), // 8 floats per particle
			  recv_buffer(max_particles * 8, resource), max_halo_particles(max_particles) {}
	};
	/**
	 * @brief Patch metadata for spatial and communication management
	 */
	struct PatchMetadata {
		patch_t patch_id = 0;				///< Unique patch identifier
		idx_t capacity = 0;					///< Allocated storage capacity
		idx_t particle_count = 0;			///< Current number of particles
		Vector3 min_bounds = Vector3(0.0f); ///< Spatial minimum bounds
		Vector3 max_bounds = Vector3(0.0f); ///< Spatial maximum bounds
		Resource resource;					///< Computational resource assignment

		// MPI/Communication specific
		int mpi_rank = -1;				 ///< MPI rank owning this patch (if applicable)
		std::vector<int> neighbor_ranks; ///< MPI ranks of neighboring patches

		// Grid coordinates for regular decompositions
		std::array<int, 3> grid_coords = {-1, -1, -1}; ///< Grid coordinates (i,j,k)

		PatchMetadata() = default;
		PatchMetadata(patch_t id, const Resource& res) : patch_id(id), resource(res) {}
	};
	/**
	 * @brief Spatial grid information for regular decompositions
	 */
	struct GridInfo {
		std::array<int, 3> dimensions = {1, 1, 1};		   ///< Grid dimensions (nx, ny, nz)
		std::array<bool, 3> periodic = {true, true, true}; ///< Periodic boundaries
		Vector3 grid_spacing = Vector3(0.0f);			   ///< Physical spacing between cells
		Vector3 system_origin = Vector3(0.0f);			   ///< Origin of simulation domain
	};
	/**
	 * @brief Get load statistics
	 */
	struct LoadStats {
		float min_load = 0.0f;
		float max_load = 0.0f;
		float avg_load = 0.0f;
		float imbalance_factor = 0.0f;
		std::vector<float> patch_loads;
	};

	/**
	 * @brief Construct patch manager with system reference
	 * @param system Reference to simulation system
	 */
	explicit PatchManager(SimSystem& system);

	//================================================================================
	// Initialization and Setup
	//================================================================================

	/**
	 * @brief Initialize patches from decomposition plan
	 *
	 * Primary initialization method that creates patches based on computed
	 * decomposition plan. Integrates directly with DeviceParticle system.
	 *
	 * @param plan Decomposition plan from patch decomposer
	 * @param estimated_particles Estimated particles per patch
	 * @param cutoff Cutoff radius for halo calculations
	 */
	void initialize_from_plan(const DecompositionPlan& plan, idx_t estimated_particles);

	/**
	 * @brief Initialize local patch for this device
	 * @param estimated_particles Estimated particles per patch
	 * @param cutoff Cutoff radius for halo calculations
	 */
	void initialize_local_patches(idx_t estimated_particles, const Length& cutoff);

	/**
	 * @brief Setup spatial grid from decomposition plan
	 * @param nx Number of grid cells in x direction
	 * @param ny Number of grid cells in y direction
	 * @param nz Number of grid cells in z direction
	 * @param px Whether to use periodic boundary conditions in x direction
	 * @param py Whether to use periodic boundary conditions in y direction
	 * @param pz Whether to use periodic boundary conditions in z direction
	 */
	void setup_spatial_grid(int nx, int ny, int nz, bool px, bool py, bool pz);

	/**
	 * @brief Distribute particles from SystemState to patches
	 * @param state System state containing global particle data
	 */
	void distribute_particles_from_state(SystemState& state);

	/**
	 * @brief Gather particles from patches back to SystemState
	 * @param state System state to store gathered particles
	 * @param need_energy Also gather ForceEnergy (skipped by default - see
	 *        Patch::copy_particles_to_host)
	 */
	void gather_particles_to_state(SystemState& state, bool need_energy = false);

	//================================================================================
	// Patch Access and Management
	//================================================================================

	/**
	 * @brief Get patch by ID
	 * @param patch_id Patch identifier
	 * @return Reference to patch
	 */
	Patch& get_patch(patch_t patch_id);
	const Patch& get_patch(patch_t patch_id) const;

	/**
	 * @brief Get local patch for this MPI rank
	 * @return Reference to local patch
	 */
	Patch& get_local_patch(int rank);
	const Patch& get_local_patch(int rank) const;

	/**
	 * @brief Get all patches managed by this instance
	 * @return Vector of patches
	 */
	std::vector<std::unique_ptr<Patch>>& get_patches() {
		return patches_;
	}
	const std::vector<std::unique_ptr<Patch>>& get_patches() const {
		return patches_;
	}

	/**
	 * @brief Get number of patches
	 */
	size_t get_num_patches() const {
		return patches_.size();
	}

	//================================================================================
	// Spatial Bounds and Queries
	//================================================================================

	/**
	 * @brief Get local patch bounds
	 */
	const Vector3& get_local_min_bounds() const;
	const Vector3& get_local_max_bounds() const;

	/**
	 * @brief Get global system bounds (across all patches)
	 */
	Vector3 get_global_min_bounds() const;
	Vector3 get_global_max_bounds() const;

	/**
	 * @brief Find patch containing given position
	 * @param position Position to query
	 * @return Patch ID containing position, or -1 if none
	 */
	patch_t find_patch_for_position(const Vector3& position) const;

	//================================================================================
	// Communication and Halo Exchange
	//================================================================================

	/**
	 * @brief Exchange halo particles between neighboring patches
	 * @param cutoff Cutoff radius for halo exchange
	 * @return Vector of events for async operations
	 */
	std::vector<Event> exchange_halo_particles(const Length& cutoff);

	/**
	 * @brief Migrate particles that have moved outside their patch bounds
	 * @return Vector of events for async migration
	 */
	std::vector<Event> migrate_particles();

	/**
	 * @brief Synchronize all communication operations
	 */
	void synchronize_communication();

#ifdef USE_MPI
	/**
	 * @brief Exchange metadata with other MPI ranks
	 */
	void exchange_metadata_mpi();

	/**
	 * @brief Gather global particle statistics
	 * @param local_count Local particle count
	 * @return Global particle count
	 */
	idx_t gather_global_particle_count(idx_t local_count);
#endif

	//================================================================================
	// Force Computation and Integration
	//================================================================================

	/**
	 * @brief Compute non-bonded forces across all patches
	 * @param interactions Non-bonded interactions from SimSystem
	 * @param particle_types Particle type data from SimSystem
	 * @return Vector of events for async computation
	 */
	std::vector<Event> compute_nonbonded_forces(const NonBondedInteractions& interactions,
												const BondedInteractions& bonded_interactions,
												const DeviceParticleTypes& particle_types,
												const GridManager& grid_manager,
												const TablesRegistry& tables_registry,
												float cutoff,
												size_t step,
												size_t rebuild_period,
												float electric_field = 0.0f,
												int interpolation_scheme = 1);

	/**
	 * @brief Compute bonded forces across all patches
	 * @param interactions Bonded interactions from SystemState
	 * @param particle_types Particle type data from SimSystem
	 * @param tables_registry Tabulated potential tables from SimSystem
	 * @return Vector of events for async computation
	 */
	std::vector<Event> compute_bonded_forces(const BondedInteractions& interactions,
											 const DeviceParticleTypes& particle_types,
											 const TablesRegistry& tables_registry);

	/**
	 * @brief Integrate particle motion across all patches
	 * @param dt Timestep from SimSystem
	 * @param temperature Temperature from SimSystem
	 * @param integrator_type Integration algorithm from SimSystem
	 * @return Vector of events for async integration
	 */
	std::vector<Event>
	integrate_motion(float dt, const Temperature& temperature, IntegratorType integrator_type);

	//================================================================================
	// Load Balancing and Redecomposition
	//================================================================================

	/**
	 * @brief Check if load balancing is needed
	 * @param imbalance_threshold Threshold for load imbalance
	 * @return True if rebalancing is recommended
	 */
	bool needs_rebalancing(float imbalance_threshold = 0.1f) const;

	/**
	 * @brief Trigger redecomposition for load balancing
	 * @param state Current system state
	 */
	void rebalance_patches(SystemState& state);

	LoadStats get_load_statistics() const;

	//================================================================================
	// Neighbor Management
	//================================================================================

	/**
	 * @brief Discover and setup neighbor relationships
	 */
	void discover_neighbors();

	/**
	 * @brief Get neighbors for specific patch
	 * @param patch_id Patch to query
	 * @return Vector of neighbor patch IDs
	 */
	std::vector<patch_t> get_patch_neighbors(patch_t patch_id) const;

	//================================================================================
	// Metadata and Information
	//================================================================================

	/**
	 * @brief Get metadata for specific patch
	 * @param patch_id Patch identifier
	 * @return Patch metadata
	 */
	const PatchMetadata& get_patch_metadata(patch_t patch_id) const;

	/**
	 * @brief Get all patch metadata
	 * @return Vector of all metadata
	 */
	const std::vector<PatchMetadata>& get_all_metadata() const {
		return patch_metadata_;
	}
	/**
	 * @brief Populate metadata for all patches
	 */
	void populate_metadata();
	/**
	 * @brief Get grid information (for regular decompositions)
	 */
	const GridInfo& get_grid_info() const {
		return grid_info_;
	}

	/**
	 * @brief Print status information
	 */
	void print_status() const;

	/**
	 * @brief Validate patch setup and consistency
	 */
	bool validate_patch_setup() const;

  private:
	//================================================================================
	// Core Data Members
	//================================================================================
	SimSystem& sim_system_;						  ///< Reference to simulation system
	std::vector<std::unique_ptr<Patch>> patches_; ///< Managed patches
	std::vector<PatchMetadata> patch_metadata_;	  ///< Metadata for all patches

	// Spatial decomposition information
	GridInfo grid_info_; ///< Grid structure info

	// Communication structures
	std::vector<std::vector<NeighborComm>> neighbor_comm_; ///< Per-patch neighbor communication

#ifdef USE_MPI
	MPI::Manager& mpi_manager_; ///< MPI communication manager
	patch_t local_patch_id_{0}; ///< ID of local patch for this rank
#endif

	//================================================================================
	// Private Helper Methods
	//================================================================================

	/**
	 * @brief Setup spatial grid from decomposition plan
	 */
	void setup_spatial_grid(const DecompositionPlan& plan);

	/**
	 * @brief Initialize communication buffers
	 */
	void initialize_communication_buffers();

	/**
	 * @brief Calculate optimal halo buffer sizes
	 */
	idx_t calculate_max_halo_particles(const Length& cutoff) const;

	/**
	 * @brief Convert grid coordinates to patch ID
	 */
	patch_t grid_coords_to_patch_id(int i, int j, int k) const;

	/**
	 * @brief Convert patch ID to grid coordinates
	 */
	std::array<int, 3> patch_id_to_grid_coords(patch_t patch_id) const;

	/**
	 * @brief Check if two patches are spatial neighbors
	 */
	bool are_spatial_neighbors(patch_t patch_id1, patch_t patch_id2) const;

#ifdef USE_MPI
	/**
	 * @brief Convert MPI rank to grid coordinates
	 */
	std::array<int, 3> rank_to_grid_coords(int rank) const;

	/**
	 * @brief Convert grid coordinates to MPI rank
	 */
	int grid_coords_to_rank(int x, int y, int z) const;
#endif
};

} // namespace ARBD
