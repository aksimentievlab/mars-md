#pragma once
/**
 * @file PatchOperation/Patch.h
 * @author Pin-Yi Li <pinyili2@illinois.edu>
 * @brief Modern patch-based spatial decomposition unit
 * @version 2.0
 * @date 2025-09-09
 *
 * Redesigned to work with DeviceParticle system and SystemState/SimSystem architecture.
 * Each patch manages a spatial region with local particles using DeviceParticle for
 * optimal GPU memory layout and performance.
 */

#include "Backend/Events.h"
#include "Interactions/BondedInteraction.h"
#include "Interactions/NonBondedInteraction.h"
#include "Objects/DeviceParticleManager.h"
#include "PatchOperation/Pairlist.h"
#include "System/PeriodicBox.h"
#include "System/SystemForward.h"
#include "Types/Types.h"
#include <memory>
#include <utility>

namespace ARBD {

// Forward declarations
class SimSystem;
class SystemState;

/**
 * @brief Spatial decomposition unit for distributed particle simulation
 *
 * Each Patch represents a spatial region containing particles and manages:
 * - Local particle data using DeviceParticle for optimal memory layout
 * - Spatial bounds and neighbor relationships
 * - Particle migration and halo exchange
 * - Local force computation and time integration
 * - Resource binding for GPU acceleration
 */
class Patch {
  public:
	/**
	 * @brief Construct patch with specified parameters
	 * @param patch_id Unique patch identifier across the system
	 * @param capacity Maximum number of particles this patch can store
	 * @param resource Computational resource (GPU/CPU) assigned to this patch
	 * @param pairlist_type Type of pairlist to use for neighbor finding
	 */
	Patch(patch_t patch_id,
		  idx_t capacity,
		  const Resource& resource,
		  PairlistBuilderType pairlist_type = PairlistBuilderType::ZOrder)
		: patch_id_(patch_id), capacity_(capacity), resource_(resource),
		  particles_(capacity, resource) {
		// Estimate max pairs: assume average of ~50 neighbors per particle
		size_t estimated_max_pairs = capacity * 50;
		pairlist_ = create_pairlist(pairlist_type, resource, capacity, estimated_max_pairs);
		initialize_spatial_structures();
	}

	//================================================================================
	// Communication Buffers for Particle Exchange
	//================================================================================
	struct HaloBuffers {
		DeviceBuffer<DeviceParticle> send_particles; ///< Outgoing halo particle data
		DeviceBuffer<DeviceParticle> recv_particles; ///< Incoming halo particle data
		DeviceBuffer<int> migration_flags;			 ///< Per-particle migration direction flags
		DeviceBuffer<int> migration_count; ///< Number of migrating particles per direction

		HaloBuffers(const Resource& resource, idx_t capacity)
			: send_particles(capacity, resource), // 8 floats per particle (pos+vel+type+flags)
			  recv_particles(capacity, resource), migration_flags(capacity, resource),
			  migration_count(6, resource) {} // 6 directions
	};

	std::unique_ptr<HaloBuffers> halo_buffers_{nullptr};

	/**
	 * @brief Set reference to system-wide periodic boundary conditions
	 * @param sim_box Pointer to system boundary conditions (not owned)
	 */
	void set_periodic_box(const PeriodicBox* sim_box) {
		periodic_box_ = sim_box;
	}

	/**
	 * @brief Set the current number of active particles in this patch
	 * @param count New particle count (must be <= capacity)
	 */
	void set_particle_count(idx_t count) {
		if (count <= capacity_) {
			particle_count_ = count;
		}
	}

	/**
	 * @brief Set halo region thickness for ghost particle exchange
	 * @param thickness Thickness of halo region (typically cutoff radius)
	 */
	void set_halo_thickness(float thickness) {
		halo_thickness_ = thickness;
	}

	/**
	 * @brief Set device particle types for this patch
	 * @param particle_types Device particle types (moved into patch)
	 */
	void set_particle_types(std::unique_ptr<DeviceParticleTypes> particle_types) {
		particle_types_ = std::move(particle_types);
	}

	/**
	 * @brief Set spatial boundaries for this patch region
	 * @param min_bounds Minimum corner coordinates
	 * @param max_bounds Maximum corner coordinates
	 */
	void set_bounds(const Vector3& min_bounds, const Vector3& max_bounds) {
		min_bounds_ = min_bounds;
		max_bounds_ = max_bounds;
		// Update any spatial acceleration structures if needed
		update_space_partition();
	}

	/**
	 * @brief Get reference to system-wide periodic boundary conditions
	 * @return Pointer to periodic boundary conditions
	 */
	const PeriodicBox* get_periodic_box() const {
		return periodic_box_;
	}

	/**
	 * @brief Get unique patch identifier
	 */
	patch_t get_patch_id() const {
		return patch_id_;
	}

	/**
	 * @brief Get current number of active particles
	 */
	idx_t get_particle_count() const {
		return particle_count_;
	}

	/**
	 * @brief Get patch capacity
	 */
	idx_t get_capacity() const {
		return capacity_;
	}

	/**
	 * @brief Check if patch has space for more particles
	 */
	bool has_space() const {
		return particle_count_ < capacity_;
	}

	/**
	 * @brief Compute non-bonded forces for particles in this patch
	 * @param interactions Non-bonded interaction parameters from SimSystem
	 * @param particle_types Device particle type data from SimSystem
	 * @return Event for async GPU execution
	 */
	Event calculate_nonbonded_forces(const NonBondedInteractions& interactions,
									 const DeviceParticleTypes& particle_types);

	/**
	 * @brief Compute bonded forces for particles in this patch
	 * @param interactions Bonded interaction parameters from SystemState
	 * @param particle_types Device particle type data from SimSystem
	 * @return Event for async GPU execution
	 */
	Event calculate_bonded_forces(const BondedInteractions& interactions,
								  const DeviceParticleTypes& particle_types);

	/**
	 * @brief Integrate particle equations of motion
	 * @param dt Timestep from SimSystem
	 * @param temperature Temperature from SimSystem (position-dependent if gridded)
	 * @param integrator_type Integration algorithm from SimSystem
	 * @param step Current simulation step (for RNG counter)
	 * @return Event for async GPU execution
	 */
	Event integrate_motion(float dt,
						   const Temperature& temperature,
						   IntegratorType integrator_type,
						   size_t step = 0);

	/**
	 * @brief
	 *
	 */
	idx_t get_remaining_capacity() const {
		return capacity_ - particle_count_;
	}

	/**
	 * @brief Get patch minimum spatial bounds
	 */
	const Vector3& get_min_bounds() const {
		return min_bounds_;
	}

	/**
	 * @brief Get patch maximum spatial bounds
	 */
	const Vector3& get_max_bounds() const {
		return max_bounds_;
	}

	/**
	 * @brief Get halo region thickness for ghost particles
	 * @return Halo thickness (typically equal to cutoff radius)
	 */
	float get_halo_thickness() const {
		return halo_thickness_;
	}

	/**
	 * @brief Check if position is outside patch core region (in halo)
	 * @param position Position to test
	 * @return True if position is in halo/ghost region
	 */
	bool is_in_halo_region(const Vector3& position) const {
		return position.x < min_bounds_.x || position.x >= max_bounds_.x ||
			   position.y < min_bounds_.y || position.y >= max_bounds_.y ||
			   position.z < min_bounds_.z || position.z >= max_bounds_.z;
	}

	/**
	 * @brief Get computational resource assigned to this patch
	 * @return Resource (GPU/CPU) for this patch
	 */
	const Resource& get_resource() const {
		return resource_;
	}

	/**
	 * @brief Determine if particle needs migration and in which direction
	 * @param position Particle position to test
	 * @return Migration direction: 0=x-, 1=x+, 2=y-, 3=y+, 4=z-, 5=z+, -1=no migration
	 */
	int check_migration_direction(const Vector3& position) const {
		if (position.x < min_bounds_.x)
			return 0; // x-
		if (position.x >= max_bounds_.x)
			return 1; // x+
		if (position.y < min_bounds_.y)
			return 2; // y-
		if (position.y >= max_bounds_.y)
			return 3; // y+
		if (position.z < min_bounds_.z)
			return 4; // z-
		if (position.z >= max_bounds_.z)
			return 5; // z+
		return -1;	  // No migration needed
	}

	/**
	 * @brief Check if a position is within the boundary region that should be sent to neighbor
	 * @param position Position to check
	 * @param direction Direction to neighbor (0=x-, 1=x+, 2=y-, 3=y+, 4=z-, 5=z+)
	 * @param halo_width Width of the halo region
	 * @return True if position is in boundary region
	 */
	bool is_in_boundary_region(const Vector3& position, int direction, float halo_width) const {
		switch (direction) {
		case 0: // x-
			return position.x >= min_bounds_.x && position.x < min_bounds_.x + halo_width;
		case 1: // x+
			return position.x >= max_bounds_.x - halo_width && position.x < max_bounds_.x;
		case 2: // y-
			return position.y >= min_bounds_.y && position.y < min_bounds_.y + halo_width;
		case 3: // y+
			return position.y >= max_bounds_.y - halo_width && position.y < max_bounds_.y;
		case 4: // z-
			return position.z >= min_bounds_.z && position.z < min_bounds_.z + halo_width;
		case 5: // z+
			return position.z >= max_bounds_.z - halo_width && position.z < max_bounds_.z;
		default:
			return false;
		}
	}

	/**
	 * @brief Calculate halo region bounds for specific direction
	 * @param direction Direction index (0=x-, 1=x+, 2=y-, 3=y+, 4=z-, 5=z+)
	 * @param halo_width Width of halo region
	 * @return Pair of (halo_min, halo_max) bounds
	 */
	std::pair<Vector3, Vector3> calculate_halo_bounds(int direction, float halo_width) const {
		Vector3 halo_min = min_bounds_;
		Vector3 halo_max = max_bounds_;

		switch (direction) {
		case 0: // x-
			halo_max.x = min_bounds_.x;
			halo_min.x = min_bounds_.x - halo_width;
			break;
		case 1: // x+
			halo_min.x = max_bounds_.x;
			halo_max.x = max_bounds_.x + halo_width;
			break;
		case 2: // y-
			halo_max.y = min_bounds_.y;
			halo_min.y = min_bounds_.y - halo_width;
			break;
		case 3: // y+
			halo_min.y = max_bounds_.y;
			halo_max.y = max_bounds_.y + halo_width;
			break;
		case 4: // z-
			halo_max.z = min_bounds_.z;
			halo_min.z = min_bounds_.z - halo_width;
			break;
		case 5: // z+
			halo_min.z = max_bounds_.z;
			halo_max.z = max_bounds_.z + halo_width;
			break;
		}

		return {halo_min, halo_max};
	}

	/**
	 * @brief Get read-only access to local particle data
	 */
	const DeviceParticle& get_particles() const {
		return particles_;
	}

	/**
	 * @brief Get mutable access to local particle data
	 */
	DeviceParticle& get_particles() {
		return particles_;
	}

	/**
	 * @brief Pack boundary particles for halo exchange with neighbor
	 * @param send_buffer Output buffer for packed data
	 * @param direction Neighbor direction (0=x-, 1=x+, 2=y-, 3=y+, 4=z-, 5=z+)
	 * @param halo_width Width of halo region (typically cutoff radius)
	 * @return Event for async packing and number of particles packed
	 */
	std::pair<Event, idx_t>
	pack_halo_particles(DeviceBuffer<float>& send_buffer, int direction, float halo_width);

	/**
	 * @brief Unpack received halo particles from neighbor
	 * @param recv_buffer Input buffer with packed particle data
	 * @param particle_count Number of particles to unpack
	 * @return Event for async unpacking
	 */
	Event unpack_halo_particles(const DeviceBuffer<float>& recv_buffer, idx_t particle_count);

	/**
	 * @brief Sort particles using pairlist's spatial sorting (e.g., Z-order curve)
	 * @return Event for async sorting
	 */
	Event sort_particles();

	/**
	 * @brief Build pairlist for neighbor finding
	 * @param cutoff Interaction cutoff distance
	 * @return Event for async pairlist building
	 */
	Event build_pairlist(float pairlist_cutoff);

	/**
	 * @brief Update pairlist if particles have moved significantly
	 * @return Event for async pairlist update
	 */
	Event update_pairlist();

	/**
	 * @brief Check if pairlist needs updating based on particle displacement
	 * @param old_positions Previous particle positions
	 * @param skin_distance Skin distance for update criterion
	 * @return True if update is needed
	 */
	bool needs_pairlist_update(const DeviceBuffer<Vector3>& old_positions, float skin_distance);

	/**
	 * @brief Get the pairlist for accessing neighbor pairs
	 * @return Reference to the pairlist
	 */
	const Pairlist& get_pairlist() const {
		return *pairlist_;
	}

	/**
	 * @brief Get mutable access to the pairlist
	 * @return Reference to the pairlist
	 */
	Pairlist& get_pairlist() {
		return *pairlist_;
	}

	/**
	 * @brief Execute integration for particles in this patch
	 * @param dt Timestep from SimSystem
	 * @param temperature Temperature from SimSystem (position-dependent if gridded)
	 * @param integrator_type Integration algorithm from SimSystem
	 * @return Event for async integration
	 */
	Event
	execute_integration(float dt, const Temperature& temperature, IntegratorType integrator_type);

	/**
	 * @brief Copy particle data from host to this patch's device storage
	 * @param host_data Host particle data from SystemState
	 * @param start_idx Starting index in host data
	 * @param count Number of particles to copy
	 */
	void copy_particles_from_host(const HostParticleData& host_data, idx_t start_idx, idx_t count);

	/**
	 * @brief Copy particle data from this patch to host storage
	 * @param host_data Output host particle data
	 * @param start_idx Starting index in host data for output
	 * @param count Number of particles to copy
	 */
	void copy_particles_to_host(HostParticleData& host_data, idx_t start_idx, idx_t count) const;

  private:
	/**
	 * @brief Initialize spatial acceleration structures
	 */
	void initialize_spatial_structures();

	/**
	 * @brief Update space partitioning after bounds change
	 */
	void update_space_partition();
	//================================================================================
	// Core Patch Properties
	//================================================================================
	patch_t patch_id_;		  ///< Unique patch identifier
	idx_t capacity_;		  ///< Maximum particle storage capacity
	idx_t particle_count_{0}; ///< Current number of active particles
	Resource resource_;		  ///< Computational resource (GPU/CPU)

	//================================================================================
	// Particle Data and Spatial Organization
	//================================================================================
	HostParticleData host_particles_;	 ///< used for staging on HOST.
	DeviceParticle particles_;			 ///< Local particle data in SoA format
	std::unique_ptr<Pairlist> pairlist_; ///< Pairlist for neighbor finding and spatial organization
	std::unique_ptr<DeviceParticleTypes> particle_types_; ///< Device buffers for all particle types

	float halo_thickness_{0.0f}; ///< Halo region thickness for ghost particles

	//================================================================================
	// Spatial Bounds and Geometry
	//================================================================================
	Vector3 min_bounds_{0.0f}; ///< Patch minimum spatial bounds
	Vector3 max_bounds_{0.0f}; ///< Patch maximum spatial bounds

	//================================================================================
	// System References (not owned)
	//================================================================================
	const PeriodicBox* periodic_box_{nullptr}; ///< System boundary conditions
};

/**
 * @brief Factory function to create patch decomposer
 * @param type Type of decomposer to create
 * @return Unique pointer to decomposer
 */
std::unique_ptr<PatchDecomposer> create_patch_decomposer(DecomposerType type);

} // namespace ARBD
