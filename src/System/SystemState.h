#pragma once
/**
 * @file System/SystemState.h
 * @author Pin-Yi Li <pinyili2@illinois.edu>
 * @brief Runtime system state management for simulation
 * @version 2.0
 * @date 2025-09-09
 *
 * @copyright Copyright (c) 2025
 */

#include "ARBDException.h"
#include "ARBDLogger.h"
#include "Backend/Buffer.h"
#include "System/SimSystem.h"
#include "SystemForward.h"
#include "Types/BaseGrid.h"
#include "Types/Types.h"
#include <memory>
#include <unordered_map>
#include <vector>

namespace ARBD {

/**
 * @brief Manages runtime, mutable system state during simulation
 *
 * SystemState serves as the HOST-side manager for particle data that is:
 * - Gathered from distributed patches (multi-GPU)
 * - Stored in host memory for I/O operations (DCD writing, analysis)
 * - Maintained in a consistent, sorted order for output
 *
 * SystemState handles ONLY changing runtime state:
 * - Particle positions, velocities, and types (change every timestep)
 * - System objects that change during simulation (bonds, grids, fields)
 * - Runtime flags and counters
 *
 * Static configuration data (temperature, cutoff, box_size) is accessed
 * directly from Configuration when needed - no duplication here.
 *
 * @note Particle data is maintained sorted by global_id for consistent DCD output
 * @note All data is stored on HOST memory, ready for file I/O operations
 */
class SystemState {
  public:
	/**
	 * @brief Construct system state (no configuration needed)
	 * @param sim_system Simulation system
	 */
	SystemState(SimSystem& sim_system) : sim_system_(sim_system) {
		initialize_system_objects();
		LOGINFO("SystemState: System state initialized");
	}

	/**
	 * @brief Destructor - cleans up GPU resources
	 */

	~SystemState() {
		cleanup_gpu_resources();
	}
	/**
	 * @brief Set particle positions (GPU-compatible)
	 * @param positions New particle positions
	 */
	void set_particle_positions(const std::vector<Vector3>& positions);

	/**
	 * @brief Get number of particles in the system
	 */
	size_t get_num_particles() const {
		return global_num_particles_;
	}

	/**
	 * @brief Set particle data
	 * @param positions Particle positions
	 * @param momenta Particle momenta
	 * @param ids Particle IDs
	 */
	void set_init_particle_data(const std::vector<ParticleRead>& particles) {
		global_particle_data_.resize(particles.size());
		for (const auto& particle : particles) {
			global_particle_data_.global_id.push_back(particle.id);
			global_particle_data_.type_id.push_back(
				sim_system_.get_particle_type_id(particle.type_name));
			global_particle_data_.pos.push_back(particle.position);
			global_particle_data_.mom.push_back(particle.momentum);
			global_particle_data_.force.push_back(particle.force);
			global_particle_data_.energy.push_back(0.0f);
			global_particle_data_.orient.push_back(Vector3(0.0f, 0.0f, 0.0f));
			global_particle_data_.is_active.push_back(true);
			global_particle_data_.is_dummy.push_back(false);
			global_particle_data_.colvars_group_id.push_back(-1);
			global_particle_data_.group_id.push_back(-1);
		}
		global_num_particles_ = global_particle_data_.size();
	}

	/**
	 * @brief Set particle data from host storage (typically from PatchManager::gather)
	 * @param particles Host particle data gathered from patches
	 * @note Particles should be sorted by ID after gathering for consistent DCD output
	 */
	void set_from_host_particle_data(const HostParticleData& particles) {
		global_particle_data_ = particles;
		global_num_particles_ = particles.size();
		// Ensure particles are sorted by ID for consistent DCD output
		sort_particles_by_id();
	}

	void update_bonded_interactions(const BondedInteractions& bonded_interactions) {
		bonded_interactions_.clear();
		bonded_interactions_ = bonded_interactions;
	}

	/**
	 * @brief Clear global particle arrays (called before gathering new data)
	 */
	void clear_global_arrays() {
		global_particle_data_.clear();
		global_num_particles_ = 0;
		state_synced_ = false;
	}

	/**
	 * @brief Mark global state as synchronized (called after gathering is complete)
	 */
	void mark_synced() {
		state_synced_ = true;
	}

	/**
	 * @brief Get global particle IDs
	 */
	const std::vector<int>& get_global_particle_ids() const {
		return global_particle_data_.global_id;
	}

	/**
	 * @brief Get global particle positions (ready for DCD writing)
	 * @return Const reference to position vector, sorted by particle ID
	 * @note Positions are guaranteed to be in ID-sorted order for consistent DCD output
	 */
	const std::vector<Vector3>& get_global_positions() const {
		return global_particle_data_.pos;
	}

	/**
	 * @brief Get global particle momenta/velocities
	 * @return Const reference to momentum vector, sorted by particle ID
	 */
	const std::vector<Vector3>& get_global_momentum() const {
		return global_particle_data_.mom;
	}

	/**
	 * @brief Get complete global particle data (for advanced operations)
	 * @return Const reference to HostParticleData structure
	 */
	const HostParticleData& get_global_particles() const {
		return global_particle_data_;
	}

	/**
	 * @brief Check if global state is synchronized with patches
	 * @return True if state has been gathered and is up-to-date
	 */
	bool is_state_synced() const {
		return state_synced_;
	}

	/**
	 * @brief Validate particle data integrity
	 * @return True if data is valid (non-empty, consistent sizes, valid IDs)
	 */
	bool validate_particle_data() const {
		if (global_num_particles_ == 0) {
			return false;
		}
		if (global_particle_data_.size() != global_num_particles_) {
			return false;
		}
		// Check that all arrays have consistent sizes
		if (global_particle_data_.pos.size() != global_num_particles_ ||
			global_particle_data_.global_id.size() != global_num_particles_) {
			return false;
		}
		return true;
	}

	/**
	 * @brief Prepare particle data for DCD output
	 * @return True if data is ready for DCD writing
	 * @note Ensures data is sorted, validated, and synchronized
	 */
	bool prepare_for_dcd_output() {
		if (!state_synced_) {
			LOGWARN("SystemState: Attempting to prepare unsynchronized data for DCD output");
			return false;
		}
		if (!validate_particle_data()) {
			LOGERROR("SystemState: Particle data validation failed before DCD output");
			return false;
		}
		// Ensure sorted order (should already be sorted, but double-check)
		sort_particles_by_id();
		return true;
	}

  private:
	//================================================================================
	// Member Variables - ONLY Runtime State
	//================================================================================
	SimSystem& sim_system_;
	void initialize_system_objects();

	/**
	 * @brief Map of particle IDs to their current memory index.
	 * @note Size = MAX_PARTICLE_ID + padding
	 * @note Value = -1 (Dead/Not Present) or [0...Capacity] (Memory Index)
	 * @note id_to_index_map_[particle_id] = current_memory_index.
	 */
	DeviceBuffer<int2> particle_id_to_index_map_;

	// Particle data (changes every timestep)
	size_t global_num_particles_{0};
	// Global particle state (host-side, ready for I/O)
	HostParticleData global_particle_data_; // Always sorted by ID.
	BondedInteractions bonded_interactions_;

	// Metadata
	bool state_synced_{false}; // Flag indicating if state is up-to-date

	//================================================================================
	// Private Methods
	//================================================================================

	/**
	 * @brief Sort particles by global ID (ensures consistent DCD output order)
	 * @note This is called automatically when data is set from patches
	 */
	void sort_particles_by_id();

	/**
	 * @brief Clean up GPU resources
	 */
	void cleanup_gpu_resources();
};

} // namespace ARBD
