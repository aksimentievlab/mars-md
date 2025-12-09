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
#include "System/PatchManager.h"
#include "Types/BaseGrid.h"
#include "Types/Types.h"
#include <memory>
#include <unordered_map>
#include <vector>

namespace ARBD {

/**
 * @brief Manages runtime, mutable system state during simulation
 *
 * SystemState handles ONLY changing runtime state:
 * - Particle positions, velocities, and types (change every timestep)
 * - System objects that change during simulation (bonds, grids, fields)
 * - Runtime flags and counters
 *
 * Static configuration data (temperature, cutoff, box_size) is accessed
 * directly from Configuration when needed - no duplication here.
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
		global_positions_.resize(particles.size());
		global_momentum_.resize(particles.size());
		global_particle_ids_.resize(particles.size());
		for (const auto& particle : particles) {
			global_positions_[particle.id] = particle.position;
			global_momentum_[particle.id] = particle.momentum;
			global_particle_ids_[particle.id] = particle.id;
		}
		global_num_particles_ = global_positions_.size();
	}

	/**
	 * @brief Clear global particle arrays (called before gathering new data)
	 */
	void clear_global_arrays() {
		global_positions_.clear();
		global_momentum_.clear();
		global_particle_ids_.clear();
		global_num_particles_ = 0;
		state_synced_ = false;
	}

	/**
	 * @brief Add particle data to global arrays (called by SimManager during gathering)
	 * @param positions Particle positions
	 * @param momenta Particle momenta
	 * @param ids Particle IDs
	 */
	void add_particle_data(const std::vector<Vector3>& positions,
						   const std::vector<Vector3>& momenta,
						   const std::vector<int>& ids) {
		global_positions_.insert(global_positions_.end(), positions.begin(), positions.end());
		global_momentum_.insert(global_momentum_.end(), momenta.begin(), momenta.end());
		global_particle_ids_.insert(global_particle_ids_.end(), ids.begin(), ids.end());
		global_num_particles_ = global_positions_.size();
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
		return global_particle_ids_;
	}

	/**
	 * @brief Get global particle positions (ready for DCD writing)
	 */
	const std::vector<Vector3>& get_global_positions() const {
		return global_positions_;
	}

	/**
	 * @brief Get global particle velocities
	 */
	const std::vector<Vector3>& get_global_momentum() const {
		return global_momentum_;
	}

	/**
	 * @brief Get total number of particles across all patches
	 */
	size_t get_global_num_particles() const {
		return global_num_particles_;
	}

	/**
	 * @brief Check if global state is synchronized
	 */
	bool is_state_synced() const {
		return state_synced_;
	}

  private:
	//================================================================================
	// Member Variables - ONLY Runtime State
	//================================================================================
	SimSystem& sim_system_;
	void initialize_system_objects();

	// Particle data (changes every timestep)
	size_t global_num_particles_{0};
	// Function index mapping

	// Global particle state (host-side, ready for I/O)
	std::vector<Vector3> global_positions_; // For DCD writing
	std::vector<Vector3> global_momentum_;	// Optional, for momentum output
	std::vector<int> global_particle_ids_;	// Particle IDs in global order

	BondedInteractions bonded_interactions_;

	// Metadata
	bool state_synced_{false}; // Flag indicating if state is up-to-date

	//================================================================================
	// Private Methods
	//================================================================================

	/**
	 * @brief Clean up GPU resources
	 */
	void cleanup_gpu_resources();
};

} // namespace ARBD
