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
 * @brief Manages runtime system state during simulation
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
	 */
	SystemState();

	/**
	 * @brief Destructor - cleans up GPU resources
	 */
	~SystemState();

	/**
	 * @brief Set particle positions (GPU-compatible)
	 * @param positions New particle positions
	 */
	void set_particle_positions(const std::vector<Vector3>& positions);

	/**
	 * @brief Get particle positions (GPU-compatible)
	 * @return Current particle positions
	 */
	std::vector<Vector3> get_particle_positions() const;

	/**
	 * @brief Get number of particles in the system
	 */
	size_t get_num_particles() const {
		return num_particles_;
	}

	//================================================================================
	// System Object Management
	//================================================================================

	/**
	 * @brief Check if system has bonded interactions
	 */
	bool has_bonds() const {
		return has_bonds_;
	}

	/**
	 * @brief Check if system has external forces
	 */
	bool has_external_forces() const {
		return has_external_forces_;
	}

	/**
	 * @brief Check if system has reactions
	 */
	bool has_reactions() const {
		return has_reactions_;
	}

	/**
	 * @brief Gather particle data from all patches into global arrays
	 * @param patch_manager PatchManager containing all patches
	 *
	 * This collects data from local patches (and via MPI if needed)
	 * and assembles into global ordered arrays ready for output.
	 */
	void gather_from_patches(PatchManager& patch_manager);

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
	void initialize_system_objects();
	// Particle data (changes every timestep)
	size_t num_particles_{0};
	// Function index mapping
	std::unordered_map<int, int> angle_function_to_potential_;
	std::unordered_map<int, int> dihedral_function_to_potential_;
	std::unordered_map<int, int> bond_function_to_potential_;

	// System objects that can change during simulation
	bool has_bonds_{false};
	bool has_external_forces_{false};
	bool has_reactions_{false};

	// Global particle state (host-side, ready for I/O)
	std::vector<Vector3> global_positions_;	 // For DCD writing
	std::vector<Vector3> global_momentum_;	 // Optional, for momentum output
	std::vector<int> global_particle_ids_;	 // Particle IDs in global order
	std::vector<int> global_particle_types_; // Particle types in global order

	std::vector<int2> global_bonds_;	   // Bond list in global order
	std::vector<int3> global_angles_;	   // Angle list in global order
	std::vector<int4> global_dihedrals_;   // Dihedral list in global order
	std::vector<int2> global_exclusitons_; // Exclusion list in global order

	// Metadata
	size_t global_num_particles_{0};
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
