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
#include "Objects/ARBDObjects.h"
#include "Types/BaseGrid.h"
#include "Types/Types.h"
#include <memory>
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
	 * @brief Initialize system objects (bonds, grids, fields) from configuration
	 */
	void initialize_system_objects();

	/**
	 * @brief Get electric field vector (if any)
	 */
	Vector3 get_electric_field() const {
		return electric_field_;
	}

	/**
	 * @brief Set electric field
	 * @param field Electric field vector
	 */
	void set_electric_field(const Vector3& field) {
		electric_field_ = field;
	}

	/**
	 * @brief Get force grid (if any)
	 */
	const BaseGrid<Vector3>* get_force_grid() const {
		return force_grid_;
	}

	/**
	 * @brief Set force grid
	 * @param grid Force grid pointer
	 */
	void set_force_grid(BaseGrid<Vector3>* grid) {
		force_grid_ = grid;
	}

	/**
	 * @brief Get bond list for bonded force calculations
	 */
	const int* get_bond_list() const {
		return bond_list_;
	}

	/**
	 * @brief Get bond list size
	 */
	size_t get_bond_list_size() const {
		return bond_list_size_;
	}

	/**
	 * @brief Set bond list
	 * @param bonds Bond list data
	 * @param size Number of bonds
	 */
	void set_bond_list(const int* bonds, size_t size) {
		bond_list_ = bonds;
		bond_list_size_ = size;
	}

	//================================================================================
	// System State Queries
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
	 * @brief Get temperature grid (if any)
	 */
	const BaseGrid<float>* get_temperature_grid() const {
		return temperature_grid_;
	}

	/**
	 * @brief Set temperature grid
	 * @param grid Temperature grid pointer
	 */
	void set_temperature_grid(BaseGrid<float>* grid) {
		temperature_grid_ = grid;
	}

	//================================================================================
	// Runtime State Queries
	//================================================================================

	/**
	 * @brief Get current number of particles
	 */
	size_t get_num_particles() const {
		return num_particles_;
	}

  private:
	//================================================================================
	// Member Variables - ONLY Runtime State
	//================================================================================

	// Particle data (changes every timestep)
	size_t num_particles_{0};
	std::vector<Vector3> particle_positions_;
	std::vector<Vector3> particle_velocities_;
	std::vector<int> particle_types_;

	// System objects that can change during simulation
	bool has_bonds_{false};
	bool has_external_forces_{false};
	bool has_reactions_{false};
	Vector3 electric_field_{0, 0, 0};
	BaseGrid<float>* temperature_grid_{nullptr}; // Device pointer
	BaseGrid<Vector3>* force_grid_{nullptr};	 // Device pointer
	int* bond_list_{nullptr};					 // Device pointer
	size_t bond_list_size_{0};

	//================================================================================
	// Private Methods
	//================================================================================

	/**
	 * @brief Clean up GPU resources
	 */
	void cleanup_gpu_resources();
};

} // namespace ARBD
