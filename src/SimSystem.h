#pragma once
/**
 * @file SimSystem (class) - Immutable System Manager
 * @author Pin-Yi Li <pinyili2@illinois.edu>
 * @brief Simulation system state manager. Manages system state, configuration, and domain
 * decomposition.
 * @note: SimManager handles simulation execution, Owns and manages time-invariant simulation data
 * and coordinates decomposition
 * @param const Configuration (or owned immutable copy after init)
 * @param BoundaryConditions (derived from config, immutable)
 * @param PatchDecomposer and PatchManager (structure can rebalance but concept is static)
 * @param ResourceCollection
 * @param Accessor methods for configuration data (get_temperature, get_cutoff, get_box_size)
 * @param Decomposition coordination (decompose_system, rebalance_system)
 * @note Should NOT contain: Mutable particle positions/velocities, Runtime state that changes every
 * timestep
 * @version 2.0
 * @date 2025-09-09
 *
 * @copyright Copyright (c) 2025
 *
 */

#include "ARBDException.h"
#include "ARBDLogger.h"
#include "Backend/Buffer.h"
#include "Backend/Resource.h"
#include "Configuration.h"
#include "IO/ConfigParser.h"
#include "System/Decomposer.h"
#include "System/PatchManager.h"
#include "System/PeriodicBox.h"
#include "Types/IndexList.h"
#include "Types/Types.h"
#include <memory>
#include <vector>

namespace ARBD {

// Forward declarations
class PatchDecomposer;
class SpatialPatchDecomposer;

class SimSystem {
	friend class PatchDecomposer;
	friend class SpatialPatchDecomposer;

  public:
	/**
	 * @brief Construct simulation system from configuration
	 * @param conf Configuration manager with validated parameters
	 * @param resources Available computational resources
	 */
	SimSystem(const ConfigParser& conf) : config_(conf.get_config()) {
		LOGINFO("SimSystem: Initializing from configuration");

		// Create the chosen decomposer instance (Factory Pattern)
		decomposer_ = create_patch_decomposer(config_.decomposer);

		LOGINFO("SimSystem: Using decomposer '{}'", decomposer_->get_name());
	}

	void decompose_system() {
		if (!decomposer_) {
			throw Exception(ExceptionType::ValueError,
							SourceLocation(),
							"No decomposer has been set");
		}
		LOGINFO("SimSystem: Starting patch decomposition");
		decomposer_->decompose(*this, resources_);
		LOGINFO("SimSystem: patch decomposition completed");
	}

	/**
	 * @brief Rebalances the system after a change in the number of resources
	 */
	void rebalance_system() {
		LOGINFO("SimSystem: Rebalancing system");
		decompose_system(); // For now, just re-decompose
	}

	/**
	 * @brief Initialize particles with positions and types
	 * @param positions Initial particle positions
	 * @param types Initial particle types
	 */
	void initialize_particles(const std::vector<Vector3>& positions, const std::vector<int>& types);

	/**
	 * @brief Build neighbor list for force calculations
	 */
	void build_neighbor_list();

	/**
	 * @brief Get particle positions (GPU-compatible)
	 * @return Current particle positions
	 */
	std::vector<Vector3> get_particle_positions() const;

	//================================================================================
	// Configuration and State Accessors
	//================================================================================

	/**
	 * @brief Get temperature at a specific position (GPU-compatible)
	 * @param position Optional position for grid-based temperature
	 * @return Temperature value at the given position
	 */
	float get_temperature(Vector3 position = {0, 0, 0}) const {
		Temperature temperature = config_.temperature;
		if (temperature.format == Temperature::Format::Value) {
			return temperature.value;
		} else if (temperature.format == Temperature::Format::Grid) {
			return temperature.grid->get_value(position);
		}
		return temperature.value;
	}

	/**
	 * @brief Get cutoff distance for interactions (GPU-compatible)
	 */
	float get_cutoff() const {
		return config_.cutoff.value;
	}

	/**
	 * @brief Get boundary conditions
	 */
	const PeriodicBox& get_boundary_conditions() const {
		return config_.sim_box;
	}

	/**
	 * @brief Get timestep (from configuration)
	 */
	float get_timestep() const {
		return config_.steps.timestep;
	}

	/**
	 * @brief Get number of simulation steps (from configuration)
	 */
	int get_num_steps() const {
		return config_.steps.steps;
	}

	/**
	 * @brief Get box dimensions (from configuration)
	 */
	Vector3 get_box_size() const {
		return Vector3(config_.sim_box.get_box_size().x,
					   config_.sim_box.get_box_size().y,
					   config_.sim_box.get_box_size().z);
	}

	/**
	 * @brief Get complete configuration
	 */
	const Configuration& get_config() const {
		return config_;
	}

	/**
	 * @brief Get available computational resources
	 */
	const ResourceCollection& get_resources() const {
		return resources_;
	}

	//================================================================================
	// System State Queries
	//================================================================================

	/**
	 * @brief Check if system has bonded interactions (GPU-compatible)
	 */
	bool has_bonds() const {
		return config_.objects.bonds.size() > 0;
	}

	/**
	 * @brief Check if system has reactions (GPU-compatible)
	 */
	bool has_reactions() const {
		return config_.has_reaction;
	}

	/**
	 * @brief Get number of particles in the system (GPU-compatible)
	 */
	size_t get_num_particles() const {
		return config_.objects.particles.size();
	}

	//================================================================================
	// System Object Accessors (for SimManager)
	//================================================================================

	/**
	 * @brief Get electric field vector (if any)
	 */
	Vector3 get_electric_field() const {
		// TODO: Add electric field to ARBDObjects or return default
		return Vector3{0, 0, 0};
	}

	/**
	 * @brief Get force grid (if any)
	 */
	const BaseGrid<Vector3>* get_force_grid() const {
		// TODO: Add force grid to ARBDObjects or return nullptr
		return nullptr;
	}

	/**
	 * @brief Get bond list for bonded force calculations
	 */
	const int* get_bond_list() const {
		return nullptr;
	}

	/**
	 * @brief Get bond list size
	 */
	size_t get_bond_list_size() const {
		return config_.objects.bonds.size();
	}

	/**
	 * @brief Get reservoirs for grand canonical simulations
	 */
	const std::vector<Reservoir>& get_reservoirs() const {
		return config_.reservoirs;
	}

	/**
	 * @brief Get output period (from configuration)
	 */
	float get_output_period() const {
		return config_.output_period;
	}

	/**
	 * @brief Get energy output period (from configuration)
	 */
	float get_energy_output_period() const {
		return config_.energy_output_period;
	}

	/**
	 * @brief Get algorithm type (from configuration)
	 */
	DynamicType get_algorithm() const {
		return config_.ParticleDynamicType;
	}

	//================================================================================
	// Patch Management Accessors
	//================================================================================

	/**
	 * @brief Get patch manager for particle decomposition
	 */
	PatchManager* get_patch_manager() const {
		return patch_manager_.get();
	}

	/**
	 * @brief Check if patch manager is initialized
	 */
	bool has_patch_manager() const {
		return patch_manager_ != nullptr;
	}

  private:
	// Configuration management (host-only)
	Configuration config_;
	size_t seed_;

	// Resources and decomposition (host-only)
	ResourceCollection resources_;
	std::unique_ptr<PatchDecomposer> decomposer_;
	std::unique_ptr<PatchManager> patch_manager_;
};

} // namespace ARBD
