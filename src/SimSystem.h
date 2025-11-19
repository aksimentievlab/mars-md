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
#include "IO/ConfigParser.h"
#include "System/Decomposer.h"
#include "System/PatchManager.h"
#include "System/PeriodicBox.h"
#include "System/SystemState.h"
#include "Types/IndexList.h"
#include "Types/Types.h"
#include <memory>
#include <vector>

namespace ARBD {

class SimSystem {

  public:
	/**
	 * @brief Construct simulation system from configuration
	 * @param conf Configuration manager with validated parameters
	 * @param resources Available computational resources
	 */
	SimSystem(ConfigParser& conf) : config_(conf.get_config()) {
		LOGINFO("SimSystem: Initializing from configuration");

		// Move grids from configuration (configuration will be discarded after init)
		grid_id_dictionary_ = std::move(conf.get_mutable_config().grid_id_dictionary);
		fname_grid_dictionary_ = std::move(conf.get_mutable_config().fname_grid_dictionary);
		fname_tab_dictionary_ = std::move(conf.get_mutable_config().fname_tab_dictionary);

		LOGINFO("SimSystem: Loaded {} grid sets from configuration", grid_id_dictionary_.size());

		// Create the chosen decomposer instance (Factory Pattern)
		decomposer_ = create_patch_decomposer(config_.decomposer);

		LOGINFO("SimSystem: Using decomposer '{}'", decomposer_->get_name());
	}

	/**
	 * @brief Perform domain decomposition (should only be called once during setup)
	 *
	 * This method:
	 * 1. Calls the decomposer to compute a decomposition plan
	 * 2. Creates and initializes PatchManager from the plan
	 * 3. Stores the PatchManager for runtime use
	 */
	void decompose_system() {
		if (!decomposer_) {
			throw Exception(ExceptionType::ValueError,
							SourceLocation(),
							"No decomposer has been set");
		}

		if (patch_manager_) {
			LOGWARN("SimSystem: Decomposition already performed. Use rebalance_system() to "
					"re-decompose.");
			return;
		}

		LOGINFO("SimSystem: Starting patch decomposition");

		// Step 1: Compute decomposition plan (Decomposer responsibility)
		DecompositionPlan plan = decomposer_->decompose(*this);

		// Step 2: Create PatchManager from plan (PatchManager responsibility)
		auto patch_manager = std::make_unique<PatchManager>(*this);

		// Get estimated particles per patch (could be refined based on actual particle
		// distribution)
		idx_t estimated_particles = 1024; // Default estimate, could be computed from system
		const Length cutoff = Length(get_cutoff());

		// Initialize PatchManager with the decomposition plan
		patch_manager->initialize_from_plan(plan, estimated_particles, cutoff);

		// Step 3: Store PatchManager (SimSystem ownership)
		patch_manager_ = std::move(patch_manager);

		LOGINFO("SimSystem: Patch decomposition completed - {} patches created",
				plan.total_patches());
	}

	/**
	 * @brief Rebalances the system after a change in the number of resources
	 */
	void rebalance_system() {
		LOGINFO("SimSystem: Rebalancing system");
		decompose_system(); // For now, just re-decompose
	}

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
	const std::vector<Resource>& get_resources() const {
		return resources_;
	}

	void add_resource(Resource resource) {
		resources_.push_back(resource);
	}

	void add_resources(std::vector<Resource>& resources) {
		resources_ = resources;
	}
	//================================================================================
	// System State Queries
	//================================================================================

	/**
	 * @brief Check if system has reactions (GPU-compatible)
	 */
	bool has_reactions() const {
		return config_.has_reaction;
	}

	//================================================================================
	// System Object Accessors (for SimManager)
	//================================================================================

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
		return config_.init_bonds.size();
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
	DynamicType get_particle_algorithm() const {
		return config_.ParticleDynamicType;
	}

	/**
	 * @brief Get rigid body algorithm type (from configuration)
	 */
	DynamicType get_rigid_body_algorithm() const {
		return config_.RigidBodyDynamicType;
	}

	//================================================================================
	// Patch Management Accessors
	//================================================================================
	// Design: SimSystem owns PatchManager as part of system structure
	// - PatchManager defines the domain decomposition (spatial structure)
	// - Decomposer initializes it during decompose()
	// - SimManager accesses it for runtime coordination via get_patch_manager()

	/**
	 * @brief Get patch manager for domain decomposition
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

	/**
	 * @brief Initialize patch manager (called by decomposer)
	 * @param manager Configured patch manager from decomposer
	 */
	void set_patch_manager(std::unique_ptr<PatchManager> manager) {
		if (patch_manager_) {
			LOGWARN("SimSystem: Replacing existing PatchManager during redecomposition");
		}
		patch_manager_ = std::move(manager);
		LOGINFO("SimSystem: PatchManager installed");
	}

	/**
	 * @brief Get the patch decomposer for this system
	 */
	PatchDecomposer* get_decomposer() const {
		return decomposer_.get();
	}

	//================================================================================
	// Grid and Tabulated Function Accessors
	//================================================================================

	/**
	 * @brief Get grid by ID
	 * @param grid_id Grid identifier
	 * @return Pointer to grid vector, or nullptr if not found
	 */
	const std::vector<BaseGrid<float>>* get_grids_by_id(int grid_id) const {
		auto it = grid_id_dictionary_.find(grid_id);
		return (it != grid_id_dictionary_.end()) ? &it->second : nullptr;
	}

	/**
	 * @brief Get grid ID by filename
	 * @param filename Grid filename
	 * @return Grid ID, or -1 if not found
	 */
	int get_grid_id_by_filename(const std::string& filename) const {
		auto it = fname_grid_dictionary_.find(filename);
		return (it != fname_grid_dictionary_.end()) ? it->second : -1;
	}

	/**
	 * @brief Get tabulated function ID by filename
	 * @param filename Tabulated function filename
	 * @return Function ID, or -1 if not found
	 */
	int get_tabulated_function_id(const std::string& filename) const {
		auto it = fname_tab_dictionary_.find(filename);
		return (it != fname_tab_dictionary_.end()) ? it->second : -1;
	}

  private:
	// Configuration management (host-only)
	Configuration config_;
	BaseGrid<Vector3>* force_grid_;

	// Loaded grids and tabulated functions (moved from Configuration)
	std::unordered_map<std::string, int> fname_tab_dictionary_; // Filename -> tabulated function ID
	std::unordered_map<std::string, int> fname_grid_dictionary_; // Filename -> grid ID
	std::unordered_map<int, std::vector<BaseGrid<float>>>
		grid_id_dictionary_; // Grid ID -> loaded grids

	// Resources and decomposition (host-only)
	std::vector<Resource> resources_;
	std::unique_ptr<PatchDecomposer> decomposer_;
	std::unique_ptr<PatchManager> patch_manager_; // Domain decomposition structure
};

} // namespace ARBD
