#pragma once
/**
 * @file SimSystem (class) - Immutable System Manager
 * @author Pin-Yi Li <pinyili2@illinois.edu>
 * @brief Simulation system state manager. Manages system state, configuration, and domain
 * decomposition.
 * @note: SimManager handles simulation execution, Owns and manages time-invariant simulation data
 * and coordinates decomposition
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
#include "IO/DxIO.h"
#include "Objects/Grid.h"
#include "Objects/RigidBodyProperties.h"
#include "Objects/Tables.h"
#include "System/Decomposers.h"
#include "System/PatchManager.h"
#include "System/PeriodicBox.h"
#include "Types/IndexList.h"
#include "Types/Types.h"
#include <memory>
#include <vector>

namespace ARBD {

class SystemState;

class SimSystem {

  public:
	/**
	 * @brief Construct simulation system from configuration
	 * @param conf Configuration manager with validated parameters
	 * @param resources Available computational resources
	 */
	SimSystem(const std::vector<Resource>& resources) : resources_(resources) {
		grid_manager_.set_resources(&resources_);
		tables_registry_.set_resources(&resources_);
	}

	void set_temperature(float temp) {
		temperature_.format = Temperature::Format::Value;
		temperature_.value = temp;
	}
	void set_base_seed(size_t seed) {
		LOGINFO("SimSystem: Setting base seed to {}", seed);
		global_seed_ = seed;
		LOGINFO("SimSystem: Base seed set to {}", global_seed_);
	}

	size_t get_base_seed() const {
		return global_seed_;
	}

	void set_temperature(const BaseGrid<float>& grid) {
		temperature_ = Temperature(grid);
	}

	void set_cutoff(Length cutoff) {
		this->cutoff_ = cutoff;
	}

	void set_pairlist_cutoff(Length pairlist_cutoff) {
		this->pairlist_cutoff_ = pairlist_cutoff;
	}

	void set_timestep(float dt) {
		steps_.timestep = dt;
	}
	void set_num_steps(int n) {
		steps_.steps = n;
	}

	void set_salt_concentration(float salt_concentration) {
		this->salt_concentration_ = salt_concentration;
	}

	void set_neighbor_list_rebuild_period(float period) {
		neighbor_list_rebuild_period = period;
	}

	void set_output_period(float period) {
		output_period_ = period;
	}

	void set_energy_output_period(float period) {
		energy_output_period_ = period;
	}

	void set_output_name(const std::string& name) {
		output_name = name;
	}

	void set_output_format(OutputFormat format) {
		output_format_ = format;
	}

	void set_decomposer_type(DecomposerType type,
							 DecomposeDirection direction = DecomposeDirection::Z) {
		decomposer_type_ = type;
		decompose_direction_ = direction;
		decomposer_ = create_patch_decomposer(decomposer_type_);
	}

	void set_long_range_method(LongRangeMethod method) {
		long_range_method_ = method;
	}

	void set_particle_integrator_type(IntegratorType type) {
		particle_integrator_type_ = type;
	}

	void set_rigid_body_integrator_type(IntegratorType type) {
		rigid_body_integrator_type_ = type;
	}
	void add_particle_type(const ParticleType& type) {
		particle_types_.push_back(type);
	}
	void add_rigid_body_type(const RigidBodyType& type) {
		rigid_body_types_.push_back(type);
	}
	void add_reservoir(const Reservoir& reservoir) {
		reservoirs_.push_back(reservoir);
	}
	void set_particle_types(const std::vector<ParticleType>& types) {
		particle_types_.clear();
		for (const auto& type : types) {
			particle_types_.push_back(type);
		}
	}
	void set_rigid_body_types(const std::vector<RigidBodyType>& types) {
		rigid_body_types_.clear();
		for (const auto& type : types) {
			rigid_body_types_.push_back(type);
		}
	}
	void set_reservoirs(const std::vector<Reservoir>& reservoirs) {
		reservoirs_.clear();
		for (const auto& reservoir : reservoirs) {
			reservoirs_.push_back(reservoir);
		}
	}

	/**
	 * @note It's recommnded to set the box size so that z is the largest dimension.
	 */
	void set_box_size(float x, float y, float z) {
		sim_box_.set_box_size(Vector3(x, y, z));
	}
	void set_origin(float x, float y, float z) {
		sim_box_.set_origin(Vector3(x, y, z));
	}
	void set_periodicity(bool px, bool py, bool pz) {
		sim_box_.set_periodicity(px, py, pz);
	}
	void set_estimated_particles(idx_t estimated_particles) {
		estimated_particles_ = estimated_particles;
	}

	/**
	 * @brief Perform domain decomposition (should only be called once during setup)
	 *
	 * This method:
	 * 1. Calls the decomposer to compute a decomposition plan
	 * 2. Creates and initializes PatchManager from the plan
	 * 3. Stores the PatchManager for runtime use
	 */
	void decompose_system(SystemState& state) {
		if (patch_manager_) {
			LOGWARN("SimSystem: Decomposition already performed. Use rebalance_system() to "
					"re-decompose.");
			return;
		}

		// Single resource optimization - skip decomposer for single GPU/CPU
		if (resources_.size() == 1) {
			LOGINFO(
				"SimSystem: Creating single patch for single resource (no decomposition needed)");
			create_single_patch_manager(state);
			return;
		}

		// Multi-resource decomposition path
		if (!decomposer_) {
			throw Exception(ExceptionType::ValueError,
							SourceLocation(),
							"No decomposer has been set for multi-resource system");
		}

		LOGINFO("SimSystem: Starting patch decomposition for {} resources", resources_.size());

		// Step 1: Compute decomposition plan (Decomposer responsibility)
		DecompositionPlan plan = decomposer_->decompose(*this, state);

		// Step 2: Create PatchManager from plan (PatchManager responsibility)
		auto patch_manager = std::make_unique<PatchManager>(*this);

		// Get estimated particles per patch (could be refined based on actual particle
		// distribution)

		// Initialize PatchManager with the decomposition plan
		patch_manager->initialize_from_plan(plan, estimated_particles_);

		// Step 3: Store PatchManager (SimSystem ownership)
		patch_manager_ = std::move(patch_manager);

		LOGINFO("SimSystem: Patch decomposition completed - {} patches created",
				plan.total_patches());
	}

	/**
	 * @brief Rebalances the system after a change in the number of resources
	 */
	void rebalance_system(SystemState& state) {
		LOGINFO("SimSystem: Rebalancing system");
		decompose_system(state); // For now, just re-decompose
	}

	/**
	 * @brief Get temperature at a specific position.
	 * @param position Optional position for grid-based temperature
	 * @return Temperature value at the given position
	 */
	float get_temperature(Vector3 position = {0, 0, 0}) const {
		if (temperature_.format == Temperature::Format::Value) {
			return temperature_.value;
		} else if (temperature_.format == Temperature::Format::Grid) {
			return temperature_.temperature_grid.get_value(position);
		}
		return temperature_.value;
	}

	/**
	 * @brief Get cutoff distance for interactions (GPU-compatible)
	 */
	Length get_cutoff() const {
		return cutoff_;
	}

	/**
	 * @brief Get boundary conditions
	 */
	const PeriodicBox& get_boundary_conditions() const {
		return sim_box_;
	}

	/**
	 * @brief Get timestep (from configuration)
	 */
	float get_timestep() const {
		return steps_.timestep;
	}

	/**
	 * @brief Get number of simulation steps (from configuration)
	 */
	int get_num_steps() const {
		return steps_.steps;
	}

	/**
	 * @brief Get box dimensions (from configuration)
	 */
	Vector3 get_box_size() const {
		return Vector3(sim_box_.get_box_size().x,
					   sim_box_.get_box_size().y,
					   sim_box_.get_box_size().z);
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

	//================================================================================
	// System State Queries
	//================================================================================

	/**
	 * @brief Check if system has reactions (GPU-compatible)
	 */
	bool has_reaction() const {
		return enable_particle_reactions_ || enable_bond_reactions_;
	};

	//================================================================================
	// System Object Accessors (for SimManager)
	//================================================================================

	/**
	 * @brief Get reservoirs for grand canonical simulations
	 */
	const std::vector<Reservoir>& get_reservoirs() const {
		return reservoirs_;
	}

	/**
	 * @brief Get output period (from configuration)
	 */
	float get_output_period() const {
		return output_period_;
	}

	/**
	 * @brief Get energy output period (from configuration)
	 */
	float get_energy_output_period() const {
		return energy_output_period_;
	}

	/**
	 * @brief Get output name
	 */
	const std::string& get_output_name() const {
		return output_name;
	}

	/**
	 * @brief Get output format
	 */
	OutputFormat get_output_format() const {
		return output_format_;
	}

	/**
	 * @brief Get decomposer type
	 */
	DecomposerType get_decomposer_type() const {
		return decomposer_type_;
	}

	/**
	 * @brief Get long range method
	 */
	LongRangeMethod get_long_range_method() const {
		return long_range_method_;
	}

	/**
	 * @brief Get neighbor list rebuild period
	 */
	float get_neighbor_list_rebuild_period() const {
		return neighbor_list_rebuild_period;
	}

	/**
	 * @brief Get temperature struct (for validation)
	 */
	const Temperature& get_temperature_struct() const {
		return temperature_;
	}

	/**
	 * @brief Get steps struct (for validation)
	 */
	const SimSteps& get_steps() const {
		return steps_;
	}

	/**
	 * @brief Get algorithm type (from configuration)
	 */
	IntegratorType get_particle_algorithm() const {
		return particle_integrator_type_;
	}

	// Type definitions (time-invariant configuration)
	std::vector<ParticleType>& get_particle_types() {
		return particle_types_;
	}

	const std::vector<ParticleType>& get_particle_types() const {
		return particle_types_;
	}

	std::vector<RigidBodyType>& get_rigid_body_types() {
		return rigid_body_types_;
	}

	const std::vector<RigidBodyType>& get_rigid_body_types() const {
		return rigid_body_types_;
	};

	/**
	 * @brief Get rigid body algorithm type (from configuration)
	 */
	IntegratorType get_rigid_body_algorithm() const {
		return rigid_body_integrator_type_;
	}

	//================================================================================
	// Runtime Lookup Tables (populated by SimManager during initialization)
	//================================================================================
	// These dictionaries map filenames/function names to IDs for fast lookup
	// during simulation. They are empty during SimSystem construction and
	// populated by SimManager after loading grids and potentials.

	/**
	 * @brief Get filename -> tabulated function ID mapping
	 */
	const std::unordered_map<std::string, int>& get_fname_bond_dictionary() const {
		return tables_registry_.get_bond_name_to_idx();
	}
	const std::unordered_map<std::string, int>& get_fname_angle_dictionary() const {
		return tables_registry_.get_angle_name_to_idx();
	}
	const std::unordered_map<std::string, int>& get_fname_dihedral_dictionary() const {
		return tables_registry_.get_dihedral_name_to_idx();
	}

	/**
	 * @brief Get GridManager for unified grid management
	 */
	GridManager& get_grid_manager() {
		return grid_manager_;
	}

	const GridManager& get_grid_manager() const {
		return grid_manager_;
	}

	/**
	 * @brief Get TablesRegistry for tabulated function management
	 */
	TablesRegistry& get_tables_registry() {
		return tables_registry_;
	}

	const TablesRegistry& get_tables_registry() const {
		return tables_registry_;
	}

	/**
	 * @brief Get NonBondedInteractions for non-bonded interaction management
	 */
	NonBondedInteractions& get_nonbonded_interactions() {
		return nonbonded_interactions_;
	}

	const NonBondedInteractions& get_nonbonded_interactions() const {
		return nonbonded_interactions_;
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

	/**
	 * @brief Get pairlist cutoff distance for interactions (GPU-compatible)
	 */
	Length get_pairlist_cutoff() const {
		return pairlist_cutoff_;
	}
	//================================================================================
	// Grid and Tabulated Function Accessors
	//================================================================================

	/**
	 * @brief Get tabulated function ID by filename for python interface.
	 * @param filename Tabulated function filename
	 * @return Function ID, or -1 if not found
	 */
	int get_tabulated_function_id(const std::string& filename, BondedPotentialType type) const {
		switch (type) {
		case BondedPotentialType::BOND:
			return tables_registry_.get_bond_name_to_idx().find(filename)->second;
		case BondedPotentialType::ANGLE:
			return tables_registry_.get_angle_name_to_idx().find(filename)->second;
		case BondedPotentialType::DIHEDRAL:
			return tables_registry_.get_dihedral_name_to_idx().find(filename)->second;
		default:
			throw Exception(ExceptionType::ValueError,
							SourceLocation(),
							"Invalid bonded potential type: {}",
							type);
		}
	}

	bool has_bond_reactions() const {
		return enable_bond_reactions_;
	}

	bool has_particle_reactions() const {
		return enable_particle_reactions_;
	}

	/**
	 * @brief Validate system configuration
	 */
	bool is_valid() const {
		validate_physical_parameters();
		validate_method_parameters();
		validate_output_parameters();
		return true;
	}

	void validate_physical_parameters() const;
	void validate_method_parameters() const;
	void validate_output_parameters() const;

	void assign_particle_type_ids() {
		LOGINFO("SimSystem: Assigning particle type IDs");
		particle_type_names_.resize(particle_types_.size());
		int particle_type_id_counter = 0;
		for (auto& particle_type : particle_types_) {
			particle_type.id = particle_type_id_counter;
			particle_type_names_[particle_type_id_counter] = particle_type.name;
			// pmf_grid_names is parallel to pmf_grids (one gridFile entry each).
			for (size_t g = 0; g < particle_type.pmf_grids.size(); ++g) {
				const std::string& gname = g < particle_type.pmf_grid_names.size()
											   ? particle_type.pmf_grid_names[g]
											   : std::string{};
				particle_type.pmf_grids[g].grid_id =
					get_grid_manager().get_grid_key(gname).grid_id;
			}
			particle_type.diffusion_grid_id =
				get_grid_manager().get_grid_key(particle_type.diffusion_grid_name).grid_id;
			particle_type.force_grid_id = {
				get_grid_manager().get_grid_key(particle_type.force_grid_names[0]).grid_id,
				get_grid_manager().get_grid_key(particle_type.force_grid_names[1]).grid_id,
				get_grid_manager().get_grid_key(particle_type.force_grid_names[2]).grid_id};
			particle_type_id_counter++;
		}
		LOGINFO("SimSystem: Particle type IDs assigned: total {} types",
				particle_type_names_.size());
	}

	void assign_rigid_body_type_ids() {
		LOGINFO("SimSystem: Assigning rigid body type IDs");
		rigid_body_type_names_.resize(rigid_body_types_.size());
		int rigid_body_id_counter = 0;
		for (auto& rigid_body_type : rigid_body_types_) {
			rigid_body_type.id = rigid_body_id_counter;
			rigid_body_type_names_[rigid_body_id_counter] = rigid_body_type.name;
			rigid_body_id_counter++;
		}
		LOGINFO("SimSystem: Rigid body type IDs assigned: total {} types",
				rigid_body_type_names_.size());
	}

	int get_particle_type_id(const std::string& name) const {
		return particle_type_name_to_id_.find(name)->second;
	}

	int get_rigid_body_type_id(const std::string& name) const {
		return rigid_body_type_name_to_id_.find(name)->second;
	}

	/**
	 * @brief Build name to ID maps for particle and rigid body types
	 */
	void build_name_to_id_maps() {
		for (const auto& particle_type : particle_types_) {
			particle_type_name_to_id_[particle_type.name] = particle_type.id;
		}
		for (const auto& rigid_body_type : rigid_body_types_) {
			rigid_body_type_name_to_id_[rigid_body_type.name] = rigid_body_type.id;
		}
	}

  private:
	/**
	 * @brief Create simplified patch manager for single resource systems
	 * @param state System state containing particles
	 */
	void create_single_patch_manager(SystemState& state);

	// Configuration management (host-only)
	Temperature temperature_{298.15f};
	int replicas_{1};
	Length cutoff_{10.0f};
	Length pairlist_cutoff_{20.0f};
	float salt_concentration_{0.15f};  // Salt concentration for solvent, in mol/L
	float dielectric_constant_{80.0f}; // Dielectric constant for solvent
	PeriodicBox sim_box_{Vector3(100.0f, 100.0f, 200.0f), true, true, true};
	idx_t estimated_particles_{1024000};
	IntegratorType particle_integrator_type_{IntegratorType::Langevin};
	IntegratorType rigid_body_integrator_type_{IntegratorType::Langevin};
	// Simulation control
	SimSteps steps_{1e-5f, 1000}; // timestep in ns.

	int output_period_{10};				   // output period in steps
	int energy_output_period_{100};		   // energy output period in steps
	int neighbor_list_rebuild_period{1000}; // neighbor list rebuild period in steps
	int rb_update_period_{1};			   // rigid body update period in steps

	size_t global_seed_{214};
	std::string output_name{"out"};
	OutputFormat output_format_{OutputFormat::DCD};

	LongRangeMethod long_range_method_{LongRangeMethod::None};
	Pressure pressure_{1.0f};
	BarostatType barostat_{BarostatType::None};
	// bool calculate_pressure_{false};
	float pressure_output_period_{100.0f};

	bool enable_smd{false};

	bool enable_bond_reactions_{false};
	bool enable_particle_reactions_{false};

	// DCD parsing features
	int dcd_stride{1};
	bool parse_dcd_mode{false};
	std::string dcd_input_file{};

	// Type definitions (time-invariant configuration)
	std::vector<RigidBodyType> rigid_body_types_{};
	std::vector<ParticleType> particle_types_{};
	std::vector<Reservoir> reservoirs_{};
	std::vector<std::string> particle_type_names_{};
	std::vector<std::string> rigid_body_type_names_{};
	std::unordered_map<std::string, int> particle_type_name_to_id_{};
	std::unordered_map<std::string, int> rigid_body_type_name_to_id_{};

	DecomposerType decomposer_type_{DecomposerType::Spatial};
	DecomposeDirection decompose_direction_{DecomposeDirection::Z};
	// Resources and decomposition (host-only)
	std::vector<Resource> resources_;
	std::unique_ptr<PatchManager> patch_manager_; // Domain decomposition structure
	std::unique_ptr<PatchDecomposer> decomposer_;
	GridManager grid_manager_;
	TablesRegistry tables_registry_;
	NonBondedInteractions nonbonded_interactions_;
};

} // namespace ARBD
