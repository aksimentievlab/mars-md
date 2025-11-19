#pragma once

#include "ARBDException.h"
#include "ARBDLogger.h"
#include "Constants.h"
#include "IO/DxIO.h"
#include "IO/Reader.h"
#include "IO/TabulatedReader.h"
#include "Interactions/BondedInteraction.h"
#include "Interactions/Interactions.h"
#include "Objects/ParticleProperties.h"
#include "Objects/RigidBodyProperties.h"
#include "SimParam.h"
#include "System/PeriodicBox.h"
#include "System/Reservoir.h"
#include "Types/Types.h"
#include <array>
#include <map>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace ARBD {

/**
 * @brief Container for initial simulation configuration and state.
 *
 * Stores both static parameters (temperature, timestep, box size) and initial state
 * (particles, bonds, grids, etc.) loaded from Python scripts or .bd text files.
 *
 * @note This is a transient object - SimSystem moves ownership of large data structures
 *       (grids, particle arrays) during initialization. Configuration should be discarded
 *       after SimSystem construction.
 *
 * @note Designed to interact with pybind11 for Python interface and text file parsers.
 */
struct Configuration {
	// Physical parameters
	Temperature temperature{298.15f};
	int replicas{1};
	Length cutoff{10.0f};
	Length pairlist_cutoff{20.0f};
	float salt_concentration{0.15f};  // Salt concentration for solvent, in mol/L
	float dielectric_constant{80.0f}; // Dielectric constant for solvent
	PeriodicBox sim_box;

	DynamicType ParticleDynamicType{DynamicType::Langevin};
	DynamicType RigidBodyDynamicType{DynamicType::Langevin};
	std::unordered_map<std::string, int> functions_id_map{};
	// Simulation control
	SimSteps steps{1e-5f, 1000}; // timestep in ns.

	float output_period{10.0f};
	float energy_output_period{100.0f};
	float neighbor_list_rebuild_period{100.0f};
	size_t global_seed{214};
	std::string output_name{"out"};
	OutputFormat output_format{OutputFormat::DCD};

	// arbd2 additions start
	DecomposerType decomposer{DecomposerType::Spatial};
	DecomposeDirection decompose_direction{DecomposeDirection::Z};

	LongRangeMethod long_range_method{LongRangeMethod::PPPM};
	ThermostatType thermostat{ThermostatType::NVE};
	Pressure pressure{1.0f};
	std::vector<Reservoir> reservoirs;
	bool enable_bond_reactions = false;
	bool enable_particle_reactions = false;
	bool has_reaction = false;
	BarostatType barostat{BarostatType::Isobaric};
	bool calculate_pressure{false};
	float pressure_output_period{100.0f};
	// arbd2 additions end

	std::vector<RigidBodyType> rigid_body_types{};
	std::vector<ParticleType> particle_types{};
	std::vector<ProductPotential> init_product_potentials{};

	// Loaded grids and tabulated functions (moved to SimSystem during initialization)
	std::unordered_map<std::string, int>
		fname_tab_dictionary{}; // Filename -> tabulated function ID
	std::unordered_map<std::string, int> fname_grid_dictionary{}; // Filename -> grid ID
	std::unordered_map<int, std::vector<BaseGrid<float>>>
		grid_id_dictionary{}; // Grid ID -> loaded grids

	// Initial Objects
	std::vector<RigidBody> init_rigid_bodies{}; // init state only
	std::vector<ParticleRead> init_particles{}; // init only
	std::vector<Bond> init_bonds{};				// init only, can be modified during simulation
	std::vector<Angle> init_angles{};			// init only, can also be modified during simulation
	std::vector<Dihedral> init_dihedrals{};		// init only
	std::vector<Exclude> init_exclusions{};		// init only
	std::vector<Restraint> restraints{};		// Can be released during simulation

	bool enable_smd{false};

	// DCD parsing features
	int dcd_stride{1};
	bool parse_dcd_mode{false};
	std::string dcd_input_file{};

	// Python-friendly accessors
	void set_temperature(float temp) {
		temperature.format = Temperature::Format::Value;
		temperature.value = temp;
	}

	void set_timestep(float dt) {
		steps.timestep = dt;
	}
	void set_num_steps(int n) {
		steps.steps = n;
	}

	void set_salt_concentration(float salt_concentration) {
		this->salt_concentration = salt_concentration;
	}
	/**
	 * @note It's recommnded to set the box size so that z is the largest dimension.
	 */
	void set_box_size(float x, float y, float z) {
		sim_box.set_box_size(Vector3(x, y, z));
	}
	void set_periodicity(bool px, bool py, bool pz) {
		sim_box.set_periodicity(px, py, pz);
	}

	// Validation
	bool is_valid() const {
		validate_physical_parameters();
		validate_method_parameters();
		validate_output_parameters();
		return true;
	}
	// Validation helpers
	void validate_physical_parameters() const;
	void validate_method_parameters() const;
	void validate_output_parameters() const;

	// Product potential management
	void add_product_potential(const ProductPotential& pp) {
		init_product_potentials.push_back(pp);
	}

	void add_bond_angle_product_potential(int i,
										  int j,
										  int k,
										  int l,
										  int angle1_idx,
										  int bond_idx,
										  int angle2_idx,
										  const std::string& name = "BondAngle") {
		init_product_potentials.push_back(
			ProductPotential::from_bond_angle(i, j, k, l, angle1_idx, bond_idx, angle2_idx, name));
	}

	void add_angle_angle_product_potential(int i,
										   int j,
										   int k,
										   int l,
										   int angle1_idx,
										   int angle2_idx,
										   const std::string& name = "AngleAngle") {
		init_product_potentials.push_back(ProductPotential(i,
														   j,
														   k,
														   l,
														   ProductPotentialType::AngleAngle,
														   angle1_idx,
														   -1,
														   angle2_idx,
														   name));
	}

	void add_bond_bond_product_potential(int i,
										 int j,
										 int k,
										 int bond1_idx,
										 int bond2_idx,
										 const std::string& name = "BondBond") {
		init_product_potentials.push_back(ProductPotential(i,
														   j,
														   k,
														   -1,
														   ProductPotentialType::BondBond,
														   bond1_idx,
														   bond2_idx,
														   -1,
														   name));
	}

	size_t get_num_product_potentials() const {
		return init_product_potentials.size();
	}

	const std::vector<ProductPotential>& get_product_potentials() const {
		return init_product_potentials;
	}
};

} // namespace ARBD
