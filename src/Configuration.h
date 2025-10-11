#pragma once

#include "ARBDException.h"
#include "ARBDLogger.h"
#include "IO/Reader.h"
#include "Objects/ARBDObjects.h"
#include "SimParam.h"
#include "System/Reservoir.h"
#include "Types/Types.h"
#include <array>
#include <string>
#include <string_view>
#include <vector>

namespace ARBD {

/**
 * @brief Main configuration structure containing all simulation parameters
 *
 * This structure is designed to be:
 * - Python-bindable via pybind11
 * - Easily serializable/deserializable
 * - Validation-friendly
 * - Default-constructible with sensible values
 */

struct Configuration {
	// Physical parameters
	Temperature temperature{298.15f};

	Length cutoff{10.0f};
	Length pairlist_distance{20.0f};
	std::array<float, 3> box_lengths{5000.0f, 5000.0f, 5000.0f};

	// Method selection
	Periodicity periodicity{Periodicity::AllPeriodic};
	DecomposerType decomposer{DecomposerType::Spatial};
	LongRangeMethod long_range_method{LongRangeMethod::PPPM};
	DynamicType ParticleDynamicType{DynamicType::Langevin};
	DynamicType RigidBodyDynamicType{DynamicType::Langevin};
	std::map<std::string, int> functions_id_map;
	// Simulation control
	SimSteps steps{1e-5f, 1000};

	ThermostatType thermostat{ThermostatType::Langevin};
	BarostatType barostat{BarostatType::Isobaric};
	float output_period{100.0f};
	float energy_output_period{100.0f};
	float decomp_period{10.0f};

	std::string output_name{"out"};
	OutputFormat output_format{OutputFormat::DCD};

	Pressure pressure{1.0f};
	std::vector<Reservoir> reservoirs;
	bool has_reaction = false;
	int replicas{1};
	ARBDObjects objects;

	// Python-friendly accessors
	void set_temperature(float temp) {
		temperature.format = Temperature::Format::Value;
		temperature.value = temp;
	}

	/**
	 * @note It's recommnded to set the box size so that x is the largest dimension.
	 */
	void set_box_size(float x, float y, float z) {
		box_lengths = {x, y, z};
	}

	void set_timestep(float dt) {
		steps.timestep = dt;
	}
	void set_num_steps(int n) {
		steps.steps = n;
	}

	// Validation
	bool is_valid() const {
		return temperature.value > 0.0f && cutoff.value > 0.0f && steps.timestep > 0.0f &&
			   steps.steps > 0;
	}
};

//================================================================================
// Configuration Manager and Parser
//================================================================================

/**
 * @brief Configuration manager with file I/O and validation
 *
 * This class handles:
 * - Loading configuration from files
 * - Validation of parameters
 * - Conversion to runtime formats
 * - Python bindings
 */
class SimConf {
  public:
	SimConf() = default;

	/**
	 * @brief Construct from configuration file
	 * @param file_name Path to configuration file
	 * @throws ARBD::Exception on I/O or validation error
	 */
	explicit SimConf(std::string_view file_name);

	/**
	 * @brief Construct from Configuration struct (Python-friendly)
	 * @param config Configuration structure
	 */
	SimConf(Configuration config);

	/**
	 * @brief Parse configuration from file
	 * @param file_name Path to configuration file
	 * @throws ARBD::Exception on I/O or validation error
	 */
	void parse_file(std::string_view file_name);

	/**
	 * @brief Get the configuration structure
	 * @return Const reference to internal configuration
	 */
	[[nodiscard]] const Configuration& get_config() const noexcept {
		return config_;
	}

	/**
	 * @brief Get mutable configuration (Python-friendly)
	 * @return Reference to internal configuration
	 */
	Configuration& get_mutable_config() {
		return config_;
	}

	/**
	 * @brief Validate the current configuration
	 * @throws ARBD::Exception if validation fails
	 */
	void validate() const;

	/**
	 * @brief Create boundary conditions from current config
	 * @return BoundaryConditions object
	 */
	[[nodiscard]] BoundaryConditions create_boundary_conditions() const;

	// Legacy compatibility
	[[nodiscard]] const SimConf& get_sim_conf() const noexcept {
		return *this;
	}

  private:
	Configuration config_;
	std::string file_name_;

	// Parsing helpers
	void parse_parameters(const Reader& reader);
	void apply_defaults();
	void get_elements(const Reader& reader);

	// Validation helpers
	void validate_physical_parameters() const;
	void validate_method_parameters() const;
	void validate_output_parameters() const;
};

} // namespace ARBD
