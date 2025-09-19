#pragma once

#include "ARBDException.h"
#include "ARBDLogger.h"
#include "IO/Reader.h"
#include "Interactions/Interactions.h"
#include "Objects/ParticleProperties.h"
#include "Objects/RigidBodyProperties.h"
#include "System/Reservoir.h"
#include "Types/BaseGrid.h"
#include "Types/Types.h"
#include <array>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace ARBD {

//================================================================================
// Core Configuration Data Structures
//================================================================================

/**
 * @brief Temperature configuration - supports both constant values and spatial grids
 */
struct Temperature {
	enum class Format { Value, Grid };
	Format format = Format::Value;
	float value = 298.15f; // Kelvin
	std::shared_ptr<BaseGrid<float>> grid = nullptr;
};

struct Pressure {
	float value = 1.0f;
};
/**
 * @brief Length/distance configuration
 */
struct Length {
	float value = 0.0f;

	// Python-friendly constructor
	explicit Length(float val = 0.0f) : value(val) {}

	// Implicit conversion for ease of use
	operator float() const {
		return value;
	}
};

/**
 * @brief Simulation step configuration
 * @note Actual parameters are timestep and steps.
 */
struct SimSteps {
	float timestep;
	int steps;
	float total_simulation_time;

	inline void set_total_steps() {
		if (timestep <= 0.0f) {
			throw Exception(ExceptionType::ValueError,
							SourceLocation(),
							"Timestep must be positive (got {})",
							timestep);
		}
		if (total_simulation_time <= 0.0f) {
			throw Exception(ExceptionType::ValueError,
							SourceLocation(),
							"Total simulation time must be positive (got {})",
							total_simulation_time);
		}
		steps = total_simulation_time / timestep;
	}
};

/**
 * @brief Boundary conditions with periodic/non-periodic support
 */
class BoundaryConditions {
  public:
	BoundaryConditions() = default;

	BoundaryConditions(Vector3 basis1,
					   Vector3 basis2,
					   Vector3 basis3,
					   Vector3 origin = {0.0f, 0.0f, 0.0f},
					   bool periodic1 = true,
					   bool periodic2 = true,
					   bool periodic3 = true)
		: origin_{origin}, basis_{basis1, basis2, basis3},
		  periodic_{periodic1, periodic2, periodic3} {}

	// Accessors
	const Vector3& get_origin() const {
		return origin_;
	}
	const std::array<Vector3, 3>& get_basis() const {
		return basis_;
	}
	const std::array<bool, 3>& get_periodicity() const {
		return periodic_;
	}

	// Python-friendly setters
	void set_origin(const Vector3& origin) {
		origin_ = origin;
	}
	void set_basis(const Vector3& b1, const Vector3& b2, const Vector3& b3) {
		basis_ = {b1, b2, b3};
	}
	void set_periodicity(bool p1, bool p2, bool p3) {
		periodic_ = {p1, p2, p3};
	}

  private:
	Vector3 origin_{0, 0, 0};
	std::array<Vector3, 3> basis_{Vector3{5000, 0, 0}, Vector3{0, 5000, 0}, Vector3{0, 0, 5000}};
	std::array<bool, 3> periodic_{true, true, true};
};

//================================================================================
// Simulation Method Enumerations
//================================================================================

enum class DecomposerType {
	Cell,				// For uniform systems
	RecursiveBisection, // For non-uniform systems (load balancing)
	Geometric			// For systems with specific shapes (e.g., membranes)
};

enum class LongRangeMethod {
	CutoffAMR, ///< Adaptive cutoff with mesh refinement
	PPPM,	   ///< Particle-Particle Particle-Mesh
	PME,	   ///< Particle Mesh Ewald
	FMM,	   ///< Fast Multipole Method
	Direct,	   ///< Direct O(N²) calculation (for small systems)
	None	   ///< No long-range interactions
};

enum class Periodicity { AllPeriodic, TwoDimensional, OneDimensional, Open };

enum class DynamicType { Brownian, Langevin, DPD };

enum class OutputFormat { DCD, PDB, HDF5 };

enum class ThermostatType { Langevin, NVE, NoseHooverLangevin };
enum class BarostatType { Isobaric, Isochoric };

struct GlobalObjects {
	std::vector<RigidBodyType> rigid_body_types;
	std::vector<ParticleType> particle_types;
	std::vector<RigidBody> rigid_bodies;
	std::vector<Particle> particles;
	std::vector<Bond> bonds;
	std::vector<Angle> angles;
	std::vector<Dihedral> dihedrals;
	std::vector<NonbondedInteraction> interactions;
	std::vector<Exclude> exclusions;
	std::vector<Restraint> restraints;
	std::vector<BaseGrid<float>> part_grid_dictionary;
	std::vector<Tables> tables;
};
//================================================================================
// Main Configuration Structure (Python-bindable)
//================================================================================

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
	Temperature temperature{Temperature::Format::Value, 298.15f};
	Length cutoff{10.0f};
	Length pairlist_distance{20.0f};
	std::array<float, 3> box_lengths{5000.0f, 5000.0f, 5000.0f};

	// Method selection
	Periodicity periodicity{Periodicity::AllPeriodic};
	DecomposerType decomposer{DecomposerType::Cell};
	LongRangeMethod long_range_method{LongRangeMethod::PPPM};
	DynamicType ParticleDynamicType{DynamicType::Langevin};
	DynamicType RigidBodyDynamicType{DynamicType::Langevin};
	std::map<std::string, int> functions_id_map;
	// Simulation control
	SimSteps steps{1e-5f, 1000, 0.0};

	ThermostatType thermostat{ThermostatType::Langevin};
	BarostatType barostat{BarostatType::Isobaric};
	float output_period{100.0f};
	float energy_output_period{100.0f};
	float decomp_period{10.0f};
	OutputFormat output_format{OutputFormat::DCD};

	std::string output_name{"out"};
	Pressure pressure{1.0f};
	std::vector<Reservoir> reservoirs;
	bool has_reaction = false;
	int replicas{1};
	GlobalObjects objects;

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
