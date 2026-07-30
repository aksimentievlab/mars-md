#pragma once

#include "BondConfigReader.h"
#include "Reader.h"
#include "System/SimSystem.h"
#include "System/SystemState.h"
#include <map>
#include <string>	   // For std::string member
#include <string_view> // For function parameters

// Forward declaration for pybind11
#ifdef USE_PYTHON
namespace pybind11 {
class object;
}
#endif

namespace ARBD {
/**
 * @brief Configuration manager with file I/O and validation
 *
 * This class handles:
 * - Loading configuration from files (configures existing SimSystem)
 * - Temporarily holds initial topology data (particles, bonds, etc.)
 * - Initial data is retrieved once and then discarded
 * - Validation of parameters
 * - Python bindings
 *
 * Architecture:
 * - SimSystem: time-invariant configuration only
 * - SystemState: current runtime state only
 * - ConfigParser: temporary initial data loader
 */
class ConfigParser {
  public:
	/**
	 * @brief Construct from configuration file
	 * @param sim_system Simulation system to configure
	 * @param file_name Path to configuration file
	 * @throws ARBD::Exception on I/O or validation error
	 */
	ConfigParser(SimSystem& sim_system, std::string_view file_name);

#ifdef USE_PYTHON
	/**
	 * @brief Construct from Python dictionary (pybind11-friendly)
	 * @param sim_system Simulation system to configure
	 * @param config_dict Python dictionary with configuration parameters
	 */
	ConfigParser(SimSystem& sim_system, const std::map<std::string, pybind11::object>& config_dict);
#endif
	/**
	 * @brief Parse configuration from file
	 * @param file_name Path to configuration file
	 * @throws ARBD::Exception on I/O or validation error
	 */
	void parse_file(std::string_view file_name);

	/**
	 * @brief Validate the current configuration
	 * @throws ARBD::Exception if validation fails
	 */
	void validate() const;

	/**
	 * @brief Get the configured SimSystem
	 */
	SimSystem& get_sim_system() {
		return *sim_system_ref_;
	}

	const SimSystem& get_sim_system() const {
		return *sim_system_ref_;
	}

	/**
	 * @brief Get initial particles (temporary data, used once during initialization)
	 */
	std::vector<ParticleIO>& get_init_particles() {
		return init_particles_;
	}

	const std::vector<ParticleIO>& get_init_particles() const {
		return init_particles_;
	}

	const BondedInteractions& get_init_bonded_interactions() const {
		return init_bonded_interactions_;
	}

	/**
	 * @brief Get initial rigid bodies (temporary data, used once during initialization)
	 */
	std::vector<RigidBodyIO>& get_init_rigid_bodies() {
		return init_rigid_bodies_;
	}

	const std::vector<RigidBodyIO>& get_init_rigid_bodies() const {
		return init_rigid_bodies_;
	}

  private:
	// Reference to the SimSystem being configured
	SimSystem* sim_system_ref_;

	std::string file_name_; //*.bd

	// Temporary initial topology data (used once during initialization, then discarded)
	std::vector<ParticleIO> init_particles_{};
	std::vector<RigidBodyIO> init_rigid_bodies_{};
	BondedInteractions init_bonded_interactions_{};

	// BondConfigReader initialized in constructor body to ensure sim_system_ref_ is set
	BondConfigReader bond_config_reader_;

	// Parsing helpers
	void parse_parameters(const Reader& reader);
	void apply_defaults();
	void get_elements(const Reader& reader);
#ifdef USE_PYTHON
	void parse_dictionary(const std::map<std::string, pybind11::object>& config_dict);
#endif
};
} // namespace ARBD
