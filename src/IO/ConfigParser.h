#pragma once

#include "BondConfigReader.h"
#include "Configuration.h"
#include "Reader.h"
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
 * - Loading configuration from files
 * - Validation of parameters
 * - Python bindings
 */
class ConfigParser {
  public:
	ConfigParser() = default;

	/**
	 * @brief Construct from configuration file
	 * @param file_name Path to configuration file
	 * @throws ARBD::Exception on I/O or validation error
	 */
	explicit ConfigParser(std::string_view file_name);

	/**
	 * @brief Construct from Configuration struct (Python-friendly)
	 * @param config Configuration structure
	 */
	ConfigParser(Configuration config);

#ifdef USE_PYTHON
	/**
	 * @brief Construct from Python dictionary (pybind11-friendly)
	 * @param config_dict Python dictionary with configuration parameters
	 */
	ConfigParser(const std::map<std::string, pybind11::object>& config_dict);
#endif
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

  private:
	Configuration config_;
	std::string file_name_;

	// Parsing helpers
	void parse_parameters(const Reader& reader);
	void apply_defaults();
	void get_elements(const Reader& reader);
#ifdef USE_PYTHON
	void parse_dictionary(const std::map<std::string, pybind11::object>& config_dict);
#endif

	// Validation helpers
	void validate_physical_parameters() const;
	void validate_method_parameters() const;
	void validate_output_parameters() const;
};
} // namespace ARBD
