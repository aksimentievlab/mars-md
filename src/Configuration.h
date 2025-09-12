#pragma once

#include "ARBDException.h"
#include "ARBDLogger.h"
#include "IO/Reader.h"
#include "SimSystem.h"
#include <array>
#include <string>
#include <string_view>

namespace ARBD {

/**
 * @brief Minimal, validated configuration that maps 1:1 to runtime parameters.
 *
 * Configuration is responsible for I/O and validation (implemented in Configuration.cpp).
 * It exposes a stable DTO (ActualParameters) and a converter to SimSystem::Conf.
 * Singleton Constructor.
 */

class Configuration {
  public:
	Configuration() = default;

	/**
	 * @brief Load and validate a configuration from file (e.g., .brown).
	 * @throws ARBD::Exception on I/O or validation error
	 */
	Configuration(std::string_view file_name);

	/**
	 * @brief Convert to SimSystem::Conf for system construction.
	 */
	[[nodiscard]] SimSystem::Conf to_sim_conf() const noexcept {
		SimSystem::Conf conf{};
		conf.temperature = params_.temperature;
		conf.periodicity = params_.periodicity;
		conf.decomposer = params_.decomposer;
		conf.algorithm = params_.algorithm;
		conf.has_reaction = params_.has_reaction;
		conf.box_lengths = params_.box_lengths;
		conf.cutoff = params_.cutoff;
		return conf;
	}

  private:
};

} // namespace ARBD
