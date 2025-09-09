#pragma once

#include "ARBDException.h"
#include "ARBDLogger.h"
#include "System/SimSystem.h"

#include <array>
#include <string>
#include <string_view>

namespace ARBD {

/**
 * @brief Minimal, validated configuration that maps 1:1 to runtime parameters.
 *
 * Configuration is responsible for I/O and validation (implemented in Configuration.cpp).
 * It exposes a stable DTO (ActualParameters) and a converter to SimSystem::Conf.
 */
struct ActualParameters {

	Temperature temperature{298.15f};
	SimSystem::Conf::Periodicity periodicity{SimSystem::Conf::Periodicity::AllPeriodic};
	SimSystem::Conf::DecomposerType decomposer{SimSystem::Conf::DecomposerType::Cell};
	SimSystem::Conf::Algorithm algorithm{SimSystem::Conf::Algorithm::Langevin};
	SimSystem::Conf::ReactionScheme reaction_scheme{SimSystem::Conf::ReactionScheme::None};

	std::array<float, 3> box_lengths{5000.0f, 5000.0f, 5000.0f};
	float cutoff{50.0f};
};

class Configuration {
  public:
	Configuration() = default;

	/**
	 * @brief Load and validate a configuration from file (e.g., .brown).
	 * @throws ARBD::Exception on I/O or validation error
	 */
	static Configuration Load(std::string_view file_name);

	/**
	 * @brief Construct directly from already-validated parameters.
	 */
	explicit Configuration(const ActualParameters& params) : params_(params) {}

	/**
	 * @brief Access validated parameters (DTO).
	 */
	[[nodiscard]] const ActualParameters& params() const noexcept {
		return params_;
	}

	/**
	 * @brief Convert to SimSystem::Conf for system construction.
	 */
	[[nodiscard]] SimSystem::Conf to_sim_conf() const noexcept {
		SimSystem::Conf conf{};
		conf.temperature = params_.temperature;
		conf.periodicity = params_.periodicity;
		conf.decomposer = params_.decomposer;
		conf.algorithm = params_.algorithm;
		conf.reaction_scheme = params_.reaction_scheme;
		conf.box_lengths = params_.box_lengths;
		conf.cutoff = params_.cutoff;
		return conf;
	}

  private:
	ActualParameters params_{};
};

} // namespace ARBD
