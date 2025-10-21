#include "Configuration.h"

namespace ARBD {

void Configuration::validate_physical_parameters() const {

	if (temperature.value <= 0.0f) {
		throw Exception(ExceptionType::ValueError,
						SourceLocation(),
						"Temperature must be greater than 0");
	}

	if (cutoff.value <= 0.0f) {
		throw Exception(ExceptionType::ValueError,
						SourceLocation(),
						"Cutoff must be greater than 0");
	}
	if (sim_box.get_box_size().x <= 0.0f) {
		throw Exception(ExceptionType::ValueError,
						SourceLocation(),
						"Box length x must be positive (got {})",
						sim_box.get_box_size().x);
	}
	if (sim_box.get_box_size().y <= 0.0f) {
		throw Exception(ExceptionType::ValueError,
						SourceLocation(),
						"Box length y must be positive (got {})",
						sim_box.get_box_size().y);
	}
	if (sim_box.get_box_size().z <= 0.0f) {
		throw Exception(ExceptionType::ValueError,
						SourceLocation(),
						"Box length z must be positive (got {})",
						sim_box.get_box_size().z);
	}
};

void Configuration::validate_method_parameters() const {

	if (steps.timestep <= 0.0f) {
		throw Exception(ExceptionType::ValueError,
						SourceLocation(),
						"Timestep must be positive (got {})",
						steps.timestep);
	}
	if (steps.steps <= 0) {
		throw Exception(ExceptionType::ValueError,
						SourceLocation(),
						"Number of steps must be positive (got {})",
						steps.steps);
	}
	if (decomp_period <= 0) {
		throw Exception(ExceptionType::ValueError,
						SourceLocation(),
						"Decomposition period must be positive (got {})",
						decomp_period);
	}
};

void Configuration::validate_output_parameters() const {

	if (output_period <= 0.0f) {
		throw Exception(ExceptionType::ValueError,
						SourceLocation(),
						"Output period must be positive (got {})",
						output_period);
	}
	if (energy_output_period <= 0.0f) {
		throw Exception(ExceptionType::ValueError,
						SourceLocation(),
						"Energy output period must be positive (got {})",
						energy_output_period);
	}
	if (output_name.empty()) {
		throw Exception(ExceptionType::ValueError, SourceLocation(), "Output name cannot be empty");
	}
};
} // namespace ARBD
