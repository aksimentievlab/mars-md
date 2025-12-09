#include "SimSystem.h"

namespace ARBD {

void SimSystem::validate_physical_parameters() const {

	if (temperature_.value <= arbd_real(0.0)) {
		throw Exception(ExceptionType::ValueError,
						SourceLocation(),
						"Temperature must be greater than 0");
	}

	if (cutoff_ <= Length(0.0)) {
		throw Exception(ExceptionType::ValueError,
						SourceLocation(),
						"Cutoff must be greater than 0");
	}
	if (sim_box_.get_box_size().x <= Length(0.0)) {
		throw Exception(ExceptionType::ValueError,
						SourceLocation(),
						"Box length x must be positive (got {})",
						sim_box_.get_box_size().x);
	}
	if (sim_box_.get_box_size().y <= Length(0.0)) {
		throw Exception(ExceptionType::ValueError,
						SourceLocation(),
						"Box length y must be positive (got {})",
						sim_box_.get_box_size().y);
	}
	if (sim_box_.get_box_size().z <= Length(0.0)) {
		throw Exception(ExceptionType::ValueError,
						SourceLocation(),
						"Box length z must be positive (got {})",
						sim_box_.get_box_size().z);
	}
};

void SimSystem::validate_method_parameters() const {

	if (steps_.timestep <= arbd_real(0.0)) {
		throw Exception(ExceptionType::ValueError,
						SourceLocation(),
						"Timestep must be positive (got {})",
						steps_.timestep);
	}
	if (steps_.steps <= 0) {
		throw Exception(ExceptionType::ValueError,
						SourceLocation(),
						"Number of steps must be positive (got {})",
						steps_.steps);
	}
	if (neighbor_list_rebuild_period <= arbd_real(0.0)) {
		throw Exception(ExceptionType::ValueError,
						SourceLocation(),
						"Neighbor list rebuild period must be positive (got {})",
						neighbor_list_rebuild_period);
	}
};

void SimSystem::validate_output_parameters() const {

	if (output_period_ <= arbd_real(0.0)) {
		throw Exception(ExceptionType::ValueError,
						SourceLocation(),
						"Output period must be positive (got {})",
						output_period_);
	}
	if (energy_output_period_ <= arbd_real(0.0)) {
		throw Exception(ExceptionType::ValueError,
						SourceLocation(),
						"Energy output period must be positive (got {})",
						energy_output_period_);
	}
	if (output_name.empty()) {
		throw Exception(ExceptionType::ValueError, SourceLocation(), "Output name cannot be empty");
	}
};
} // namespace ARBD
