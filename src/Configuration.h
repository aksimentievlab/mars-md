#pragma once

#include "ARBDException.h"
#include "ARBDLogger.h"
#include "IO/Reader.h"
#include "Interactions/Interactions.h"
#include "Objects/ARBDObjects.h"
#include "SimParam.h"
#include "System/PeriodicBox.h"
#include "System/Reservoir.h"
#include "Types/Types.h"
#include <array>
#include <string>
#include <string_view>
#include <vector>

namespace ARBD {

struct Configuration {
	// Physical parameters
	Temperature temperature{298.15f};

	Length cutoff{10.0f};
	Length pairlist_distance{20.0f};
	PeriodicBox sim_box;
	DecomposerType decomposer{DecomposerType::Spatial};
	LongRangeMethod long_range_method{LongRangeMethod::PPPM};
	DynamicType ParticleDynamicType{DynamicType::Langevin};
	DynamicType RigidBodyDynamicType{DynamicType::Langevin};
	std::map<std::string, int> functions_id_map;
	// Simulation control
	SimSteps steps{1e-5f, 1000};
	DecomposeDirection decompose_direction{DecomposeDirection::Z};

	ThermostatType thermostat{ThermostatType::Langevin};
	BarostatType barostat{BarostatType::Isobaric};
	float output_period{10.0f};
	float energy_output_period{100.0f};
	float neighbor_list_rebuild_period{100.0f};

	std::string output_name{"out"};
	OutputFormat output_format{OutputFormat::DCD};

	Pressure pressure{1.0f};
	std::vector<Reservoir> reservoirs;
	bool has_reaction = false;
	int replicas{1};
	ARBDObjects objects; // initialized only. Stored in SimState.

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
};

} // namespace ARBD
