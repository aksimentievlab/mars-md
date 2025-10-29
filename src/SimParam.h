#pragma once
/**
 * @file SimParam.h
 * @author Pin-Yi Li <pinyili2@illinois.edu>
 * @brief Simulation parameter structures, enum classes for global parameters.
 * @version 0.1
 * @date 2025-10-10
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "ARBDException.h"
#include "Constants.h"
#include "Types/BaseGrid.h"
#include "Types/Types.h"
#include <memory>

namespace ARBD {

/**
 * @brief Temperature configuration - supports both constant values and spatial grids
 */
struct Temperature {
	enum class Format { Value, Grid };
	Format format;
	float value; // Kelvin
	float kT;
	std::shared_ptr<BaseGrid<float>> grid = nullptr;

	Temperature(float value = 298.15f) : value(value) {
		format = Format::Value;
		kT = value * constants::BOLTZMANN;
	}
	Temperature(BaseGrid<float>* grid) : grid(grid) {
		format = Format::Grid;
		value = 0.0f; // No scalar value defined for grid.
		kT = 0.0f;	  // No global kT, but for compatibility we set it to zero.
		if (grid) {
			std::shared_ptr<BaseGrid<float>> kT_grid = std::make_shared<BaseGrid<float>>(*grid);
#ifdef HOST_GUARD
			kT_grid->scale(constants::BOLTZMANN);
#endif
			this->grid = kT_grid;
		} else {
			this->grid = nullptr;
		}
	}
	float get_kT(Vector3 position) {
		if (format == Format::Grid) {
			return grid->get_value(position);
		}
		return value;
	}
};

struct Pressure {
	float value = 1.0f;
};
/**
 * @brief Length/distance configuration
 */
struct Length {
	/**
	 * @brief Length in angstroms
	 */
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
	size_t steps;
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
		if (static_cast<size_t>(total_simulation_time / timestep) >
			std::numeric_limits<size_t>::max()) {
			throw Exception(ExceptionType::ValueError,
							SourceLocation(),
							"Total simulation time is too large for the number of steps");
		}
		if (total_simulation_time < timestep) {
			throw Exception(ExceptionType::ValueError,
							SourceLocation(),
							"Total simulation time is too small for the number of steps");
		}
		steps = static_cast<size_t>(total_simulation_time / timestep);
	}
	SimSteps(float timestep, float total_simulation_time)
		: timestep(timestep), total_simulation_time(total_simulation_time) {
		set_total_steps();
	}
	SimSteps(float timestep, int steps) : timestep(timestep), steps(steps) {
		total_simulation_time = timestep * steps;
	}
};

//================================================================================
// Simulation Method Enumerations
//================================================================================

enum class DecomposerType {
	Spatial,			// For uniform systems - divides space into equal patches
	ZOrder,				// For cache-friendly spatial sorting and multi-device scaling
	RecursiveBisection, // For non-uniform systems (load balancing)
	Geometric			// For systems with specific shapes (e.g., membranes)
};
enum class DecomposeDirection { X, Y, Z }; // default is Z

enum class LongRangeMethod {
	CutoffAMR, ///< Adaptive cutoff with mesh refinement
	PPPM,	   ///< Particle-Particle Particle-Mesh
	PME,	   ///< Particle Mesh Ewald
	FMM,	   ///< Fast Multipole Method
	Direct,	   ///< Direct O(N²) calculation (for small systems)
	None	   ///< No long-range interactions
};

enum class DynamicType { Brownian, Langevin, NoseHooverLangevin, DPD };
enum class OutputFormat { DCD, PDB, HDF5 };
enum class ThermostatType { NVE, NoseHooverLangevin };
enum class BarostatType { Isobaric, Isochoric };
enum class InteractionForm { Grid, Tabulated, Analytical };
enum class TabulatedType { NBPair, Bond, Angle, Dihedral, Custom };

} // namespace ARBD
