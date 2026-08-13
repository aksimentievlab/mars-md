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
typedef float Length;
/**
 * @brief Temperature configuration - supports both constant values and spatial grids
 * @tparam TemperatureType Temperature value type (float/double)

 */
struct Temperature {
	enum class Format { Value, Grid };
	Format format;
	float value; // Kelvin
	float kT;
	BaseGrid<arbd_real> temperature_grid;

	Temperature(float value = 298.15f) : value(value) {
		format = Format::Value;
		kT = value * constants::BOLTZMANN;
	}
	Temperature(BaseGrid<arbd_real> grid) : temperature_grid(grid) {
		format = Format::Grid;
		value = 0.0f; // No scalar value defined for grid.
		kT = 0.0f;	  // No global kT, but for compatibility we set it to zero.
#ifdef HOST_GUARD
		if (temperature_grid.config().is_valid()) {
			// Scale temperature grid by Boltzmann constant on host
			temperature_grid.scale(constants::BOLTZMANN);
		}
#endif
	}

	float get_kT(Vector3 position) {
		if (format == Format::Grid) {
			return temperature_grid.get_value(position);
		}
		return value;
	}

	/**
	 * @brief Transfer temperature data to device for GPU access
	 * @param resource Target computational resource
	 */
	HOST void sync_to_device(const Resource& resource) {
#ifdef HOST_GUARD
		if (format == Format::Grid) {
			temperature_grid.sync_to_device(resource);
		}
#endif
	}

	/**
	 * @brief Get device pointer for temperature data
	 * @param resource Target computational resource
	 * @return Device pointer (null for constant temperature)
	 */
	HOST float* get_device_pointer(const Resource& resource) {
#ifdef HOST_GUARD
		if (format == Format::Grid) {
			return temperature_grid.get_device_pointer(resource);
		}
		return nullptr; // Constant temperature doesn't need device pointer
#else
		return nullptr;
#endif
	}

	/**
	 * @brief Check if temperature uses grid format
	 */
	bool is_grid() const {
		return format == Format::Grid;
	}
};

struct Pressure {
	float value = 1.0f;
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
enum class IntegratorType { Langevin, Brownian, VelocityVerlet };

enum class LangevinSplitting {
	BBK,   // Standard implementations (GROMACS/NAMD legacy)
	BAOAB, // Leimkuhler-Matthews (Better sampling at high dt)
	OBABO  // Bussi-Parrinello style
};

enum class OutputFormat { DCD, PDB, HDF5 };
enum class BarostatType {
	None,
	MonteCarlo,
	LangevinPiston // The "Standard" for membrane kinetics (NAMD style)
};
enum class InteractionForm { Grid, Tabulated, Analytical };

} // namespace ARBD
