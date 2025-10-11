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
#include "Objects/ParticleProperties.h"
#include "Objects/RigidBodyProperties.h"
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
		kT = value * 0.0019872065f;
	}
	Temperature(BaseGrid<float>* grid) : grid(grid) {
		format = Format::Grid;
		value = 0.0f; // No scalar value defined for grid.
		kT = 0.0f;	  // No global kT, but for compatibility we set it to zero.
		if (grid) {
			std::shared_ptr<BaseGrid<float>> kT_grid = std::make_shared<BaseGrid<float>>(*grid);
			kT_grid->scale(0.0019872065f);
			this->grid = kT_grid;
		} else {
			this->grid = nullptr;
		}
	}
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
	SimSteps(float timestep, float total_simulation_time)
		: timestep(timestep), total_simulation_time(total_simulation_time) {
		set_total_steps();
	}
	SimSteps(float timestep, int steps) : timestep(timestep), steps(steps) {
		total_simulation_time = timestep * steps;
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
	Spatial,			// For uniform systems - divides space into equal patches
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
enum class InteractionForm { Tabulated, Analytical };

} // namespace ARBD
