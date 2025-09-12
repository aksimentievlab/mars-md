#pragma once
/**
 * @file SimSystem.h
 * @author Pin-Yi Li <pinyili2@illinois.edu>
 * @brief Simulation system class. Stores the system configuration and objects that won't change
 * during the simulation.
 * @version 2.0
 * @date 2025-09-09
 *
 * @copyright Copyright (c) 2025
 *
 */

#include "ARBDException.h"
#include "ARBDLogger.h"
#include "Backend/Buffer.h"
#include "Backend/Resource.h"
#include "Types/IndexList.h"
#include "Types/Types.h"
#include <array>
#include <iostream> // For logging placeholders
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace ARBD {

// Forward-declare SimSystem to be used in the Decomposer interface
class SimSystem;

struct Temperature { // Temperature grids can be stored here?
	float value;
};
struct Length {
	float value;
};

struct OutputPeriod {
	float Period;
	float EnergyPeriod;
};
struct SimSteps {
	float timestep;
	int steps;
	int decompPeriod;
};

class BoundaryConditions {
  public:
	BoundaryConditions()
		: origin_{0, 0, 0}, basis_{Vector3{5000, 0, 0}, Vector3{0, 5000, 0}, Vector3{0, 0, 5000}},
		  periodic_{true, true, true} {
		LOGINFO("BoundaryConditions default constructor.");
	}

	BoundaryConditions(Vector3 basis1,
					   Vector3 basis2,
					   Vector3 basis3,
					   Vector3 origin = {0.0f, 0.0f, 0.0f},
					   bool periodic1 = true,
					   bool periodic2 = true,
					   bool periodic3 = true)
		: origin_{origin}, basis_{basis1, basis2, basis3},
		  periodic_{periodic1, periodic2, periodic3} {
		LOGINFO("BoundaryConditions parameterized constructor.");
	}

	const Vector3& get_origin() const {
		return origin_;
	}
	const std::array<Vector3, 3>& get_basis() const {
		return basis_;
	}
	const std::array<bool, 3>& get_periodicity() const {
		return periodic_;
	}

  private:
	Vector3 origin_;
	std::array<Vector3, 3> basis_;
	std::array<bool, 3> periodic_;
};

//================================================================================
// Decomposer Interface (Strategy Pattern)
//================================================================================
/**
 * @brief Abstract base class for domain decomposition strategies.
 *
 * This class defines the interface for different ways to partition the simulation
 * domain and distribute work among available resources.
 */
class Decomposer {
  public:
	virtual ~Decomposer() = default;

	/**
	 * @brief The core function that performs the decomposition.
	 * @param sys The global system containing particle data and configuration.
	 * @param resources The collection of hardware resources (e.g., GPUs) to distribute to.
	 */
	virtual void decompose(SimSystem& sys, const ResourceCollection& resources) = 0;

	virtual std::string get_name() const = 0;
};

class CellDecomposer : public Decomposer {
  public:
	void decompose(SimSystem& sys, const ResourceCollection& resources) override;
	std::string get_name() const override {
		return "CellDecomposer";
	}
};
//================================================================================
// Global Simulation System
//================================================================================
class SimSystem {
  public:
	/**
	 * @brief Central configuration struct for initializing a simulation.
	 */
	struct Conf {
		enum class DecomposerType {
			Cell,				// For uniform systems
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
		// Particle Mesh Ewald, Fast multipole method
		enum class Periodicity { AllPeriodic, TwoDimensional, OneDimensional, Open };
		enum class Algorithm { Brownian, Langevin, DPD };
		enum class OutputFormat { DCD, PDB, HDF5 };
		// std::unordered_map<partilce1, particle2, reaction_type> reaction;

		Temperature temperature{298.15f};
		Periodicity periodicity{Periodicity::AllPeriodic};
		DecomposerType decomposer{DecomposerType::Cell};
		LongRangeMethod long_range_method{LongRangeMethod::PPPM};
		Algorithm algorithm{Algorithm::Langevin};
		bool has_reaction = false;
		OutputPeriod OutputPeriod{100, 10};
		OutputFormat outputFormat{OutputFormat::DCD};
		Length cutoff{50.0f};
		std::array<float, 3> box_lengths{5000.0f, 5000.0f, 5000.0f};
		std::string output_name{"out"};
	};

	/**
	 * @brief The primary constructor. Builds the entire system from a configuration struct.
	 */
	SimSystem(const Conf& conf, const ResourceCollection& resources)
		: temperature_(conf.temperature), cutoff_(conf.cutoff),
		  boundary_conditions_{}, // Will be overwritten below
		  resources_(resources) {
		LOGINFO("SimSystem constructor from Conf.");

		// 1. Set up boundary conditions based on config
		switch (conf.periodicity) {
		case Conf::Periodicity::AllPeriodic:
			boundary_conditions_ = BoundaryConditions(Vector3(conf.box_lengths[0], 0, 0),
													  Vector3(0, conf.box_lengths[1], 0),
													  Vector3(0, 0, conf.box_lengths[2]),
													  {0, 0, 0},
													  true,
													  true,
													  true);
			break;
		// Add other cases for TwoDimensional, Open, etc.
		default:
			throw std::runtime_error("Unsupported periodicity specified in configuration.");
		}
		// 2. Create the chosen decomposer instance (Factory Pattern)
		switch (conf.decomposer) {
		case Conf::DecomposerType::Cell:
			decomposer_ = std::make_unique<CellDecomposer>();
			break;
		default:
			throw std::runtime_error("Unsupported decomposer type specified in configuration.");
		}
		LOGINFO("Using Decomposer: " + decomposer_->get_name());
	}

	/**
	 * Q2: Should the Decomposer also be the object that converts
	 * SymbolicPatchOps to concrete PatchOp?
	 * A2: Yes, absolutely. The Decomposer is the only object with complete
	 * knowledge of the spatial layout of patches/cells. It is therefore
	 * the perfect candidate to translate abstract operations (e.g., "apply force
	 * in region X") into concrete work on specific patches.
	 *
	 * Q3: Neighbour list for in-GPU decomposition, Halo exchange for inter-GPU?
	 * A3: Correct. This is the standard pattern.
	 * - Neighbor Lists: For interactions within a single patch (or between
	 * adjacent patches) residing on the *same* computational resource.
	 * - Halo Exchanges: For communicating data (particle positions, forces)
	 * for particles in the "halo" or "ghost" regions between patches on
	 * *different* resources (inter-GPU or inter-node).
	 */

	// --- Public Accessors ---
	const Temperature& get_temperature() const {
		return temperature_;
	}
	const Length& get_cutoff() const {
		return cutoff_;
	}
	const BoundaryConditions& get_boundary_conditions() const {
		return boundary_conditions_;
	}
	/**
	 * @brief Triggers the domain decomposition process.
	 *
	 * Q1: Should this normally only happen at initialization?
	 * A1: Yes, a full decomposition is typically done once at initialization.
	 * Subsequent changes (e.g., for dynamic load balancing) would be handled
	 * by more specific operations, potentially triggered by a 'rebalance()' method.
	 */
	void decompose_system() {
		if (!decomposer_) {
			throw std::runtime_error("No decomposer has been set.");
		}
		LOGINFO("decompose_system() called.");
		decomposer_->decompose(*this, resources_);
	}

  private:
	// --- Core Configuration & State ---
	Temperature temperature_;
	Length cutoff_;
	BoundaryConditions boundary_conditions_;

	// --- Decomposition & Resources ---
	std::unique_ptr<Decomposer> decomposer_;
	ResourceCollection resources_;
};

} // namespace ARBD
