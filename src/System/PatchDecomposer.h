#pragma once
/**
 * @file System/PatchDecomposer.h
 * @author Pin-Yi Li <pinyili2@illinois.edu>
 * @brief Base patch decomposer interfaces for DeviceParticle system
 * @version 2.0
 * @date 2025-09-09
 *
 * Contains abstract base classes for patch decomposition algorithms.
 * Concrete implementations are in Decomposers.h/cpp to avoid circular dependencies.
 *
 * @copyright Copyright (c) 2025
 */

#include "ARBDException.h"
#include "ARBDLogger.h"
#include "Backend/Resource.h"
#include "Objects/ParticleProperties.h"
#include "SimParam.h"
#include "Types/Types.h"
#include <array>
#include <memory>
#include <string>
#include <vector>

namespace ARBD {

// Forward declarations
class SimSystem;
class SystemState;

/**
 * @brief Statistics from decomposition process
 */
struct DecompositionStats {
	float decomposition_time_ms = 0.0f;		 ///< Time taken for decomposition
	float load_balance_factor = 1.0f;		 ///< Load imbalance metric
	float communication_volume = 0.0f;		 ///< Estimated communication overhead
	size_t total_particles = 0;				 ///< Total particles decomposed
	std::vector<size_t> particles_per_patch; ///< Particle distribution
};

/**
 * @brief Result of patch decomposition with DeviceParticle integration
 *
 * This structure encapsulates spatial decomposition results and provides
 * all information needed by PatchManager to create patches with proper
 * resource binding and particle distribution.
 */
struct DecompositionPlan {
	// Grid structure
	std::array<int, 3> grid_dimensions = {0, 0, 0};		  ///< Number of patches (nx, ny, nz)
	std::array<bool, 3> periodicity = {true, true, true}; ///< Periodic boundary flags

	// Patch boundaries (one per patch in the grid)
	std::vector<Vector3> patch_min_bounds; ///< Minimum bounds for each patch
	std::vector<Vector3> patch_max_bounds; ///< Maximum bounds for each patch

	// Resource assignment
	std::vector<Resource> patch_resources; ///< Resource assigned to each patch

	// System-wide information
	Vector3 system_min = Vector3(0.0f); ///< Global system minimum bounds
	Vector3 system_max = Vector3(0.0f); ///< Global system maximum bounds

	// Particle assignment (for non-regular decompositions)
	std::vector<std::vector<idx_t>> patch_particle_indices; ///< Particle indices per patch

	// Performance and validation
	DecompositionStats statistics; ///< Decomposition performance metrics

	/**
	 * @brief Get total number of patches
	 */
	size_t total_patches() const {
		return static_cast<size_t>(grid_dimensions[0]) * static_cast<size_t>(grid_dimensions[1]) *
			   static_cast<size_t>(grid_dimensions[2]);
	}

	/**
	 * @brief Validate that the decomposition plan is consistent
	 */
	bool is_valid() const {
		size_t expected_patches = total_patches();
		return expected_patches > 0 && patch_min_bounds.size() == expected_patches &&
			   patch_max_bounds.size() == expected_patches &&
			   patch_resources.size() == expected_patches;
	}
};

/**
 * @brief Abstract base class for patch decomposition strategies
 *
 * Patch decomposers are responsible for dividing the simulation domain
 * into patches and assigning them to available computational resources
 * (GPUs, MPI ranks, etc.)
 */
class PatchDecomposer {
  public:
	virtual ~PatchDecomposer() = default;

	/**
	 * @brief Get decomposer type identifier
	 */
	DecomposerType get_type() const {
		return type_;
	}

	/**
	 * @brief Perform patch decomposition with DeviceParticle integration
	 *
	 * This method computes spatial decomposition based on:
	 * - Global particle data from SystemState
	 * - System configuration from SimSystem
	 * - Available computational resources
	 * - Optimization criteria (load balance, communication, etc.)
	 *
	 * @param system Simulation system with configuration and resources
	 * @param state System state with global particle data
	 * @return DecompositionPlan for PatchManager initialization
	 */
	virtual DecompositionPlan decompose(SimSystem& system, SystemState& state) = 0;

	/**
	 * @brief Get human-readable decomposer name
	 */
	virtual std::string get_name() const {
		switch (type_) {
		case DecomposerType::Spatial:
			return "Spatial Decomposer";
		case DecomposerType::RecursiveBisection:
			return "Recursive Bisection Decomposer";
		case DecomposerType::Geometric:
			return "Geometric Decomposer";
		case DecomposerType::ZOrder:
			return "Z-Order Decomposer";
		default:
			throw Exception(ExceptionType::ValueError,
							SourceLocation(),
							"Unsupported decomposer type");
		}
	}

	/**
	 * @brief Get decomposition statistics (if available)
	 */
	virtual DecompositionStats get_statistics() const {
		return {}; // Default empty stats
	}

  protected:
	DecomposerType type_;

	/**
	 * @brief Helper function to validate decomposition plan
	 */
	bool validate_decomposition_plan(const DecompositionPlan& plan) const {
		return plan.is_valid() && plan.patch_min_bounds.size() == plan.patch_max_bounds.size() &&
			   plan.patch_min_bounds.size() == plan.patch_resources.size();
	}

	/**
	 * @brief Helper to distribute particles to patches based on position
	 */
	std::vector<std::vector<idx_t>>
	assign_particles_to_patches(const HostParticleData& particles,
								const std::vector<Vector3>& patch_min_bounds,
								const std::vector<Vector3>& patch_max_bounds) const;
};

} // namespace ARBD
