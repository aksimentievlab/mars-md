#pragma once
/**
 * @file System/Decompose2Patch.h
 * @author Pin-Yi Li <pinyili2@illinois.edu>
 * @brief Patch decomposition interfaces and implementations
 * @version 2.0
 * @date 2025-09-09
 *
 * @copyright Copyright (c) 2025
 */

#include "ARBDException.h"
#include "ARBDLogger.h"
#include "Backend/Resource.h"
#include "Configuration.h"
#include "SimParam.h"
#include "Types/Types.h"
#include <array>
#include <memory>
#include <string>
#include <vector>

namespace ARBD {

// Forward declarations
class SimSystem;

/**
 * @brief Result of patch decomposition - describes the patch layout and structure
 *
 * This structure encapsulates all information computed during decomposition
 * and is used to initialize the PatchManager. It represents a one-time
 * decomposition plan created during system setup.
 */
struct DecompositionPlan {
	// Grid structure
	std::array<int, 3> grid_dimensions = {0,
										  0,
										  0}; ///< Number of patches in each dimension (nx, ny, nz)
	std::array<bool, 3> periodicity = {true, true, true}; ///< Periodic boundary flags (x, y, z)

	// Patch boundaries (one per patch in the grid)
	std::vector<Vector3> patch_min_bounds; ///< Minimum bounds for each patch
	std::vector<Vector3> patch_max_bounds; ///< Maximum bounds for each patch

	// Resource assignment
	std::vector<Resource> patch_resources; ///< Resource assigned to each patch

	// System-wide information
	Vector3 system_min = Vector3(0.0f); ///< Global system minimum bounds
	Vector3 system_max = Vector3(0.0f); ///< Global system maximum bounds

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

	DecomposerType get_type() const {
		return type_;
	}

	/**
	 * @brief The core function that performs the patch decomposition.
	 *
	 * This method computes the decomposition plan and stores it in the SimSystem.
	 * Decomposition should only be called once during system setup.
	 *
	 * @param sys The global system containing particle data and configuration.
	 * @return DecompositionPlan containing the computed patch layout
	 */
	virtual DecompositionPlan decompose(SimSystem& sys) = 0;

	virtual const std::string get_name() {
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
			throw ARBD::Exception(ARBD::ExceptionType::ValueError, "Unsupported decomposer type.");
		}
	}

  protected:
	DecomposerType type_;
};

/**
 * @brief Spatial patch decomposition for uniform systems
 *
 * Divides the simulation box into equal-sized spatial regions and assigns
 * each region as a patch to different computational resources.
 */
class SpatialPatchDecomposer : public PatchDecomposer {
  public:
	SpatialPatchDecomposer() : type_(DecomposerType::Spatial) {}

	DecompositionPlan decompose(SimSystem& sys) override;

  private:
	DecomposerType type_;
};

/**
 * @brief Recursive bisection patch decomposition for load balancing
 *
 * Recursively divides the domain to balance computational load
 * across resources. Useful for non-uniform particle distributions.
 */
class RecursiveBisectionPatchDecomposer : public PatchDecomposer {
  public:
	RecursiveBisectionPatchDecomposer() : type_(DecomposerType::RecursiveBisection) {}

	DecompositionPlan decompose(SimSystem& sys) override;

  private:
	DecomposerType type_;
};

/**
 * @brief Geometric patch decomposition for systems with specific shapes
 *
 * Uses geometric information (e.g., membrane boundaries) to
 * create domain partitions that respect system geometry.
 */
class GeometricPatchDecomposer : public PatchDecomposer {
  public:
	GeometricPatchDecomposer() : type_(DecomposerType::Geometric) {}

	DecompositionPlan decompose(SimSystem& sys) override;

  private:
	DecomposerType type_;
};

/**
 * @brief Factory function to create patch decomposer instances
 * @param type The type of decomposer to create
 * @return Unique pointer to the created decomposer
 */
std::unique_ptr<PatchDecomposer> create_patch_decomposer(DecomposerType type);

} // namespace ARBD
