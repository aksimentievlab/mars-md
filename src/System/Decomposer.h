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
#include <memory>
#include <string>

namespace ARBD {

// Forward declarations
class SimSystem;
class PatchManager;

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
	 * @param sys The global system containing particle data and configuration.
	 * @param resources The collection of hardware resources (e.g., GPUs) to distribute to.
	 */
	virtual void decompose(SimSystem& sys, const ResourceCollection& resources) = 0;

	virtual const std::string get_name() {
		switch (type_) {
		case DecomposerType::Spatial:
			return "Spatial Decomposer";
		case DecomposerType::RecursiveBisection:
			return "Recursive Bisection Decomposer";
		case DecomposerType::Geometric:
			return "Geometric Decomposer";
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

	void decompose(SimSystem& sys, const ResourceCollection& resources) override;

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

	void decompose(SimSystem& sys, const ResourceCollection& resources) override;

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

	void decompose(SimSystem& sys, const ResourceCollection& resources) override;

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
