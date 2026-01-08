#pragma once

#include "Backend/Buffer.h"
#include "System/PatchDecomposer.h"
#include "Types/Types.h"

namespace ARBD {

/**
 * @brief Spatial patch decomposition for uniform systems
 *
 * Divides the simulation box into equal-sized spatial regions and assigns
 * each region as a patch to different computational resources.
 */
class SpatialPatchDecomposer : public PatchDecomposer {
  public:
	SpatialPatchDecomposer() {
		type_ = DecomposerType::Spatial;
	}

	/**
	 * @brief Spatial decomposition with uniform patch sizes
	 *
	 * Creates regular grid decomposition based on:
	 * - System box dimensions
	 * - Interaction cutoff radius
	 * - Available computational resources
	 *
	 * Optimizes for:
	 * - Minimal communication volume
	 * - Balanced patch sizes
	 * - GPU memory alignment
	 */
	DecompositionPlan decompose(SimSystem& system, SystemState& state) override;

	std::string get_name() const override {
		return "Spatial Decomposer (Regular Grid)";
	}

  private:
	/**
	 * @brief Calculate optimal grid dimensions
	 */
	std::array<int, 3> calculate_grid_dimensions(const Vector3& system_size,
												 const Length& cutoff,
												 size_t num_resources) const;

	/**
	 * @brief Estimate communication overhead
	 */
	float estimate_communication_cost(const std::array<int, 3>& grid_dims,
									  const Vector3& system_size,
									  const Length& cutoff) const;
};
} // namespace ARBD
