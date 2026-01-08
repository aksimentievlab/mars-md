#pragma once

#include "../PatchDecomposer.h"
#include "Backend/Buffer.h"
#include "Types/Types.h"

namespace ARBD {

/**
 * @brief Recursive bisection patch decomposition for load balancing
 *
 * Recursively divides the domain to balance computational load
 * across resources. Useful for non-uniform particle distributions.
 */
class RecursiveBisectionPatchDecomposer : public PatchDecomposer {
  public:
	RecursiveBisectionPatchDecomposer() {
		type_ = DecomposerType::RecursiveBisection;
	}

	/**
	 * @brief Recursive bisection for load-balanced decomposition
	 *
	 * Recursively divides domain to achieve:
	 * - Balanced particle counts across patches
	 * - Minimized surface area (communication)
	 * - Adaptive boundaries based on particle distribution
	 */
	DecompositionPlan decompose(SimSystem& system, SystemState& state) override;

	std::string get_name() const override {
		return "Recursive Bisection Decomposer";
	}

  private:
	/**
	 * @brief Recursive bisection algorithm
	 */
	struct BisectionNode {
		Vector3 min_bounds, max_bounds;
		std::vector<idx_t> particle_indices;
		int level;
		patch_t patch_id;
	};

	void recursive_bisect(BisectionNode& node,
						  const HostParticleData& particles,
						  std::vector<BisectionNode>& result_nodes,
						  size_t target_patches) const;

	int find_optimal_split_dimension(const BisectionNode& node,
									 const HostParticleData& particles) const;
};
} // namespace ARBD
