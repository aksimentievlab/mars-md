#pragma once
/**
 * @file PatchOperation/ReorderManager.h
 * @brief Intra-patch Morton reorder policy (ENABLE_ZORDER_REORDER).
 */

#include "Types/Types.h"

namespace MARS {

class Patch;

#ifdef ENABLE_ZORDER_REORDER
/**
 * @brief Decides when to Morton-reorder a patch's particles for cache locality.
 */
class ParticleReorderManager {
  public:
	explicit ParticleReorderManager(int reorder_period) : reorder_period_(reorder_period) {}

	/// Reorder `patch` if due this step. Returns true iff a reorder happened.
	bool maybe_reorder(size_t step, Patch& patch);

	int reorder_period() const {
		return reorder_period_;
	}

  private:
	int reorder_period_; ///< Reorder every N steps; <= 0 disables
};
#endif

} // namespace MARS