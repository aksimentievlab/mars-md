#include "PatchOperation/ReorderManager.h"

#ifdef ENABLE_ZORDER_REORDER
#include "PatchOperation/Patch.h"

namespace ARBD {

bool ParticleReorderManager::maybe_reorder(size_t step, Patch& patch) {
	if (reorder_period_ <= 0) {
		return false;
	}
	if (step == 0) {
		return false;
	}
	if ((step - 1) % static_cast<size_t>(reorder_period_) != 0) {
		return false;
	}
	// Bonded topology must be on-device before its indices can be remapped; it is
	// prepared lazily by the first force calc, so skip until then.
	if (!patch.is_bonded_prepared()) {
		return false;
	}
	patch.reorder_particles();
	return true;
}

} // namespace ARBD
#endif
