#include "Interactions/NonBondedInteraction.h"

namespace MARS {

void NonBondedInteractions::prepare_device_data() {
	// Pairwise device buffers are built per-patch in DevicePairNonBondedInteractions.
	// Grid device buffers are built in GridManager::build_device_arrays().
}

void NonBondedInteractions::cleanup_device_data() {}

} // namespace MARS
