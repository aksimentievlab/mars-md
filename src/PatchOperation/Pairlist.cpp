#include "PatchOperation/Pairlist.h"
#include "ARBDException.h"
#include "PairListKernels/ZOrderPairlist.h"

namespace ARBD {

std::unique_ptr<Pairlist> create_pairlist(PairlistBuilderType type,
										  const Resource& resource,
										  size_t max_particles,
										  size_t max_pairs) {
	switch (type) {
	case PairlistBuilderType::ZOrder:
		return std::make_unique<ZOrderPairlist>(resource, max_particles, max_pairs);

	case PairlistBuilderType::CellList:
		ARBD_Exception(ExceptionType::NotImplementedError, "CellList pairlist not yet implemented");

	case PairlistBuilderType::VerletList:
		ARBD_Exception(ExceptionType::NotImplementedError,
					   "VerletList pairlist not yet implemented");

	case PairlistBuilderType::Hierarchical:
		ARBD_Exception(ExceptionType::NotImplementedError,
					   "Hierarchical pairlist not yet implemented");

	default:
		ARBD_Exception(ExceptionType::ValueError,
					   "Unknown pairlist type: {}",
					   static_cast<int>(type));
	}
}

} // namespace ARBD
