#include "PatchOperation/Pairlist.h"
#include "ARBDException.h"
#include "PatchOperation/ZOrderPairlist.h"

namespace ARBD {

std::unique_ptr<Pairlist> create_pairlist(PairlistBuilderType type,
										  const Resource& resource,
										  size_t max_particles,
										  size_t max_pairs) {
	switch (type) {
	case PairlistBuilderType::ZOrder:
		return std::make_unique<ZOrderPairlist>(resource, max_particles, max_pairs);
		break;

	case PairlistBuilderType::CellList:
		ARBD_Exception(ExceptionType::NotImplementedError, "CellList pairlist not yet implemented");
		break;

	case PairlistBuilderType::VerletList:
		// TODO: Implement VerletListPairlist
		ARBD_Exception(ExceptionType::NotImplementedError,
					   "VerletList pairlist not yet implemented");
		break;

	case PairlistBuilderType::Hierarchical:
		// TODO: Implement HierarchicalPairlist
		ARBD_Exception(ExceptionType::NotImplementedError,
					   "Hierarchical pairlist not yet implemented");
		break;

	default:
		ARBD_Exception(ExceptionType::ValueError,
					   "Unknown pairlist type: {}",
					   static_cast<int>(type));
	}

	return nullptr;
}

} // namespace ARBD
