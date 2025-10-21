#include "Compute/Pairlist.h"
#include "ARBDException.h"
#include "Compute/ZOrderPairlist.h"
#include "Compute/CellListPairlist.h"

namespace ARBD {

std::unique_ptr<Pairlist> create_pairlist(PairlistType type,
										  const Resource& resource,
										  size_t max_particles,
										  size_t max_pairs) {
	switch (type) {
	case PairlistType::ZOrder:
		return std::make_unique<ZOrderPairlist>(resource, max_particles, max_pairs);

	case PairlistType::CellList:
		return std::make_unique<CellListPairlist>(resource, max_particles, max_pairs);

	case PairlistType::VerletList:
		// TODO: Implement VerletListPairlist
		ARBD_Exception(ExceptionType::NotImplementedError,
					   "VerletList pairlist not yet implemented");
		break;

	case PairlistType::Hierarchical:
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
