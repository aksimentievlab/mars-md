#include "PatchOperation/Pairlist.h"
#include "MARSException.h"
#include "PairListKernels/ZOrderPairlist.h"

namespace MARS {

std::unique_ptr<Pairlist> create_pairlist(PairlistBuilderType type,
										  const Resource& resource,
										  size_t max_particles,
										  size_t max_pairs) {
	switch (type) {
	case PairlistBuilderType::ZOrder:
		return std::make_unique<ZOrderPairlist>(resource, max_particles, max_pairs);

	case PairlistBuilderType::CellList:
		MARS_Exception(ExceptionType::NotImplementedError, "CellList pairlist not yet implemented");

	case PairlistBuilderType::VerletList:
		MARS_Exception(ExceptionType::NotImplementedError,
					   "VerletList pairlist not yet implemented");

	case PairlistBuilderType::Hierarchical:
		MARS_Exception(ExceptionType::NotImplementedError,
					   "Hierarchical pairlist not yet implemented");

	default:
		MARS_Exception(ExceptionType::ValueError,
					   "Unknown pairlist type: {}",
					   static_cast<int>(type));
	}
}

} // namespace MARS
