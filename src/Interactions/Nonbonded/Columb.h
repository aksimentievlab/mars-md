#pragma once
#include "Header.h"
#include "Interactions/Interactions.h"

namespace ARBD {

struct ColumbInteractionKernel {
	KERNEL_FUNC void operator()(ScalarForceEnergy force_energy,
								const Vector3* positions,
								const int2& neighbor_indices,
								const PeriodicBox* pbox) {
		Vector3 r_ij =
			pbox->wrapDiff(positions[neighbor_indices.y] - positions[neighbor_indices.x]);
		float distance = r_ij.length();
		ScalarForceEnergy fe = ColumbInteractionKernel::compute(r_ij, distance);
	}
	KERNEL_FUNC static ScalarForceEnergy compute(const Vector3& r_ij, float distance) {
		float2 f{1.0f / (distance * distance), 1.0f / distance};
		return ScalarForceEnergy{f};
	}
};

} // namespace ARBD
