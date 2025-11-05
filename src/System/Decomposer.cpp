#include "System/Decomposer.h"
#include "SimSystem.h"
#include "System/PatchManager.h"
#include "System/ZOrderDecomposer.h"
#include "Types/Types.h"
#include <algorithm>
#include <cmath>

namespace ARBD {

//================================================================================
// SpatialPatchDecomposer Implementation
//================================================================================

DecompositionPlan SpatialPatchDecomposer::decompose(SimSystem& sys) {
	const PeriodicBox& bcs = sys.get_boundary_conditions();
	const Length cutoff = Length(sys.get_cutoff());

	LOGINFO("Starting spatial patch decomposition with cutoff: {}", cutoff.value);

	// Get system bounds from boundary conditions
	Vector3 origin = bcs.get_origin();
	const auto& basis = bcs.get_basis();

	// Calculate system dimensions
	Vector3 min = origin;
	Vector3 max = origin + basis[0] + basis[1] + basis[2];
	Vector3 dr = max - min;

	LOGINFO("System bounds: min={}, max={}, dimensions={}", min, max, dr);

	// Calculate number of patches in each dimension
	Vector3_t<size_t> n_patches = {
		static_cast<size_t>(std::max(1.0f, std::ceil(dr.x / cutoff.value))),
		static_cast<size_t>(std::max(1.0f, std::ceil(dr.y / cutoff.value))),
		static_cast<size_t>(std::max(1.0f, std::ceil(dr.z / cutoff.value)))};

	size_t total_patches = n_patches.x * n_patches.y * n_patches.z;
	size_t num_resources = sys.get_resources().size();

	LOGINFO("Computed {} patches ({}x{}x{}) across {} resources",
			total_patches,
			n_patches.x,
			n_patches.y,
			n_patches.z,
			num_resources);

	// Create decomposition plan
	DecompositionPlan plan;
	plan.grid_dimensions = {static_cast<int>(n_patches.x),
							static_cast<int>(n_patches.y),
							static_cast<int>(n_patches.z)};
	plan.system_min = min;
	plan.system_max = max;

	const auto& periodicity = bcs.get_periodicity();
	plan.periodicity = {periodicity[0], periodicity[1], periodicity[2]};

	// Reserve space for patch data
	plan.patch_min_bounds.reserve(total_patches);
	plan.patch_max_bounds.reserve(total_patches);
	plan.patch_resources.reserve(total_patches);

	// Calculate patch boundaries and assign resources
	for (size_t idx = 0; idx < total_patches; ++idx) {
		Vector3_t<size_t> ijk = index_to_ijk(idx, n_patches.x, n_patches.y, n_patches.z);

		// Calculate patch boundaries
		Vector3 pmin =
			min + Vector3(ijk.x * cutoff.value, ijk.y * cutoff.value, ijk.z * cutoff.value);
		Vector3 pmax = pmin + Vector3(cutoff.value, cutoff.value, cutoff.value);

		// Clamp to system bounds
		pmax.x = std::min(pmax.x, max.x);
		pmax.y = std::min(pmax.y, max.y);
		pmax.z = std::min(pmax.z, max.z);

		plan.patch_min_bounds.push_back(pmin);
		plan.patch_max_bounds.push_back(pmax);

		// Assign resource (round-robin distribution)
		size_t resource_idx = idx % num_resources;
		plan.patch_resources.push_back(sys.get_resources()[resource_idx]);
	}

	// Validate the plan
	if (!plan.is_valid()) {
		throw Exception(ExceptionType::RuntimeError,
						SourceLocation(),
						"Generated invalid decomposition plan");
	}

	LOGINFO("Spatial patch decomposition completed: {} patches computed", plan.total_patches());
	return plan;
}

//================================================================================
// RecursiveBisectionPatchDecomposer Implementation
//================================================================================

DecompositionPlan RecursiveBisectionPatchDecomposer::decompose(SimSystem& sys) {
	LOGINFO("RecursiveBisectionPatchDecomposer: Starting recursive bisection patch decomposition");

	// TODO: Implement recursive bisection algorithm
	// This would involve:
	// 1. Analyzing particle distribution
	// 2. Recursively dividing domain to balance load
	// 3. Creating hierarchical decomposition structure
	// 4. Returning DecompositionPlan with computed patch layout

	throw Exception(ExceptionType::NotImplementedError,
					SourceLocation(),
					"RecursiveBisectionPatchDecomposer not yet implemented");
}

//================================================================================
// GeometricPatchDecomposer Implementation
//================================================================================

DecompositionPlan GeometricPatchDecomposer::decompose(SimSystem& sys) {
	LOGINFO("GeometricPatchDecomposer: Starting geometric patch decomposition");

	// TODO: Implement geometric decomposition
	// This would involve:
	// 1. Analyzing system geometry (membranes, interfaces, etc.)
	// 2. Creating partitions that respect geometric boundaries
	// 3. Optimizing for minimal cross-boundary communication
	// 4. Returning DecompositionPlan with computed patch layout

	throw Exception(ExceptionType::NotImplementedError,
					SourceLocation(),
					"GeometricPatchDecomposer not yet implemented");
}

//================================================================================
// Factory Function
//================================================================================

std::unique_ptr<PatchDecomposer> create_patch_decomposer(DecomposerType type) {
	switch (type) {
	case DecomposerType::Spatial:
		return std::make_unique<SpatialPatchDecomposer>();

	case DecomposerType::RecursiveBisection:
		return std::make_unique<RecursiveBisectionPatchDecomposer>();

	case DecomposerType::Geometric:
		return std::make_unique<GeometricPatchDecomposer>();

	case DecomposerType::ZOrder:
		return std::make_unique<ZOrderDecomposer>();

	default:
		throw Exception(ExceptionType::ValueError, SourceLocation(), "Unsupported decomposer type");
	}
}

} // namespace ARBD
