#include "System/Decompose2Patch.h"
#include "SimSystem.h"
#include "System/PatchManager.h"
#include "Types/Types.h"
#include <algorithm>
#include <cmath>

namespace ARBD {

//================================================================================
// SpatialPatchDecomposer Implementation
//================================================================================

void SpatialPatchDecomposer::decompose(SimSystem& sys, const ResourceCollection& resources) {
	const BoundaryConditions& bcs = sys.get_boundary_conditions();
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
	size_t num_resources = resources.resources.size();

	LOGINFO("Creating {} patches ({}x{}x{}) across {} resources",
			total_patches,
			n_patches.x,
			n_patches.y,
			n_patches.z,
			num_resources);

	// Create patch boundaries
	std::vector<Vector3> patch_mins(total_patches);
	std::vector<Vector3> patch_maxs(total_patches);
	std::vector<Resource> patch_resources(total_patches);

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

		patch_mins[idx] = pmin;
		patch_maxs[idx] = pmax;

		// Assign resource (round-robin distribution)
		size_t resource_idx = idx % num_resources;
		patch_resources[idx] = resources.resources[resource_idx];
	}

	// Create PatchManager and assign particles to patches
	LOGINFO("Creating PatchManager and assigning particles to patches");

	// Create PatchManager
	sys.patch_manager_ = std::make_unique<PatchManager>(sys);

	// Initialize PatchManager with grid dimensions
	const auto& periodicity = bcs.get_periodicity();
	sys.patch_manager_->initialize(static_cast<int>(n_patches.x),
								   static_cast<int>(n_patches.y),
								   static_cast<int>(n_patches.z),
								   periodicity[0], // periodic_x
								   periodicity[1], // periodic_y
								   periodicity[2]  // periodic_z
	);

	// Get particle data from system
	const auto& particles = sys.get_particle_positions();
	if (!particles.empty()) {
		LOGINFO("Assigning {} particles to patches", particles.size());

		// TODO: Implement actual particle assignment to patches
		// This would use the ParticleAssignmentFunctor from DecomposeKernels.h
		// to assign particles to patches based on their positions

		// For now, just log that we have particles to assign
		LOGINFO("Particle assignment to patches not yet implemented");
	}

	LOGINFO("Spatial patch decomposition completed with PatchManager created");
}

//================================================================================
// RecursiveBisectionPatchDecomposer Implementation
//================================================================================

void RecursiveBisectionPatchDecomposer::decompose(SimSystem& sys,
												  const ResourceCollection& resources) {
	LOGINFO("RecursiveBisectionPatchDecomposer: Starting recursive bisection patch decomposition");

	// TODO: Implement recursive bisection algorithm
	// This would involve:
	// 1. Analyzing particle distribution
	// 2. Recursively dividing domain to balance load
	// 3. Creating hierarchical decomposition structure

	throw Exception(ExceptionType::NotImplementedError,
					SourceLocation(),
					"RecursiveBisectionPatchDecomposer not yet implemented");
}

//================================================================================
// GeometricPatchDecomposer Implementation
//================================================================================

void GeometricPatchDecomposer::decompose(SimSystem& sys, const ResourceCollection& resources) {
	LOGINFO("GeometricPatchDecomposer: Starting geometric patch decomposition");

	// TODO: Implement geometric decomposition
	// This would involve:
	// 1. Analyzing system geometry (membranes, interfaces, etc.)
	// 2. Creating partitions that respect geometric boundaries
	// 3. Optimizing for minimal cross-boundary communication

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

	default:
		throw Exception(ExceptionType::ValueError, SourceLocation(), "Unsupported decomposer type");
	}
}

} // namespace ARBD
