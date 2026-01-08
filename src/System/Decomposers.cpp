/**
 * @file System/Decomposers.cpp
 * @author Pin-Yi Li <pinyili2@illinois.edu>
 * @brief Implementation of decomposer factory functions
 * @version 2.0
 * @date 2025-09-09
 */

#include "System/Decomposers.h"
#include "ARBDException.h"
#include "ARBDLogger.h"
#include "System/SimSystem.h"
#include "System/SystemState.h"

namespace ARBD {

//================================================================================
// Helper Functions (moved from old Decomposer.cpp)
//================================================================================

std::vector<std::vector<idx_t>>
PatchDecomposer::assign_particles_to_patches(const HostParticleData& particles,
											  const std::vector<Vector3>& patch_min_bounds,
											  const std::vector<Vector3>& patch_max_bounds) const {

	std::vector<std::vector<idx_t>> patch_assignments(patch_min_bounds.size());

	for (idx_t i = 0; i < particles.size(); ++i) {
		const Vector3& pos = particles.pos[i];

		// Find which patch contains this particle
		bool assigned = false;
		for (size_t patch_idx = 0; patch_idx < patch_min_bounds.size(); ++patch_idx) {
			const Vector3& min_bound = patch_min_bounds[patch_idx];
			const Vector3& max_bound = patch_max_bounds[patch_idx];

			if (pos.x >= min_bound.x && pos.x < max_bound.x &&
				pos.y >= min_bound.y && pos.y < max_bound.y &&
				pos.z >= min_bound.z && pos.z < max_bound.z) {
				patch_assignments[patch_idx].push_back(i);
				assigned = true;
				break;
			}
		}

		// If particle not assigned to any patch, assign to first patch (error handling)
		if (!assigned) {
			LOGWARN("Particle {} at position ({}, {}, {}) not assigned to any patch, using patch 0",
					i, pos.x, pos.y, pos.z);
			if (!patch_assignments.empty()) {
				patch_assignments[0].push_back(i);
			}
		}
	}

	return patch_assignments;
}

//================================================================================
// Factory Functions
//================================================================================

std::unique_ptr<PatchDecomposer> create_decomposer(DecomposerType type) {
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
		throw Exception(ExceptionType::ValueError,
						SourceLocation(),
						"Unknown decomposer type: {}",
						static_cast<int>(type));
	}
}

std::vector<DecomposerType> get_available_decomposer_types() {
	return {
		DecomposerType::Spatial,
		DecomposerType::RecursiveBisection,
		DecomposerType::Geometric,
		DecomposerType::ZOrder
	};
}

DecomposerType recommend_decomposer_type(const SimSystem& system, const SystemState& state) {
	size_t num_particles = state.get_num_particles();
	size_t num_resources = system.get_resources().size();

	// Simple heuristics for decomposer selection
	if (num_particles == 0) {
		return DecomposerType::Spatial; // Default for systems without particles
	}

	if (num_particles < 10000) {
		return DecomposerType::Spatial; // Simple spatial for small systems
	}

	if (num_resources > 8) {
		return DecomposerType::ZOrder; // Z-order for high parallelism
	}

	// Check for non-uniform distribution (could analyze particle density)
	// For now, default to recursive bisection for load balancing
	return DecomposerType::RecursiveBisection;
}

bool is_decomposer_available(DecomposerType type) {
	auto available_types = get_available_decomposer_types();
	return std::find(available_types.begin(), available_types.end(), type) != available_types.end();
}

const char* get_decomposer_name(DecomposerType type) {
	switch (type) {
	case DecomposerType::Spatial:
		return "Spatial Decomposer (Regular Grid)";
	case DecomposerType::RecursiveBisection:
		return "Recursive Bisection Decomposer";
	case DecomposerType::Geometric:
		return "Geometric Decomposer (Boundary-Aware)";
	case DecomposerType::ZOrder:
		return "Z-Order Decomposer";
	default:
		return "Unknown Decomposer";
	}
}

} // namespace ARBD