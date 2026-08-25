#include "ZOrder.h"
#include "MARSException.h"
#include "MARSLogger.h"
#include "System/PeriodicBox.h"
#include "System/SimSystem.h"
#include "System/SystemState.h"
#include <algorithm>
#include <chrono>
#include <limits>

namespace MARS {

ZOrderDecomposer::ZOrderDecomposer() : PatchDecomposer() {
	type_ = DecomposerType::ZOrder;

	// Set default configuration
	config_ = Config{};

	LOGINFO("Created ZOrderDecomposer with default configuration");
}

DecompositionPlan ZOrderDecomposer::decompose(SimSystem& sys, SystemState& state) {
	auto start_time = std::chrono::high_resolution_clock::now();

	const auto& resources = sys.get_resources();
	const PeriodicBox& bcs = sys.get_boundary_conditions();

	LOGINFO("Starting Z-order decomposition for {} resources", resources.size());

	// Step 1: Get particle positions from SimSystem (not from patches!)
	std::vector<Vector3> particle_positions = state.get_global_positions();
	size_t total_particles = particle_positions.size();

	if (total_particles == 0) {
		LOGWARN("No particles found for decomposition");
		// Return minimal plan - could use boundary conditions for system bounds
		DecompositionPlan plan;
		plan.system_min = bcs.get_origin();
		auto basis = bcs.get_basis();
		plan.system_max = plan.system_min + basis[0] + basis[1] + basis[2];
		plan.grid_dimensions = {1, 1, 1};
		plan.periodicity = {bcs.get_periodicity()[0],
							bcs.get_periodicity()[1],
							bcs.get_periodicity()[2]};
		plan.system_box = bcs;
		plan.set_periodic_box();
		return plan;
	}

	LOGINFO("Collected {} particles for decomposition", total_particles);

	// Step 2: Initialize buffers and compute bounding box
	const auto& first_resource = resources.empty() ? Resource(ResourceType::CPU, 0) : resources[0];

	if (!global_positions_ || global_positions_->size() < total_particles) {
		global_positions_ =
			std::make_unique<DeviceBuffer<Vector3>>(total_particles, first_resource);
		global_morton_codes_ =
			std::make_unique<DeviceBuffer<morton_t>>(total_particles, first_resource);
		global_indices_ = std::make_unique<DeviceBuffer<uint32_t>>(total_particles, first_resource);
		global_sorter_ = std::make_unique<ZOrderSort>(first_resource,
													  total_particles,
													  ZOrderOptimizationMode::System);
	}

	// Copy positions to device buffer
	global_positions_->copy_from_host(particle_positions.data(), total_particles);

	Vector3 global_box_min, global_box_max;
	if (config_.auto_bounding_box) {
		compute_global_bounding_box(*global_positions_,
									total_particles,
									global_box_min,
									global_box_max);
		LOGTRACE(
			"Computed global bounding box: [{:.3f}, {:.3f}, {:.3f}] to [{:.3f}, {:.3f}, {:.3f}]",
			global_box_min.x,
			global_box_min.y,
			global_box_min.z,
			global_box_max.x,
			global_box_max.y,
			global_box_max.z);
	} else {
		global_box_min = config_.manual_box_min;
		global_box_max = config_.manual_box_max;
		LOGTRACE("Using manual bounding box: [{:.3f}, {:.3f}, {:.3f}] to [{:.3f}, {:.3f}, {:.3f}]",
				 global_box_min.x,
				 global_box_min.y,
				 global_box_min.z,
				 global_box_max.x,
				 global_box_max.y,
				 global_box_max.z);
	}

	// Step 3: Sort particles globally by Morton code
	global_sorter_->sort_particles(*global_positions_,
								   total_particles,
								   global_box_min,
								   global_box_max);
	LOGTRACE("Sorted {} particles by Morton code", total_particles);

	// Step 4: Compute patch boundaries based on Z-order
	size_t num_patches = resources.size();
	auto patch_boundaries = compute_patch_boundaries(total_particles, num_patches);

	// Step 5: Create DecompositionPlan from Z-order results
	DecompositionPlan plan;
	plan.system_min = global_box_min;
	plan.system_max = global_box_max;

	// For Z-order, we need to compute actual spatial bounds for each patch
	// This is simplified - full implementation would compute bounds from Morton ranges
	const auto& periodicity = bcs.get_periodicity();
	plan.periodicity = {periodicity[0], periodicity[1], periodicity[2]};
	plan.system_box = bcs;
	plan.set_periodic_box();

	// Determine grid dimensions (for Z-order, patches may not form a regular grid)
	// For now, create a simple 1D arrangement
	plan.grid_dimensions = {static_cast<int>(num_patches), 1, 1};

	// Compute patch bounds from sorted particle positions
	plan.patch_min_bounds.reserve(num_patches);
	plan.patch_max_bounds.reserve(num_patches);
	plan.patch_resources.reserve(num_patches);

	// Get sorted positions from device
	std::vector<Vector3> sorted_positions(total_particles);
	global_positions_->copy_to_host(sorted_positions.data(), total_particles);

	size_t particles_per_patch = total_particles / num_patches;
	size_t remainder = total_particles % num_patches;

	size_t start_idx = 0;
	for (size_t patch_id = 0; patch_id < num_patches; ++patch_id) {
		size_t patch_size = particles_per_patch + (patch_id < remainder ? 1 : 0);
		size_t end_idx = std::min(start_idx + patch_size, total_particles);

		// Compute bounding box for this patch's particles
		Vector3 patch_min = Vector3(std::numeric_limits<float>::max());
		Vector3 patch_max = Vector3(std::numeric_limits<float>::lowest());

		for (size_t i = start_idx; i < end_idx; ++i) {
			const auto& pos = sorted_positions[i];
			patch_min.x = std::min(patch_min.x, pos.x);
			patch_min.y = std::min(patch_min.y, pos.y);
			patch_min.z = std::min(patch_min.z, pos.z);
			patch_max.x = std::max(patch_max.x, pos.x);
			patch_max.y = std::max(patch_max.y, pos.y);
			patch_max.z = std::max(patch_max.z, pos.z);
		}

		// Add small margin
		Vector3 margin = (patch_max - patch_min) * 0.01f;
		if (margin.x < 0.1f)
			margin.x = 0.1f;
		if (margin.y < 0.1f)
			margin.y = 0.1f;
		if (margin.z < 0.1f)
			margin.z = 0.1f;

		patch_min -= margin;
		patch_max += margin;

		plan.patch_min_bounds.push_back(patch_min);
		plan.patch_max_bounds.push_back(patch_max);

		// Assign resource
		size_t resource_idx = patch_id % resources.size();
		plan.patch_resources.push_back(resources[resource_idx]);

		start_idx = end_idx;
	}

	plan.set_periodic_box();

	// Validate plan
	if (!plan.is_valid()) {
		throw Exception(ExceptionType::RuntimeError,
						SourceLocation(),
						"Generated invalid Z-order decomposition plan");
	}

	auto end_time = std::chrono::high_resolution_clock::now();
	stats_.decomposition_time_ms =
		std::chrono::duration<double, std::milli>(end_time - start_time).count();
	stats_.global_box_min = global_box_min;
	stats_.global_box_max = global_box_max;

	LOGINFO("Z-order decomposition completed in {:.2f} ms - {} patches created",
			stats_.decomposition_time_ms,
			plan.total_patches());

	return plan;
}

// Removed collect_global_positions - particles now come directly from SimSystem
// This method was moved to decompose() since it's decomposition-specific logic

void ZOrderDecomposer::compute_global_bounding_box(const DeviceBuffer<Vector3>& positions,
												   size_t num_particles,
												   Vector3& box_min,
												   Vector3& box_max) {
	// Simple host-side computation (can be optimized with GPU kernel later)
	std::vector<Vector3> pos_host(num_particles);
	positions.copy_to_host(pos_host.data(), num_particles);

	// Initialize bounds
	box_min = Vector3(std::numeric_limits<float>::max());
	box_max = Vector3(std::numeric_limits<float>::lowest());

	// Compute bounding box
	for (size_t i = 0; i < num_particles; ++i) {
		const auto& pos = pos_host[i];
		box_min.x = std::min(box_min.x, pos.x);
		box_min.y = std::min(box_min.y, pos.y);
		box_min.z = std::min(box_min.z, pos.z);
		box_max.x = std::max(box_max.x, pos.x);
		box_max.y = std::max(box_max.y, pos.y);
		box_max.z = std::max(box_max.z, pos.z);
	}

	// Add small margin to avoid boundary issues
	Vector3 margin = (box_max - box_min) * 0.001f;
	box_min -= margin;
	box_max += margin;
}

std::vector<std::pair<morton_t, morton_t>>
ZOrderDecomposer::compute_patch_boundaries(size_t num_particles, size_t num_patches) {
	std::vector<std::pair<morton_t, morton_t>> boundaries;
	boundaries.reserve(num_patches);

	if (num_patches == 1) {
		// Special case: all particles go to single patch
		boundaries.emplace_back(0, std::numeric_limits<morton_t>::max());
		return boundaries;
	}

	// Get sorted Morton codes
	const auto& morton_codes = global_sorter_->get_morton_codes();

	// Simple equal-size partitioning
	// More sophisticated approaches could consider load balancing
	size_t particles_per_patch = num_particles / num_patches;
	size_t remainder = num_particles % num_patches;

	size_t start_idx = 0;
	for (size_t patch_id = 0; patch_id < num_patches; ++patch_id) {
		size_t patch_size = particles_per_patch + (patch_id < remainder ? 1 : 0);

		if (patch_size < config_.min_particles_per_patch && num_patches > 1) {
			patch_size = config_.min_particles_per_patch;
		}

		size_t end_idx = std::min(start_idx + patch_size, num_particles);

		morton_t start_morton = 0;
		morton_t end_morton = std::numeric_limits<morton_t>::max();

		if (start_idx < num_particles) {
			// Get Morton code from device (simplified - production code would be more efficient)
			morton_codes.copy_to_host(&start_morton, 1, start_idx);
		}

		if (end_idx < num_particles) {
			morton_codes.copy_to_host(&end_morton, 1, end_idx);
		}

		boundaries.emplace_back(start_morton, end_morton);
		start_idx = end_idx;

		if (start_idx >= num_particles) {
			break;
		}
	}

	return boundaries;
}

// Removed redistribute_particles and update_patch_metadata
// These are PatchManager responsibilities, not decomposer responsibilities
// Patches are created by PatchManager from the DecompositionPlan

void ZOrderDecomposer::validate_and_compute_stats(const SimSystem& sys, size_t num_patches) {
	// This method could be used for validation after decomposition
	// But it shouldn't depend on patches existing yet
	// Could compute stats from the DecompositionPlan instead

	stats_.num_patches = num_patches;
	stats_.load_imbalance_factor = 1.0f; // Would be computed from plan if needed

	LOGTRACE("Z-order decomposition stats: {} patches", num_patches);
}

} // namespace MARS
