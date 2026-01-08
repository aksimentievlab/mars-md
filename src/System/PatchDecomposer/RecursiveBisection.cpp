#include "System/PatchDecomposer/RecursiveBisection.h"
#include "Spatial.h"
#include "System/SimSystem.h"
#include "System/SystemState.h"
#include <chrono>

namespace ARBD {
DecompositionPlan RecursiveBisectionPatchDecomposer::decompose(SimSystem& system,
															   SystemState& state) {
	auto start_time = std::chrono::high_resolution_clock::now();

	LOGINFO("Starting recursive bisection decomposition");

	if (state.get_num_particles() == 0) {
		LOGWARN("No particles for recursive bisection, falling back to spatial decomposition");
		SpatialPatchDecomposer spatial_decomposer;
		return spatial_decomposer.decompose(system, state);
	}

	const auto& particles = state.get_global_particles();
	const auto& resources = system.get_resources();

	// Initialize root node with all particles
	BisectionNode root;
	root.min_bounds = Vector3(std::numeric_limits<float>::max());
	root.max_bounds = Vector3(std::numeric_limits<float>::lowest());
	root.level = 0;
	root.patch_id = 0;

	// Calculate bounding box
	for (size_t i = 0; i < particles.size(); ++i) {
		const Vector3& pos = particles.pos[i];
		root.min_bounds.x = std::min(root.min_bounds.x, pos.x);
		root.min_bounds.y = std::min(root.min_bounds.y, pos.y);
		root.min_bounds.z = std::min(root.min_bounds.z, pos.z);
		root.max_bounds.x = std::max(root.max_bounds.x, pos.x);
		root.max_bounds.y = std::max(root.max_bounds.y, pos.y);
		root.max_bounds.z = std::max(root.max_bounds.z, pos.z);
		root.particle_indices.push_back(static_cast<idx_t>(i));
	}

	// Perform recursive bisection
	std::vector<BisectionNode> result_nodes;
	recursive_bisect(root, particles, result_nodes, resources.size());

	// Create decomposition plan from result
	DecompositionPlan plan;
	plan.system_min = root.min_bounds;
	plan.system_max = root.max_bounds;
	plan.grid_dimensions = {static_cast<int>(result_nodes.size()), 1, 1}; // Linear arrangement

	const PeriodicBox& boundary = system.get_boundary_conditions();
	const auto& periodicity = boundary.get_periodicity();
	plan.periodicity = {periodicity[0], periodicity[1], periodicity[2]};

	// Fill plan data
	for (const auto& node : result_nodes) {
		plan.patch_min_bounds.push_back(node.min_bounds);
		plan.patch_max_bounds.push_back(node.max_bounds);
		plan.patch_particle_indices.push_back(node.particle_indices);

		size_t resource_idx = plan.patch_resources.size() % resources.size();
		plan.patch_resources.push_back(resources[resource_idx]);
	}

	// Calculate statistics
	auto end_time = std::chrono::high_resolution_clock::now();
	plan.statistics.decomposition_time_ms =
		std::chrono::duration<float, std::milli>(end_time - start_time).count();
	plan.statistics.total_particles = particles.size();

	std::vector<size_t> particles_per_patch;
	for (const auto& indices : plan.patch_particle_indices) {
		particles_per_patch.push_back(indices.size());
	}
	plan.statistics.particles_per_patch = particles_per_patch;

	if (!particles_per_patch.empty()) {
		auto minmax = std::minmax_element(particles_per_patch.begin(), particles_per_patch.end());
		plan.statistics.load_balance_factor =
			static_cast<float>(*minmax.second) / std::max(1.0f, static_cast<float>(*minmax.first));
	}

	LOGINFO("Recursive bisection completed: {} patches, load balance: {:.2f}, time: {:.2f} ms",
			plan.total_patches(),
			plan.statistics.load_balance_factor,
			plan.statistics.decomposition_time_ms);

	return plan;
}

void RecursiveBisectionPatchDecomposer::recursive_bisect(BisectionNode& node,
														 const HostParticleData& particles,
														 std::vector<BisectionNode>& result_nodes,
														 size_t target_patches) const {

	// Base case: if we have enough patches or too few particles, stop
	if (result_nodes.size() >= target_patches || node.particle_indices.size() < 10) {
		result_nodes.push_back(node);
		return;
	}

	// Find optimal split dimension
	int split_dim = find_optimal_split_dimension(node, particles);

	// Calculate split position (median)
	std::vector<float> coords;
	coords.reserve(node.particle_indices.size());
	for (idx_t idx : node.particle_indices) {
		switch (split_dim) {
		case 0:
			coords.push_back(particles.pos[idx].x);
			break;
		case 1:
			coords.push_back(particles.pos[idx].y);
			break;
		case 2:
			coords.push_back(particles.pos[idx].z);
			break;
		}
	}

	std::nth_element(coords.begin(), coords.begin() + coords.size() / 2, coords.end());
	float split_pos = coords[coords.size() / 2];

	// Create child nodes
	BisectionNode left_node, right_node;
	left_node.min_bounds = node.min_bounds;
	left_node.max_bounds = node.max_bounds;
	right_node.min_bounds = node.min_bounds;
	right_node.max_bounds = node.max_bounds;
	left_node.level = right_node.level = node.level + 1;

	// Set split boundaries
	switch (split_dim) {
	case 0:
		left_node.max_bounds.x = split_pos;
		right_node.min_bounds.x = split_pos;
		break;
	case 1:
		left_node.max_bounds.y = split_pos;
		right_node.min_bounds.y = split_pos;
		break;
	case 2:
		left_node.max_bounds.z = split_pos;
		right_node.min_bounds.z = split_pos;
		break;
	}

	// Distribute particles
	for (idx_t idx : node.particle_indices) {
		const Vector3& pos = particles.pos[idx];
		bool goes_left = false;

		switch (split_dim) {
		case 0:
			goes_left = pos.x < split_pos;
			break;
		case 1:
			goes_left = pos.y < split_pos;
			break;
		case 2:
			goes_left = pos.z < split_pos;
			break;
		}

		if (goes_left) {
			left_node.particle_indices.push_back(idx);
		} else {
			right_node.particle_indices.push_back(idx);
		}
	}

	// Recursively process children
	recursive_bisect(left_node, particles, result_nodes, target_patches);
	recursive_bisect(right_node, particles, result_nodes, target_patches);
}

int RecursiveBisectionPatchDecomposer::find_optimal_split_dimension(
	const BisectionNode& node,
	const HostParticleData& particles) const {

	// Choose dimension with largest extent
	Vector3 extent = node.max_bounds - node.min_bounds;

	if (extent.x >= extent.y && extent.x >= extent.z) {
		return 0; // x
	} else if (extent.y >= extent.z) {
		return 1; // y
	} else {
		return 2; // z
	}
}
} // namespace ARBD
