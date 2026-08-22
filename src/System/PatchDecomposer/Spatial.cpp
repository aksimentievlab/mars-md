#include "System/PatchDecomposer/Spatial.h"
#include "ARBDException.h"
#include "ARBDLogger.h"
#include "System/PeriodicBox.h"
#include "System/SystemState.h"
#include <algorithm>
#include <chrono>

namespace ARBD {

DecompositionPlan SpatialPatchDecomposer::decompose(SimSystem& system, SystemState& state) {
	auto start_time = std::chrono::high_resolution_clock::now();

	const PeriodicBox& boundary = system.get_boundary_conditions();
	const Length cutoff = system.get_pairlist_cutoff();
	const auto& resources = system.get_resources();

	LOGINFO("Starting spatial decomposition with cutoff: {} and {} resources",
			static_cast<float>(cutoff),
			resources.size());

	// Get system bounds from boundary conditions
	Vector3 origin = boundary.get_origin();
	const auto& basis = boundary.get_basis();

	Vector3 system_min = origin;
	Vector3 system_max = origin + basis[0] + basis[1] + basis[2];
	Vector3 system_size = system_max - system_min;

	LOGINFO("System bounds: min=({}, {}, {}), max=({}, {}, {})",
			system_min.x,
			system_min.y,
			system_min.z,
			system_max.x,
			system_max.y,
			system_max.z);

	// Calculate optimal grid dimensions
	auto grid_dims = calculate_grid_dimensions(system_size, cutoff, resources.size());
	size_t total_patches = static_cast<size_t>(grid_dims[0]) * grid_dims[1] * grid_dims[2];

	LOGINFO("Grid dimensions: {}x{}x{} = {} patches",
			grid_dims[0],
			grid_dims[1],
			grid_dims[2],
			total_patches);

	// Create decomposition plan
	DecompositionPlan plan;
	plan.grid_dimensions = grid_dims;
	plan.system_min = system_min;
	plan.system_max = system_max;

	const auto& periodicity = boundary.get_periodicity();
	plan.periodicity = {periodicity[0], periodicity[1], periodicity[2]};
	plan.system_box = boundary;

	// Reserve space for patch data
	plan.patch_min_bounds.reserve(total_patches);
	plan.patch_max_bounds.reserve(total_patches);
	plan.patch_resources.reserve(total_patches);
	plan.patch_particle_indices.reserve(total_patches);

	// Calculate patch sizes
	Vector3 patch_size = {system_size.x / grid_dims[0],
						  system_size.y / grid_dims[1],
						  system_size.z / grid_dims[2]};

	// Create patches
	for (int k = 0; k < grid_dims[2]; ++k) {
		for (int j = 0; j < grid_dims[1]; ++j) {
			for (int i = 0; i < grid_dims[0]; ++i) {
				// Calculate patch bounds
				Vector3 patch_min =
					system_min + Vector3(i * patch_size.x, j * patch_size.y, k * patch_size.z);
				Vector3 patch_max = patch_min + patch_size;

				// Clamp to system bounds
				patch_max.x = std::min(patch_max.x, system_max.x);
				patch_max.y = std::min(patch_max.y, system_max.y);
				patch_max.z = std::min(patch_max.z, system_max.z);

				plan.patch_min_bounds.push_back(patch_min);
				plan.patch_max_bounds.push_back(patch_max);

				// Assign resource (round-robin)
				size_t resource_idx = plan.patch_resources.size() % resources.size();
				plan.patch_resources.push_back(resources[resource_idx]);
			}
		}
	}

	// Assign particles to patches if particle data is available
	if (state.get_num_particles() > 0) {
		const auto& particles = state.get_global_particles();
		plan.patch_particle_indices =
			assign_particles_to_patches(particles, plan.patch_min_bounds, plan.patch_max_bounds);
	}

	// Calculate statistics
	auto end_time = std::chrono::high_resolution_clock::now();
	plan.statistics.decomposition_time_ms =
		std::chrono::duration<float, std::milli>(end_time - start_time).count();
	plan.statistics.total_particles = state.get_num_particles();
	plan.statistics.communication_volume =
		estimate_communication_cost(grid_dims, system_size, cutoff);

	// Calculate load balance
	if (!plan.patch_particle_indices.empty()) {
		std::vector<size_t> particles_per_patch;
		for (const auto& indices : plan.patch_particle_indices) {
			particles_per_patch.push_back(indices.size());
		}
		plan.statistics.particles_per_patch = particles_per_patch;

		if (!particles_per_patch.empty()) {
			auto minmax =
				std::minmax_element(particles_per_patch.begin(), particles_per_patch.end());
			plan.statistics.load_balance_factor = static_cast<float>(*minmax.second) /
												  std::max(1.0f, static_cast<float>(*minmax.first));
		}
	}

	plan.set_periodic_box();

	// Validate the plan
	if (!validate_decomposition_plan(plan)) {
		throw Exception(ExceptionType::RuntimeError,
						SourceLocation(),
						"Generated invalid spatial decomposition plan");
	}

	LOGINFO(
		"Spatial decomposition completed: {} patches, load balance factor: {:.2f}, time: {:.2f} ms",
		plan.total_patches(),
		plan.statistics.load_balance_factor,
		plan.statistics.decomposition_time_ms);

	return plan;
}

std::array<int, 3> SpatialPatchDecomposer::calculate_grid_dimensions(const Vector3& system_size,
																	 const Length& cutoff,
																	 size_t num_resources) const {

	// Start with cutoff-based dimensions
	std::array<int, 3> dims = {std::max(1, static_cast<int>(std::ceil(system_size.x / cutoff))),
							   std::max(1, static_cast<int>(std::ceil(system_size.y / cutoff))),
							   std::max(1, static_cast<int>(std::ceil(system_size.z / cutoff)))};

	size_t total_patches = static_cast<size_t>(dims[0]) * dims[1] * dims[2];

	// Adjust to better match number of resources
	if (total_patches > num_resources * 2) {
		// Too many patches, reduce (round up so we don't overshoot below num_resources)
		float scale_factor = std::cbrt(static_cast<float>(num_resources) / total_patches);
		dims[0] = std::max(1, static_cast<int>(std::ceil(dims[0] * scale_factor)));
		dims[1] = std::max(1, static_cast<int>(std::ceil(dims[1] * scale_factor)));
		dims[2] = std::max(1, static_cast<int>(std::ceil(dims[2] * scale_factor)));
	}

	// Guarantee enough patches to use every resource, growing the largest
	// dimension until parallelism is possible.
	total_patches = static_cast<size_t>(dims[0]) * dims[1] * dims[2];
	while (total_patches < num_resources) {
		int* largest = &dims[0];
		if (dims[1] > *largest)
			largest = &dims[1];
		if (dims[2] > *largest)
			largest = &dims[2];
		++(*largest);
		total_patches = static_cast<size_t>(dims[0]) * dims[1] * dims[2];
	}

	return dims;
}

float SpatialPatchDecomposer::estimate_communication_cost(const std::array<int, 3>& grid_dims,
														  const Vector3& system_size,
														  const Length& cutoff) const {

	// Estimate surface area requiring communication
	float patch_volume = (system_size.x / grid_dims[0]) * (system_size.y / grid_dims[1]) *
						 (system_size.z / grid_dims[2]);

	float patch_surface = 2.0f * ((system_size.x / grid_dims[0]) * (system_size.y / grid_dims[1]) +
								  (system_size.x / grid_dims[0]) * (system_size.z / grid_dims[2]) +
								  (system_size.y / grid_dims[1]) * (system_size.z / grid_dims[2]));

	// Communication volume proportional to surface area and cutoff width
	return patch_surface * static_cast<float>(cutoff);
}
} // namespace ARBD
