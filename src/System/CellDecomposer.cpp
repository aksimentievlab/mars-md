#include "ARBDLogger.h"
#include "Backend/Resource.h"
#include "SimSystem.h"
#include "Types/Types.h"

namespace ARBD {

void CellDecomposer::decompose(SimSystem& sys, const ResourceCollection& resources) {
	const BoundaryConditions& bcs = sys.get_boundary_conditions();
	const Length cutoff = Length(sys.get_cutoff());

	LOGINFO("Starting cell decomposition with cutoff: {}", cutoff.value);

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

	// TODO: Load particle data from system
	// For now, we'll create a placeholder for the particle assignment logic
	// This would typically come from the SimSystem's particle data

	LOGINFO("Patch boundaries calculated, ready for particle assignment");

	// Example of how particle assignment would work with the new Buffer system:
	/*
	// Create buffers for particle data (this would come from SimSystem)
	DeviceBuffer<Vector3> positions(num_particles, resource);
	DeviceBuffer<Vector3> momenta(num_particles, resource);
	DeviceBuffer<size_t> types(num_particles, resource);
	DeviceBuffer<size_t> global_indices(num_particles, resource);

	// Create output buffers
	DeviceBuffer<size_t> patch_assignments(num_particles, resource);
	DeviceBuffer<size_t> particle_counts(total_patches, resource);

	// Initialize particle counts to zero
	std::vector<size_t> zero_counts(total_patches, 0);
	particle_counts.copy_from_host(zero_counts.data(), total_patches);

	// Create buffers for patch boundaries
	DeviceBuffer<Vector3> patch_mins_buffer(patch_mins.size(), resource);
	DeviceBuffer<Vector3> patch_maxs_buffer(patch_maxs.size(), resource);
	patch_mins_buffer.copy_from_host(patch_mins.data(), patch_mins.size());
	patch_maxs_buffer.copy_from_host(patch_maxs.data(), patch_maxs.size());

	// Launch kernel to assign particles to patches
	ParticleAssignmentFunctor func{
		positions.data(),
		momenta.data(),
		types.data(),
		global_indices.data(),
		num_particles,
		patch_mins_buffer.data(),
		patch_maxs_buffer.data(),
		total_patches,
		patch_assignments.data(),
		particle_counts.data()
	};

	KernelConfig config;
	config.block_size = {256, 1, 1};

	auto event = launch_kernel(resource, num_particles, config, func);
	event.wait();

	// Now create patches with assigned particles
	std::vector<Patch> new_patches;
	new_patches.reserve(total_patches);

	for (size_t patch_idx = 0; patch_idx < total_patches; ++patch_idx) {
		// Get particle count for this patch
		std::vector<size_t> count(1);
		particle_counts.copy_to_host(count.data(), 1, patch_idx);
		size_t patch_particle_count = count[0];

		// Create patch with appropriate capacity
		Patch new_patch(patch_particle_count);
		new_patch.metadata.min = patch_mins[patch_idx];
		new_patch.metadata.max = patch_maxs[patch_idx];

		// TODO: Copy particles assigned to this patch
		// This would involve filtering the particle data based on patch_assignments

		new_patches.push_back(std::move(new_patch));
	}

	// Update system with new patches
	// sys.set_patches(std::move(new_patches));
	*/

	LOGINFO("Cell decomposition completed");
}

} // namespace ARBD
