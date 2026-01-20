/**
 * @file PatchOperation/Patch.cpp
 * @author Pin-Yi Li <pinyili2@illinois.edu>
 * @brief Implementation of modern patch-based spatial decomposition
 * @version 2.0
 * @date 2025-09-09
 */

#include "PatchOperation/Patch.h"
#include "ARBDException.h"
#include "ARBDLogger.h"
#include "Backend/Events.h"
#include "Interactions/Nonbonded/PmfKernels.h"
#include "PatchOperation/Integrator.h"
#include "PatchOperation/Pairlist.h"
namespace ARBD {

//================================================================================
// Force Calculation Methods
//================================================================================

Event Patch::calculate_nonbonded_forces(const NonBondedInteractions& interactions,
										const DeviceParticleTypes& particle_types) {
	// Clear forces before computation
	particles_.clear_forces();

	// Get particle view for kernel launch
	idx_t num_particles = particles_.size();

	// For now, only compute PMF forces from grids
	// TODO: Add pairwise tabulated forces using neighbor pairs

	// Launch PMF kernel if PMF grids are present
	// This uses BaseGrid for position-dependent forces
	if (num_particles > 0) {
		KernelConfig config = KernelConfig::for_1d(num_particles, resource_);

		// TODO: Get grid data pointers and configs from interactions
		// For now, just log that we would compute PMF forces
		LOGTRACE("Patch {}: Computing PMF forces for {} particles", patch_id_, num_particles);

		// Placeholder: Would launch ComputePMFKernel here
		// Event evt = launch_kernel(resource_, config, ComputePMFKernel(),
		//                           particle_view.pos, particle_view.type_id,
		//                           particle_view.ForceEnergy, ...);
	}

	LOGTRACE("Patch {}: Computed non-bonded forces for {} particles", patch_id_, num_particles);

	// Return event for async execution
	return Event(nullptr, resource_);
}

Event Patch::calculate_bonded_forces(const BondedInteractions& interactions,
									 const DeviceParticleTypes& particle_types) {
	// Get particle view for kernel launch
	idx_t num_particles = particles_.size();

	// Launch tabulated bond force computation
	if (num_particles > 0) {
		LOGTRACE("Patch {}: Computing {} tabulated bonds", patch_id_, num_particles);

		// TODO: Launch TabulatedBondComputer kernel
		// Need device-side bond data structures (particle_indices, tables)
		// Event evt = launch_kernel(resource_, config, TabulatedBondComputer::compute,
		//                           bond_indices, particle_view.pos, particle_view.ForceEnergy,
		//                           bond_tables, periodic_box_, false);
	}

	// Launch tabulated angle force computation
	if (num_particles > 0) {
		LOGTRACE("Patch {}: Computing {} tabulated angles", patch_id_, num_particles);

		// TODO: Launch TabulatedAngleComputer kernel
	}

	// Launch tabulated dihedral force computation
	if (num_particles > 0) {
		LOGTRACE("Patch {}: Computing {} tabulated dihedrals", patch_id_, num_particles);

		// TODO: Launch TabulatedDihedralComputer kernel
	}

	LOGTRACE("Patch {}: Computed bonded forces for {} particles", patch_id_, num_particles);

	// Return event for async execution
	return Event(nullptr, resource_);
}

Event Patch::integrate_motion(float dt,
							  const Temperature& temperature,
							  IntegratorType integrator_type,
							  size_t step) {
	// Get particle view for integration
	auto particle_view = particles_.view();

	// Get temperature value (uniform for now, could be position-dependent)
	float kT = temperature.format == Temperature::Format::Value
				   ? temperature.kT
				   : temperature.kT; // TODO: Sample from grid if gridded (for now use constant)

	// Get box size from periodic box
	Vector3 box_size(0.0f, 0.0f, 0.0f);
	if (periodic_box_) {
		box_size = periodic_box_->get_box_size();
	}

	// Generate RNG seeds for deterministic, reproducible random numbers
	// base_seed: unique per patch to avoid correlation between patches
	// base_ctr: varies per timestep to ensure different random numbers each step
	// Within a timestep, each particle gets base_ctr + idx for uniqueness
	// Using large multiplier for base_seed to ensure no overlap between patches
	uint64_t base_seed = static_cast<uint64_t>(patch_id_) * 1000000ULL;
	// Use step number as counter offset, scaled to avoid overflow
	// Each step advances counter by max_particles to ensure no overlap
	// Assuming max 1M particles per patch, we use step * 1000000
	uint32_t base_ctr = static_cast<uint32_t>(step * 1000000ULL);

	// Get particle type view if available
	ParticleTypeView particle_type_view;
	if (particle_types_) {
		particle_type_view = particle_types_->view();
	}
	Event evt;

	// Launch integration kernel based on algorithm type
	switch (integrator_type) {
	case IntegratorType::Brownian:
		LOGTRACE("Patch {}: Integrating {} particles using Brownian dynamics (dt={}, kT={})",
				 patch_id_,
				 particle_count_,
				 dt,
				 kT);

		evt = launch_BD<float>(resource_,
							   particle_view,
							   particle_type_view,
							   dt,
							   step,
							   kT,
							   particle_count_,
							   box_size,
							   base_seed,
							   base_ctr);
		break;

	case IntegratorType::Langevin:
		LOGTRACE("Patch {}: Integrating {} particles using Langevin dynamics (dt={}, kT={})",
				 patch_id_,
				 particle_count_,
				 dt,
				 kT);

		evt = launch_BAOAB<float>(resource_,
								  particle_view,
								  particle_type_view,
								  box_size,
								  dt,
								  step,
								  kT,
								  particle_count_,
								  base_seed,
								  base_ctr);
		break;

	default:
		throw Exception(ExceptionType::ValueError,
						SourceLocation(),
						"Unsupported integrator type: {}",
						static_cast<int>(integrator_type));
	}

	return evt;
}

//================================================================================
// Particle Communication Methods
//================================================================================

std::pair<Event, idx_t>
Patch::pack_halo_particles(DeviceBuffer<float>& send_buffer, int direction, float halo_width) {
	// Ensure halo buffers are allocated
	if (!halo_buffers_) {
		halo_buffers_ = std::make_unique<HaloBuffers>(resource_, capacity_);
	}

	// Get particle view
	auto particle_view = particles_.view();

	// Calculate boundary region for this direction
	auto [halo_min, halo_max] = calculate_halo_bounds(direction, halo_width);

	// TODO: Launch kernel to pack particles in boundary region
	// - Identify particles in boundary region for given direction
	// - Pack position, velocity, type, and other data into send_buffer
	// - Return count of packed particles

	LOGTRACE("Patch {}: Packing halo particles for direction {} with width {}",
			 patch_id_,
			 direction,
			 halo_width,
			 particle_count_);

	idx_t packed_count = 0; // Placeholder - would be returned by kernel

	return {Event(nullptr, resource_), packed_count};
}

Event Patch::unpack_halo_particles(const DeviceBuffer<float>& recv_buffer, idx_t particle_count) {
	// Ensure halo buffers are allocated
	if (!halo_buffers_) {
		halo_buffers_ = std::make_unique<HaloBuffers>(resource_, capacity_);
	}

	// TODO: Launch kernel to unpack received halo particles
	// - Unpack particle data from recv_buffer
	// - Store in ghost/halo region of particle data structures
	// - Update halo particle counts

	LOGTRACE("Patch {}: Unpacking {} halo particles", patch_id_, particle_count);

	return Event(nullptr, resource_);
}

Event Patch::sort_particles() {
	// Build pairlist which internally sorts particles (e.g., Z-order)
	// Use a small cutoff just for sorting purposes
	float sorting_cutoff = 0.1f; // Small value just to trigger sorting
	build_pairlist(sorting_cutoff);
	LOGTRACE("Patch {}: Sorted {} particles using pairlist ({})",
			 patch_id_,
			 particle_count_,
			 pairlist_->get_name());
	return Event(nullptr, resource_);
}

Event Patch::build_pairlist(float cutoff) {
	// Build pairlist using current particle positions
	pairlist_->build_pairlist(particles_.pos(), particle_count_, cutoff);
	LOGTRACE("Patch {}: Built pairlist ({}) with cutoff {} for {} particles, found {} pairs",
			 patch_id_,
			 pairlist_->get_name(),
			 cutoff,
			 particle_count_,
			 pairlist_->get_num_pairs());
	return Event(nullptr, resource_);
}

Event Patch::update_pairlist() {
	// Update pairlist using current particle positions
	pairlist_->update_pairlist(particles_.pos(), particle_count_);
	LOGTRACE("Patch {}: Updated pairlist ({}) for {} particles, found {} pairs",
			 patch_id_,
			 pairlist_->get_name(),
			 particle_count_,
			 pairlist_->get_num_pairs());
	return Event(nullptr, resource_);
}

bool Patch::needs_pairlist_update(const DeviceBuffer<Vector3>& old_positions, float skin_distance) {
	// Check if pairlist needs updating based on particle displacement
	return pairlist_->needs_update(particles_.pos(), old_positions, particle_count_, skin_distance);
}

//================================================================================
// Host-Device Data Transfer Methods
//================================================================================

void Patch::copy_particles_from_host(const HostParticleData& host_data,
									 idx_t start_idx,
									 idx_t count) {
	if (start_idx + count > host_data.size()) {
		throw Exception(ExceptionType::ValueError,
						SourceLocation(),
						"Host data range exceeds available data: start={}, count={}, available={}",
						start_idx,
						count,
						host_data.size());
	}

	if (count > capacity_) {
		throw Exception(ExceptionType::RuntimeError,
						SourceLocation(),
						"Cannot copy {} particles to patch with capacity {}",
						count,
						capacity_);
	}

	// Copy particle data using DeviceParticle's bulk copy functionality
	// Create temporary host data slice
	HostParticleData temp_data;
	temp_data.resize(count);

	// Copy subset of host data
	for (idx_t i = 0; i < count; ++i) {
		size_t host_idx = start_idx + i;
		temp_data.global_id[i] = host_data.global_id[host_idx];
		temp_data.type_id[i] = host_data.type_id[host_idx];
		temp_data.pos[i] = host_data.pos[host_idx];
		temp_data.mom[i] = host_data.mom[host_idx];
		temp_data.force[i] = host_data.force[host_idx];
		temp_data.energy[i] = host_data.energy[host_idx];
		temp_data.orient[i] = host_data.orient[host_idx];
		temp_data.flags[i] = host_data.flags[host_idx];
	}

	// Copy to device
	particles_.copy_from_host(temp_data, count);
	particle_count_ = count;

	LOGTRACE("Patch {}: Copied {} particles from host (start_idx={})", patch_id_, count, start_idx);
}

void Patch::copy_particles_to_host(HostParticleData& host_data,
								   idx_t start_idx,
								   idx_t count) const {
	if (count > particle_count_) {
		throw Exception(ExceptionType::RuntimeError,
						SourceLocation(),
						"Cannot copy {} particles from patch containing only {}",
						count,
						particle_count_);
	}

	// Ensure host data has sufficient space
	if (host_data.size() < start_idx + count) {
		host_data.resize(start_idx + count);
	}

	// Copy from device to temporary host buffer
	HostParticleData temp_data;
	temp_data.resize(count);
	particles_.copy_to_host(temp_data, count);

	// Copy to target location in host data
	for (idx_t i = 0; i < count; ++i) {
		size_t host_idx = start_idx + i;
		host_data.global_id[host_idx] = temp_data.global_id[i];
		host_data.type_id[host_idx] = temp_data.type_id[i];
		host_data.pos[host_idx] = temp_data.pos[i];
		host_data.mom[host_idx] = temp_data.mom[i];
		host_data.force[host_idx] = temp_data.force[i];
		host_data.energy[host_idx] = temp_data.energy[i];
		host_data.orient[host_idx] = temp_data.orient[i];
		host_data.flags[host_idx] = temp_data.flags[i];
	}

	LOGTRACE("Patch {}: Copied {} particles to host (start_idx={})", patch_id_, count, start_idx);
}

//================================================================================
// Private Helper Methods
//================================================================================

void Patch::initialize_spatial_structures() {
	// Initialize halo buffers for communication
	halo_buffers_ = std::make_unique<HaloBuffers>(resource_, capacity_);

	LOGTRACE("Patch {}: Initialized spatial structures with capacity {}", patch_id_, capacity_);
}

void Patch::update_space_partition() {
	// Update any spatial acceleration structures when bounds change
	// This could include cell lists, neighbor lists, etc.

	LOGTRACE("Patch {}: Updated space partition with bounds [{}, {}] to [{}, {}]",
			 patch_id_,
			 min_bounds_.x,
			 min_bounds_.y,
			 min_bounds_.z,
			 max_bounds_.x,
			 max_bounds_.y,
			 max_bounds_.z);
}

} // namespace ARBD
