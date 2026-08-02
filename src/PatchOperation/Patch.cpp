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
#include "Backend/Kernels.h"
#include "Interactions/Bonded/BondComputer.h"
#include "Interactions/Nonbonded/Pairwise.h"
#include "Interactions/Nonbonded/PmfKernels.h"
#include "PatchOperation/Integrator.h"
#include "PatchOperation/PairListKernels/ZOrderPairlist.h"
#include "PatchOperation/Pairlist.h"
namespace ARBD {

void Patch::set_periodic_box(const PeriodicBox* sim_box) {
	periodic_box_ = sim_box;
	if (sim_box) {
		periodic_box_device_ = DeviceBuffer<PeriodicBox>(1, resource_);
		periodic_box_device_.copy_from_host(std::vector<PeriodicBox>{*sim_box});
	} else {
		periodic_box_device_ = DeviceBuffer<PeriodicBox>();
	}

	// The pairlist needs the box too, not just the force kernel. Without it,
	// neighbor finding treats the domain as open: pairs separated by less than
	// the cutoff across a periodic face are never enumerated, so those
	// interactions are missing even though the force kernel would have applied
	// the minimum image convention to them. For a box only ~18 cutoffs wide
	// that affects a substantial fraction of the particles.
	if (auto* zpl = dynamic_cast<ZOrderPairlist*>(pairlist_.get())) {
		const bool fully_periodic =
			sim_box && sim_box->is_periodic(0) && sim_box->is_periodic(1) && sim_box->is_periodic(2);
		if (fully_periodic) {
			const auto bs = sim_box->get_box_size();
			zpl->set_periodic_box(Vector3(bs.x, bs.y, bs.z));
		} else {
			// Mixed or open boundaries: fall back to non-periodic neighbor
			// finding rather than wrapping in directions that are not periodic.
			zpl->set_periodic_box(Vector3(0.0f, 0.0f, 0.0f));
		}
	}
}

//================================================================================
// Force Calculation Methods
//================================================================================

void Patch::ensure_bonded_topology_ready(const BondedInteractions& interactions,
										 const TablesRegistry& tables_registry,
										 size_t resource_idx) {
	// Build the device-side bonded topology/table arrays once; bonded
	// topology doesn't change during the run, so this is cached across steps.
	if (!bonded_device_data_prepared_) {
		device_bonded_.copy_from_host(interactions);
		device_bonded_.link_tables(tables_registry, resource_idx);
		bonded_device_data_prepared_ = true;
		LOGINFO("Patch {}: Prepared {} bonds, {} angles, {} dihedrals, {} exclusions for device",
				patch_id_,
				device_bonded_.num_bonds(),
				device_bonded_.num_angles(),
				device_bonded_.num_dihedrals(),
				device_bonded_.num_exclusions());
	}
}

Event Patch::calculate_nonbonded_forces(const NonBondedInteractions& interactions,
										const BondedInteractions& bonded_interactions,
										const DeviceParticleTypes& particle_types,
										const DeviceBuffer<BaseGridView<float>>& grid_views,
										const TablesRegistry& tables_registry,
										size_t resource_idx,
										float cutoff,
										float interaction_cutoff,
										size_t step,
										size_t rebuild_period,
										float electric_field,
										int interpolation_scheme,
										bool compute_energy) {
	(void)interactions;

	particles_.clear_forces();

	const idx_t num_particles = particles_.size();
	if (num_particles == 0) {
		return Event(nullptr, resource_);
	}

	// Exclusions must be ready before the pairwise kernel runs below - this
	// function runs before calculate_bonded_forces() every step (see
	// SimManager::execute_force_calculation), so bonded topology can't rely
	// on that call having happened yet. Idempotent, cheap after the first call.
	ensure_bonded_topology_ready(bonded_interactions, tables_registry, resource_idx);

	const ParticleView particle_view = particles_.view();
	const PeriodicBox* pbox = get_device_periodic_box();
	Event evt(nullptr, resource_);

	if (!grid_views.empty()) {
		const ParticleTypeView type_view = particle_types.view();
		const KernelConfig config = KernelConfig::for_1d(num_particles, resource_);

		const ComputePMFKernel kernel{electric_field, interpolation_scheme};
		evt = launch_kernel(resource_,
							config,
							kernel,
							particle_view.pos,
							particle_view.type_id,
							particle_view.ForceEnergy,
							type_view,
							grid_views.data(),
							num_particles);
	}

	// Lazily build the type-pair -> table matrix; pairwise topology (which
	// type pairs have a tabulated potential) is static for the run, same as
	// bonded topology above.
	if (!pairwise_nb_device_data_prepared_) {
		device_pair_nb_ =
			std::make_unique<DevicePairNonBondedInteractions>(particle_types.size(), resource_);
		device_pair_nb_->copy_pairwise_from_host(tables_registry.get_pair_nonbonded_types());
		device_pair_nb_->link_pairwise_tables(tables_registry, resource_idx);
		pairwise_nb_device_data_prepared_ = true;
	}

	// Rebuild the neighbor list every rebuild_period steps (mirrors legacy's
	// decompPeriod gating: `s % decompPeriod == 0`), not every call - always
	// rebuilds on step == 1 since the pairlist starts out empty. Pairs found
	// at the last rebuild are reused (with cutoff/pairlist skin) on the steps
	// in between.
	if (rebuild_period == 0 || (step - 1) % rebuild_period == 0) {
		pairlist_->build_pairlist(particles_.pos(), particle_count_, cutoff);
	}

	evt = launch_pairwise_nonbonded(resource_,
									pairlist_->get_neighbor_pairs().data(),
									particle_view.pos,
									particle_view.ForceEnergy,
									particle_view.type_id,
									device_pair_nb_->pairwise_table_matrix(),
									device_pair_nb_->pairwise_form_matrix(),
									device_pair_nb_->nonbonded_potentials(),
									device_pair_nb_->num_particle_types(),
									device_bonded_.exclusion_offsets(),
									device_bonded_.exclusion_neighbors(),
									device_bonded_.num_excl_particles(),
									pbox,
									compute_energy,
									pairlist_->get_num_pairs(),
									interaction_cutoff > 0.0f
										? interaction_cutoff * interaction_cutoff
										: 0.0f);

	return evt;
}

Event Patch::calculate_bonded_forces(const BondedInteractions& interactions,
									 const DeviceParticleTypes& particle_types,
									 const TablesRegistry& tables_registry,
									 size_t resource_idx,
									 bool compute_energy) {
	(void)particle_types;

	if (particles_.size() == 0) {
		return Event(nullptr, resource_);
	}

	ensure_bonded_topology_ready(interactions, tables_registry, resource_idx);

	if (device_bonded_.num_bonds() == 0 && device_bonded_.num_angles() == 0 &&
		device_bonded_.num_dihedrals() == 0) {
		return Event(nullptr, resource_);
	}

	const ParticleView particle_view = particles_.view();
	const PeriodicBox* pbox = get_device_periodic_box();
	const bool get_energy = compute_energy;

	// Each kernel is submitted to the same in-order resource_ queue, so they
	// execute sequentially regardless of which Event is returned/waited on;
	// only the last launch's Event is kept, matching how callers already
	// treat these results (SimManager discards them with (void)).
	Event evt(nullptr, resource_);

	if (device_bonded_.num_bonds() > 0) {
		LOGTRACE("Patch {}: Computing {} bonds", patch_id_, device_bonded_.num_bonds());
		evt = launch_tabulated_bonds(resource_,
									device_bonded_.bond_indices(),
									particle_view.pos,
									particle_view.ForceEnergy,
									device_bonded_.bond_potentials(),
									device_bonded_.bond_table_indices(),
									device_bonded_.bond_forms(),
									pbox,
									get_energy,
									device_bonded_.num_bonds());
	}

	if (device_bonded_.num_angles() > 0) {
		LOGTRACE("Patch {}: Computing {} angles", patch_id_, device_bonded_.num_angles());
		evt = launch_tabulated_angles(resource_,
									 device_bonded_.angle_indices(),
									 particle_view.pos,
									 particle_view.ForceEnergy,
									 device_bonded_.angle_potentials(),
									 device_bonded_.angle_table_indices(),
									 device_bonded_.angle_forms(),
									 pbox,
									 get_energy,
									 device_bonded_.num_angles());
	}

	if (device_bonded_.num_dihedrals() > 0) {
		LOGTRACE("Patch {}: Computing {} dihedrals", patch_id_, device_bonded_.num_dihedrals());
		evt = launch_tabulated_dihedrals(resource_,
										device_bonded_.dihedral_indices(),
										particle_view.pos,
										particle_view.ForceEnergy,
										device_bonded_.dihedral_potentials(),
										device_bonded_.dihedral_table_indices(),
										device_bonded_.dihedral_forms(),
										pbox,
										get_energy,
										device_bonded_.num_dihedrals());
	}

	return evt;
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

	// Get the periodic box; wrapping needs its origin as well as its size, so
	// pass the box itself rather than just the extents. A default-constructed
	// box is non-periodic, so positions pass through unwrapped.
	PeriodicBox sim_box;
	if (periodic_box_) {
		sim_box = *periodic_box_;
	}

	// Derive the Philox stream key. Philox takes a 64-bit seed and a 32-bit
	// counter, so everything that must vary independently goes in the seed and
	// the counter carries the particle index alone.
	//
	// This replaces two defects. First, the seed was `patch_id_ * 1000000`,
	// which discarded the configured seed entirely - every run of a given input
	// drew the identical random sequence, and `seed` in the config file had no
	// effect. Second, the counter was `step * 1000000` truncated to 32 bits,
	// which wraps: 4295 * 10^6 mod 2^32 = 32704, so steps s and s+4295 started
	// only 32704 apart and, with 304910 particles drawing consecutive counters,
	// overlapped across 89% of their range and reused draws.
	//
	// Mixing with the golden-ratio constant decorrelates neighbouring seeds and
	// steps; patch_id occupies the high half so patches never collide.
	uint64_t base_seed = (static_cast<uint64_t>(base_seed_) * 0x9E3779B97F4A7C15ULL) ^
						 (static_cast<uint64_t>(patch_id_) << 32) ^
						 (static_cast<uint64_t>(step) * 0xBF58476D1CE4E5B9ULL);
	// Counter is the particle index (added in the kernels), so it cannot
	// overflow for any patch that fits in memory.
	uint32_t base_ctr = 0;

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
							   sim_box,
							   base_seed,
							   base_ctr);
		break;

	case IntegratorType::Langevin:
		LOGTRACE("Patch {}: Integrating {} particles using Langevin dynamics (dt={}, kT={})",
				 patch_id_,
				 particle_count_,
				 dt,
				 kT);

		// BAOAB is B-A-O-A-B, but launch_BAOAB only performs B-A-O-A: the
		// closing half-kick needs the force at the *new* positions, which does
		// not exist until the next force evaluation has run. Apply it here, at
		// the top of the following step, immediately before that step's opening
		// half-kick - the two use the same force and together form the full
		// kick of the standard merged-kick formulation.
		//
		// This was previously never launched at all (BAOAB_LastUpdate was
		// defined and instantiated but had no launch site), so every step
		// delivered only half the intended impulse while the O-step noise was
		// applied in full. The result was an effectively weakened potential:
		// the engine could not hold a condensed phase and drifted toward a
		// dilute one regardless of temperature.
		//
		// Skipped on the first step, which has no predecessor to complete.
		if (step > 1) {
			launch_BAOAB_LastUpdate<float>(resource_,
										   particle_view,
										   particle_type_view,
										   sim_box.get_box_size(),
										   dt,
										   step,
										   kT,
										   particle_count_,
										   base_seed,
										   base_ctr)
				.wait();
		}

		evt = launch_BAOAB<float>(resource_,
								  particle_view,
								  particle_type_view,
								  sim_box,
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
								   idx_t count,
								   bool need_energy) const {
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
	particles_.copy_to_host(temp_data, count, need_energy);

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
