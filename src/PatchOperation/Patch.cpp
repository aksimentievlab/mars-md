/**
 * @file PatchOperation/Patch.cpp
 * @author Pin-Yi Li <pinyili2@illinois.edu>
 * @brief Implementation of modern patch-based spatial decomposition
 * @version 2.0
 * @date 2025-09-09
 */

#include "PatchOperation/Patch.h"
#include "MARSException.h"
#include "MARSLogger.h"
#include "Backend/Events.h"
#include "Interactions/DeviceBondedInteraction.h"
#include "Interactions/Nonbonded/Pairwise.h"
#include "PatchOperation/Integrator.h"
#include "PatchOperation/PairListKernels/ZOrderPairlist.h"
#include "PatchOperation/Pairlist.h"
namespace MARS {

//================================================================================
// Force Calculation Methods
//================================================================================

void Patch::ensure_bonded_topology_ready(const BondedInteractions& interactions,
										 const TablesRegistry& tables_registry,
										 size_t resource_idx) {
	if (!bonded_device_data_prepared_) {
		device_bonded_.copy_from_host(interactions);
		device_bonded_.link_tables(tables_registry, resource_idx);
		bonded_device_data_prepared_ = true;
		LOGINFO(
			"Patch {}: Prepared {} bonds, {} angles, {} dihedrals, {} exclusions, {} restraints "
			"for device",
			patch_id_,
			device_bonded_.num_bonds(),
			device_bonded_.num_angles(),
			device_bonded_.num_dihedrals(),
			device_bonded_.num_exclusions(),
			device_bonded_.num_restraints());
	}
}

Event Patch::calculate_nonbonded_forces(const NonBondedInteractions& interactions,
										const BondedInteractions& bonded_interactions,
										const DeviceParticleTypes& particle_types,
										const DeviceBuffer<BaseGridView<mars_real>>& grid_views,
										const TablesRegistry& tables_registry,
										size_t resource_idx,
										float pairlist_cutoff,
										float interaction_cutoff,
										size_t step,
										size_t rebuild_period,
										const Vector3& electric_field,
										int interpolation_scheme,
										bool compute_energy) {
	(void)interactions;

	particles_.clear_forces();

	const idx_t num_particles = particles_.size();
	if (num_particles == 0) {
		return Event(nullptr, resource_);
	}

	ensure_bonded_topology_ready(bonded_interactions, tables_registry, resource_idx);

	const ParticleView particle_view = particles_.view();
	const PeriodicBox* pbox = get_device_periodic_box();
	Event evt(nullptr, resource_);

	// Position-dependent force stashed the grid/field state so integrate_motion can fold it into
	// the integrator kernels (v1-faithful, one per-particle read). The uniform E field applies
	pmf_grid_configs_ = grid_views.empty() ? nullptr : grid_views.data();
	electric_field_ = electric_field;
	interpolation_scheme_ = interpolation_scheme;

	// Lazily build the type-pair -> table matrix; pairwise topology (which
	// type pairs have a tabulated potential) is static for the run
	if (!pairwise_nb_device_data_prepared_) {
		device_pair_nb_ =
			std::make_unique<DevicePairNonBondedInteractions>(particle_types.size(), resource_);
		device_pair_nb_->copy_pairwise_from_host(tables_registry.get_pair_nonbonded_types());
		device_pair_nb_->link_pairwise_tables(tables_registry, resource_idx);
		pairwise_nb_device_data_prepared_ = true;
	}

	// Rebuild the neighbor list every rebuild_period steps
	const bool at_period = (rebuild_period == 0 || (step - 1) % rebuild_period == 0);
	bool rebuild = at_period;
#ifdef ENABLE_ZORDER_REORDER
	// A reorder invalidates the old-order pairlist; force a full rebuild and skip
	// the displacement shortcut (the sorter's reference positions are now stale).
	const bool forced_rebuild = force_rebuild_;
	force_rebuild_ = false;
	rebuild = rebuild || forced_rebuild;
#else
	constexpr bool forced_rebuild = false;
#endif

	const float skin = pairlist_cutoff - interaction_cutoff;
	if (rebuild && !forced_rebuild && rebuild_period != 0 && pairlist_built_ && skin > 0.0f &&
		particle_count_ > 0) {
		rebuild =
			pairlist_->needs_update(particles_.pos(), particles_.pos(), particle_count_, skin);
		if (!rebuild) {
			LOGTRACE("Patch {}: skipping scheduled rebuild at step {} - max displacement "
					 "below half the {} A skin",
					 patch_id_,
					 step,
					 skin);
		}
	}

	if (rebuild) {
		pairlist_->build_pairlist(particles_.pos(), particle_count_, pairlist_cutoff);
		pairlist_built_ = true;

		// Resolve per-pair table indices once per rebuild (see Pairwise.h / dev_notes).
		const size_t np = pairlist_->get_num_pairs();
		if (pair_table_idx_.size() < np)
			pair_table_idx_.resize(np);
		if (np > 0) {
			launch_resolve_pair_tables(resource_,
									   pairlist_->get_neighbor_pairs().data(),
									   particle_view.type_id,
									   device_pair_nb_->pairwise_table_matrix(),
									   device_pair_nb_->pairwise_form_matrix(),
									   device_pair_nb_->num_particle_types(),
									   device_bonded_.exclusion_offsets(),
									   device_bonded_.exclusion_neighbors(),
									   device_bonded_.num_excl_particles(),
									   pair_table_idx_.data(),
									   np)
				.wait();
		}
	}

	evt = launch_pairwise_nonbonded(
		resource_,
		pairlist_->get_neighbor_pairs().data(),
		particle_view.pos,
		particle_view.ForceEnergy,
		pair_table_idx_.data(),
		device_pair_nb_->nonbonded_potentials(),
		pbox,
		compute_energy,
		pairlist_->get_num_pairs(),
		interaction_cutoff > 0.0f ? interaction_cutoff * interaction_cutoff : 0.0f);

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
		device_bonded_.num_dihedrals() == 0 && device_bonded_.num_restraints() == 0) {
		return Event(nullptr, resource_);
	}

	const ParticleView particle_view = particles_.view();
	const PeriodicBox* pbox = get_device_periodic_box();
	const bool get_energy = compute_energy;

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

	if (device_bonded_.num_restraints() > 0) {
		LOGTRACE("Patch {}: Computing {} restraints", patch_id_, device_bonded_.num_restraints());
		evt = launch_harmonic_restraints(resource_,
										 device_bonded_.restraint_particle_ids(),
										 particle_view.pos,
										 particle_view.ForceEnergy,
										 device_bonded_.restraint_positions(),
										 device_bonded_.restraint_spring_constants(),
										 pbox,
										 get_energy,
										 device_bonded_.num_restraints());
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
		// pmf already baked in launch_BD kernel.
		evt = launch_BD<float>(resource_,
							   particle_view,
							   particle_type_view,
							   dt,
							   step,
							   kT,
							   particle_count_,
							   periodic_box_,
							   base_seed,
							   base_ctr,
							   pmf_grid_configs_,
							   electric_field_,
							   interpolation_scheme_);
		break;

	case IntegratorType::Langevin:
		LOGTRACE("Patch {}: Integrating {} particles using Langevin dynamics (dt={}, kT={})",
				 patch_id_,
				 particle_count_,
				 dt,
				 kT);

		if (pmf_grid_configs_ != nullptr && particle_count_ > 0) {
			launch_PMF(resource_,
					   particle_view,
					   particle_type_view,
					   particle_count_,
					   pmf_grid_configs_,
					   electric_field_,
					   interpolation_scheme_)
				.wait();
		}

		if (deferred_kick_pending_) {
			launch_BAOAB_LastUpdate<float>(resource_,
										   particle_view,
										   particle_type_view,
										   dt,
										   step,
										   kT,
										   particle_count_,
										   base_seed,
										   base_ctr,
										   pmf_grid_configs_,
										   electric_field_,
										   interpolation_scheme_)
				.wait();
			deferred_kick_pending_ = false;
		}

		// The kick this step's B-A-O-A now owes will be paid at the top of the
		// next step (or earlier, by an output flush).
		deferred_kick_pending_ = true;

		evt = launch_BAOAB<float>(resource_,
								  particle_view,
								  particle_type_view,
								  periodic_box_,
								  dt,
								  step,
								  kT,
								  particle_count_,
								  base_seed,
								  base_ctr,
								  pmf_grid_configs_,
								  electric_field_,
								  interpolation_scheme_);
		break;

	default:
		throw Exception(ExceptionType::ValueError,
						SourceLocation(),
						"Unsupported integrator type: {}",
						static_cast<int>(integrator_type));
	}

	return evt;
}

Event Patch::finish_deferred_kick(float dt) {
	if (!deferred_kick_pending_) {
		return Event(nullptr, resource_);
	}
	deferred_kick_pending_ = false;

	return launch_BAOAB_LastUpdate<float>(resource_,
										  particles_.view(),
										  particle_types_ ? particle_types_->view()
														  : ParticleTypeView{},
										  dt,
										  /*current_step=*/0,
										  /*kT=*/0.0f,
										  particle_count_,
										  /*base_seed=*/0,
										  /*base_ctr=*/0,
										  pmf_grid_configs_,
										  electric_field_,
										  interpolation_scheme_);
}

//================================================================================
// Particle Communication Methods
//================================================================================

std::pair<Event, idx_t>
Patch::pack_halo_particles(DeviceBuffer<mars_real>& send_buffer, int direction, float halo_width) {
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

/**
 * @brief
 * @todo Launch kernel to unpack received halo particles:
 * - Unpack particle data from recv_buffer
 * - Store in ghost/halo region of particle data structures
 * - Update halo particle counts
 * @param recv_buffer
 * @param particle_count
 * @return Event
 */
Event Patch::unpack_halo_particles(const DeviceBuffer<mars_real>& recv_buffer,
								   idx_t particle_count) {
	// Ensure halo buffers are allocated
	if (!halo_buffers_) {
		halo_buffers_ = std::make_unique<HaloBuffers>(resource_, capacity_);
	}
	LOGTRACE("Patch {}: Unpacking {} halo particles", patch_id_, particle_count);

	return Event(nullptr, resource_);
}

Event Patch::sort_particles() {
	// Build pairlist which internally sorts particles (e.g., Z-order)
	// Use a small cutoff just for sorting purposes
	constexpr mars_real sorting_cutoff = mars_real(0.1); // Small value just to trigger sorting
	build_pairlist(sorting_cutoff);
	LOGTRACE("Patch {}: Sorted {} particles using pairlist ({})",
			 patch_id_,
			 particle_count_,
			 pairlist_->get_name());
	return Event(nullptr, resource_);
}

Event Patch::build_pairlist(float pairlist_cutoff) {
	// Build pairlist using current particle positions
	pairlist_->build_pairlist(particles_.pos(), particle_count_, pairlist_cutoff);
	LOGTRACE("Patch {}: Built pairlist ({}) with cutoff {} for {} particles, found {} pairs",
			 patch_id_,
			 pairlist_->get_name(),
			 pairlist_cutoff,
			 particle_count_,
			 pairlist_->get_num_pairs());
	return Event(nullptr, resource_);
}

#ifdef ENABLE_ZORDER_REORDER
Patch::~Patch() = default;

void Patch::reorder_particles() {
	const idx_t n = particle_count_;
	if (n == 0) {
		return;
	}

	// Morton box: prefer the periodic box, else this patch's spatial bounds.
	Vector3 box_min = periodic_box_.get_origin();
	Vector3 box_max = box_min + periodic_box_.get_box_size();

	if (!(box_max.x > box_min.x && box_max.y > box_min.y && box_max.z > box_min.z)) {
		LOGWARN("Patch {}: skipping reorder - degenerate Morton box", patch_id_);
		return;
	}

	if (!reorder_sorter_) {
		reorder_sorter_ =
			std::make_unique<ZOrderSort>(resource_, capacity_, ZOrderOptimizationMode::System);
	}
	reorder_sorter_->sort_particles(particles_.pos(), n, box_min, box_max);
	particles_.permute(*reorder_sorter_);
	device_bonded_.remap_particle_indices(*reorder_sorter_);
	force_rebuild_ = true;
	LOGTRACE("Patch {}: reordered {} particles into Morton order", patch_id_, n);
}

ZOrderSort& Patch::reorder_sorter() {
	return *reorder_sorter_;
}
#endif

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

} // namespace MARS
