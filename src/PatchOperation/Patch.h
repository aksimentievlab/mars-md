#pragma once
/**
 * @file PatchOperation/Patch.h
 * @author Pin-Yi Li <pinyili2@illinois.edu>
 * @brief Modern patch-based spatial decomposition unit
 * @version 2.0
 * @date 2025-09-09
 *
 * Redesigned to work with DeviceParticle system and SystemState/SimSystem architecture.
 * Each patch manages a spatial region with local particles using DeviceParticle for
 * optimal GPU memory layout and performance.
 */

#include "Backend/Events.h"
#include "Interactions/BondedInteraction.h"
#include "Interactions/DeviceBondedInteraction.h"
#include "Interactions/DevicePairNonBondedInteraction.h"
#include "Interactions/NonBondedInteraction.h"
#include "Interactions/TabulatedPotential.h"
#include "Objects/DeviceParticleManager.h"
#include "Objects/Grid.h"
#include "Objects/Tables.h"
#include "PatchOperation/Pairlist.h"
#include "PatchOperation/ZOrderKernels/ZOrderSort.h"
#include "System/PeriodicBox.h"
#include "System/SystemForward.h"
#include "Types/Types.h"
#include "PairListKernels/ZOrderPairlist.h"
#include <memory>
#include <utility>

namespace ARBD {

// Forward declarations
class SimSystem;
class ZOrderSort;

/**
 * @brief
 *
 * Each Patch represents a spatial region containing particles and manages:
 * - Local particle data using DeviceParticle for optimal memory layout
 * - Spatial bounds and neighbor relationships
 * - Particle migration and halo exchange
 * - Local force computation and time integration
 * - Resource binding for GPU acceleration
 */
class Patch {
  public:
	/**
	 * @brief Construct patch with specified parameters
	 * @param patch_id Unique patch identifier across the system
	 * @param capacity Maximum number of particles this patch can store
	 * @param resource Computational resource (GPU/CPU) assigned to this patch
	 * @param pairlist_type Type of pairlist to use for neighbor finding
	 */
	Patch(patch_t patch_id,
		  idx_t capacity,
		  const Resource& resource,
		  const PeriodicBox& periodic_box,
		  PairlistBuilderType pairlist_type = PairlistBuilderType::ZOrder)
		: patch_id_(patch_id), capacity_(capacity), resource_(resource), particles_(capacity, resource),
		  pair_table_idx_(capacity, resource), device_bonded_(resource) {
		set_periodic_box(periodic_box);
		pairlist_ = create_pairlist(pairlist_type, resource, capacity, kPairlistMaxPairs);
		initialize_spatial_structures();
	}

#ifdef ENABLE_ZORDER_REORDER
	// Out-of-line (=default in Patch.cpp) so unique_ptr<ZOrderSort> sees the
	// complete type at destruction.
	~Patch();
#endif

	//================================================================================
	// Communication Buffers for Particle Exchange
	//================================================================================
	struct HaloBuffers {
		DeviceBuffer<DeviceParticle> send_particles; ///< Outgoing halo particle data
		DeviceBuffer<DeviceParticle> recv_particles; ///< Incoming halo particle data
		DeviceBuffer<int> migration_flags;			 ///< Per-particle migration direction flags
		DeviceBuffer<int> migration_count; ///< Number of migrating particles per direction

		HaloBuffers(const Resource& resource, idx_t capacity)
			: send_particles(capacity, resource), // 8 floats per particle (pos+vel+type+flags)
			  recv_particles(capacity, resource), migration_flags(capacity, resource),
			  migration_count(6, resource) {} // 6 directions
	};

	std::unique_ptr<HaloBuffers> halo_buffers_{nullptr};

	/**
	 * @brief Set this patch's periodic wrapping box
	 *
	 * Also refreshes a device-resident copy of the box: bond/angle/dihedral
	 * force kernels (see BondComputer.h) take a `const PeriodicBox*` that
	 * they dereference on-device, so a host value like periodic_box_
	 * itself is not usable there.
	 * @param box This patch's periodic wrapping box - patch-extent basis,
	 *        with per-axis periodicity gated off on any split axis (see
	 *        DecompositionPlan::set_periodic_box())
	 */
	void set_periodic_box(const PeriodicBox& box){
		periodic_box_ = box;
		periodic_box_device_ = DeviceBuffer<PeriodicBox>(1, resource_);
		periodic_box_device_.copy_from_host(std::vector<PeriodicBox>{box});

	if (auto* zpl = dynamic_cast<ZOrderPairlist*>(pairlist_.get())) {
		const auto bs = box.get_box_size();
		zpl->set_periodic_box(Vector3(box.is_periodic(0) ? bs.x : 0.0f,
		                              box.is_periodic(1) ? bs.y : 0.0f,
		                              box.is_periodic(2) ? bs.z : 0.0f),
		                      box.get_origin());
	}
	};

	/**
	 * @brief Set the run-wide RNG seed (SimSystem::get_base_seed()).
	 *
	 * Without this the integrators derive their Philox seed from the patch id
	 * alone, so the configured seed is discarded and every run of a given input
	 * draws the identical random sequence.
	 */
	void set_base_seed(uint64_t seed) {
		base_seed_ = seed;
	}

	/**
	 * @brief Get a device-resident pointer to the periodic box, or nullptr if unset
	 */
	const PeriodicBox* get_device_periodic_box() const {

		return periodic_box_device_.size() > 0 ? periodic_box_device_.device_data() : nullptr;
	}

	/**
	 * @brief Set the current number of active particles in this patch
	 * @param count New particle count (must be <= capacity)
	 */
	void set_particle_count(idx_t count) {
		if (count <= capacity_) {
			particle_count_ = count;
		}
	}

	/**
	 * @brief Set halo region thickness for ghost particle exchange
	 * @param thickness Thickness of halo region (typically cutoff radius)
	 */
	void set_halo_thickness(float thickness) {
		halo_thickness_ = thickness;
	}

	/**
	 * @brief Set device particle types for this patch
	 * @param particle_types Device particle types (moved into patch)
	 */
	void set_particle_types(std::unique_ptr<DeviceParticleTypes> particle_types) {
		particle_types_ = std::move(particle_types);
	}

	/**
	 * @brief Get unique patch identifier
	 */
	patch_t get_patch_id() const {
		return patch_id_;
	}

	/**
	 * @brief Get current number of active particles
	 */
	idx_t get_particle_count() const {
		return particle_count_;
	}

	/**
	 * @brief Get patch capacity
	 */
	idx_t get_capacity() const {
		return capacity_;
	}

	/**
	 * @brief Check if patch has space for more particles
	 */
	bool has_space() const {
		return particle_count_ < capacity_;
	}

	/**
	 * @brief Compute non-bonded forces for particles in this patch
	 *
	 * Evaluates both the PMF/grid term (position-dependent, per-particle) and
	 * the pairwise tabulated nonbonded term (residue-pair potentials, via a
	 * neighbor list rebuilt each call - see Pairlist.h). `bonded_interactions`
	 * and `tables_registry`/`resource_idx` are needed here (not just in
	 * calculate_bonded_forces) because exclusions must be ready before the
	 * pairwise kernel runs, and this function is called first each step (see
	 * SimManager::execute_force_calculation) - so bonded topology is prepared
	 * lazily from whichever of the two calculate_* calls runs first.
	 * @param interactions Non-bonded interaction parameters from SimSystem (currently unused -
	 *        pairwise metadata is sourced from tables_registry, see get_pair_nonbonded_types())
	 * @param bonded_interactions Bonded topology, needed to populate exclusions
	 * @param particle_types Device particle type data from SimSystem
	 * @param tables_registry Tabulated potential tables (bonded and nonbonded)
	 * @param resource_idx This patch's index into tables_registry's per-resource arrays
	 * @param pairlist_cutoff Neighbor-list search radius used to rebuild the pairlist - should
	 *        include the pairlist skin (SimSystem::get_pairlist_cutoff(), i.e.
	 *        interaction cutoff + pairlistDistance), not just the raw interaction
	 *        cutoff, since pairs found at one rebuild are reused for rebuild_period
	 *        steps and particles can drift in the meantime.
	 * @param interaction_cutoff The radius beyond which pairs do not interact
	 *        (SimSystem::get_cutoff(), i.e. `pairlist_cutoff` above minus the skin). The
	 *        force kernel rejects pairs beyond this, which the pairlist cannot
	 *        do for it - the skin is precisely what lets the list outlive a
	 *        single step. Pass <= 0 to evaluate every pair in the list.
	 * @param step Current simulation step (1-indexed, matches SimManager's loop)
	 * @param rebuild_period Rebuild the pairlist every `rebuild_period` steps (mirrors
	 *        legacy's decompPeriod - see SimSystem::neighbor_list_rebuild_period),
	 *        instead of every call. The pairlist always rebuilds on the first call
	 *        (step == 1) since it starts out empty.
	 * @return Event for async GPU execution
	 */
	Event calculate_nonbonded_forces(const NonBondedInteractions& interactions,
									 const BondedInteractions& bonded_interactions,
									 const DeviceParticleTypes& particle_types,
									 const DeviceBuffer<BaseGridView<arbd_real>>& grid_views,
									 const TablesRegistry& tables_registry,
									 size_t resource_idx,
									 float pairlist_cutoff,
									 float interaction_cutoff,
									 size_t step,
									 size_t rebuild_period,
									 const Vector3& electric_field = Vector3{0.0f, 0.0f, 0.0f},
									 int interpolation_scheme = 0,
									 bool compute_energy = false);

	/**
	 * @brief Compute bonded forces (bonds, angles, dihedrals) for particles in this patch
	 *
	 * Tabulated bonds/angles/dihedrals only for now; analytical forms are
	 * loaded (BondedInteractions still records them) but skipped by the
	 * Tabulated*Computer kernels - see AnalyticalBondComputer for the
	 * per-BondTypeId launch that would be needed to cover them too.
	 *
	 * Bond/Angle/Dihedral ind1..ind4 are global particle indices; this
	 * assumes they equal this patch's local particle-array indices, which
	 * only holds while every particle lives in a single patch (true today -
	 * true multi-patch decomposition, with a global-to-local index remap,
	 * doesn't exist yet).
	 * @param interactions Bonded interaction parameters from SystemState
	 * @param particle_types Device particle type data from SimSystem
	 * @param tables_registry Tabulated potential tables (device arrays must
	 *        already be built via TablesRegistry::build_device_arrays())
	 * @param resource_idx This patch's index into tables_registry's per-resource arrays
	 * @return Event for async GPU execution
	 */
	Event calculate_bonded_forces(const BondedInteractions& interactions,
								  const DeviceParticleTypes& particle_types,
								  const TablesRegistry& tables_registry,
								  size_t resource_idx,
								  bool compute_energy = false);

	/**
	 * @brief Get the bonded-pair exclusions consulted by the pairwise nonbonded kernel
	 */
	DEVICE_PTR(const int2) get_exclusions() const {
		return device_bonded_.exclusion_pairs();
	}

	/**
	 * @brief Number of excluded pairs currently stored for this patch
	 */
	idx_t get_num_exclusions() const {
		return device_bonded_.num_exclusions();
	}

	/**
	 * @brief Integrate particle equations of motion
	 * @param dt Timestep from SimSystem
	 * @param temperature Temperature from SimSystem (position-dependent if gridded)
	 * @param integrator_type Integration algorithm from SimSystem
	 * @param step Current simulation step (for RNG counter)
	 * @return Event for async GPU execution
	 */
	Event integrate_motion(float dt,
						   const Temperature& temperature,
						   IntegratorType integrator_type,
						   size_t step = 0);

	/**
	 * @brief Apply BAOAB's outstanding closing half-kick, if one is pending.
	 *
	 * @param dt Timestep, matching the one integrate_motion() was called with
	 * @return Event for async GPU execution; a no-op event if nothing is pending
	 */
	Event finish_deferred_kick(float dt);

	/**
	 * @brief Whether a BAOAB closing half-kick is outstanding (see finish_deferred_kick)
	 */
	bool has_deferred_kick() const {
		return deferred_kick_pending_;
	}

	/**
	 * @brief
	 *
	 */
	idx_t get_remaining_capacity() const {
		return capacity_ - particle_count_;
	}

	/**
	 * @brief Get patch minimum spatial bounds (this patch's periodic box origin)
	 */
	const Vector3& get_min_bounds() const {
		return periodic_box_.get_origin();
	}

	/**
	 * @brief Get patch maximum spatial bounds (origin + this patch's box size)
	 */
	Vector3 get_max_bounds() const {
		return periodic_box_.get_origin() + periodic_box_.get_box_size();
	}

	/**
	 * @brief Get halo region thickness for ghost particles
	 * @return Halo thickness (typically equal to cutoff radius)
	 */
	float get_halo_thickness() const {
		return halo_thickness_;
	}

	/**
	 * @brief Check if position is outside patch core region (in halo)
	 * @param position Position to test
	 * @return True if position is in halo/ghost region
	 */
	bool is_in_halo_region(const Vector3& position) const {
		const Vector3& lo = get_min_bounds();
		const Vector3 hi = get_max_bounds();
		return position.x < lo.x || position.x >= hi.x || position.y < lo.y || position.y >= hi.y ||
			   position.z < lo.z || position.z >= hi.z;
	}

	/**
	 * @brief Get computational resource assigned to this patch
	 * @return Resource (GPU/CPU) for this patch
	 */
	const Resource& get_resource() const {
		return resource_;
	}

	/**
	 * @brief Determine if particle needs migration and in which direction
	 * @param position Particle position to test
	 * @return Migration direction: 0=x-, 1=x+, 2=y-, 3=y+, 4=z-, 5=z+, -1=no migration
	 */
	int check_migration_direction(const Vector3& position) const {
		const Vector3& lo = get_min_bounds();
		const Vector3 hi = get_max_bounds();
		if (position.x < lo.x)
			return 0; // x-
		if (position.x >= hi.x)
			return 1; // x+
		if (position.y < lo.y)
			return 2; // y-
		if (position.y >= hi.y)
			return 3; // y+
		if (position.z < lo.z)
			return 4; // z-
		if (position.z >= hi.z)
			return 5; // z+
		return -1;	  // No migration needed
	}

	/**
	 * @brief Check if a position is within the boundary region that should be sent to neighbor
	 * @param position Position to check
	 * @param direction Direction to neighbor (0=x-, 1=x+, 2=y-, 3=y+, 4=z-, 5=z+)
	 * @param halo_width Width of the halo region
	 * @return True if position is in boundary region
	 */
	bool is_in_boundary_region(const Vector3& position, int direction, float halo_width) const {
		const Vector3& lo = get_min_bounds();
		const Vector3 hi = get_max_bounds();
		switch (direction) {
		case 0: // x-
			return position.x >= lo.x && position.x < lo.x + halo_width;
		case 1: // x+
			return position.x >= hi.x - halo_width && position.x < hi.x;
		case 2: // y-
			return position.y >= lo.y && position.y < lo.y + halo_width;
		case 3: // y+
			return position.y >= hi.y - halo_width && position.y < hi.y;
		case 4: // z-
			return position.z >= lo.z && position.z < lo.z + halo_width;
		case 5: // z+
			return position.z >= hi.z - halo_width && position.z < hi.z;
		default:
			return false;
		}
	}

	/**
	 * @brief Calculate halo region bounds for specific direction
	 * @param direction Direction index (0=x-, 1=x+, 2=y-, 3=y+, 4=z-, 5=z+)
	 * @param halo_width Width of halo region
	 * @return Pair of (halo_min, halo_max) bounds
	 */
	std::pair<Vector3, Vector3> calculate_halo_bounds(int direction, float halo_width) const {
		const Vector3 lo = get_min_bounds();
		const Vector3 hi = get_max_bounds();
		Vector3 halo_min = lo;
		Vector3 halo_max = hi;

		switch (direction) {
		case 0: // x-
			halo_max.x = lo.x;
			halo_min.x = lo.x - halo_width;
			break;
		case 1: // x+
			halo_min.x = hi.x;
			halo_max.x = hi.x + halo_width;
			break;
		case 2: // y-
			halo_max.y = lo.y;
			halo_min.y = lo.y - halo_width;
			break;
		case 3: // y+
			halo_min.y = hi.y;
			halo_max.y = hi.y + halo_width;
			break;
		case 4: // z-
			halo_max.z = lo.z;
			halo_min.z = lo.z - halo_width;
			break;
		case 5: // z+
			halo_min.z = hi.z;
			halo_max.z = hi.z + halo_width;
			break;
		}

		return {halo_min, halo_max};
	}

	/**
	 * @brief Get read-only access to local particle data
	 */
	const DeviceParticle& get_particles() const {
		return particles_;
	}

	/**
	 * @brief Get mutable access to local particle data
	 */
	DeviceParticle& get_particles() {
		return particles_;
	}

	/**
	 * @brief Pack boundary particles for halo exchange with neighbor
	 * @param send_buffer Output buffer for packed data
	 * @param direction Neighbor direction (0=x-, 1=x+, 2=y-, 3=y+, 4=z-, 5=z+)
	 * @param halo_width Width of halo region (typically cutoff radius)
	 * @return Event for async packing and number of particles packed
	 */
	std::pair<Event, idx_t>
	pack_halo_particles(DeviceBuffer<arbd_real>& send_buffer, int direction, float halo_width);

	/**
	 * @brief Unpack received halo particles from neighbor
	 * @param recv_buffer Input buffer with packed particle data
	 * @param particle_count Number of particles to unpack
	 * @return Event for async unpacking
	 */
	Event unpack_halo_particles(const DeviceBuffer<arbd_real>& recv_buffer, idx_t particle_count);

	/**
	 * @brief Sort particles using pairlist's spatial sorting (e.g., Z-order curve)
	 * @return Event for async sorting
	 */
	Event sort_particles();

	/**
	 * @brief Build pairlist for neighbor finding
	 * @param pairlist_cutoff Pairlist search radius (force cutoff + skin)
	 * @return Event for async pairlist building
	 */
	Event build_pairlist(float pairlist_cutoff);

#ifdef ENABLE_ZORDER_REORDER
	/// Morton-reorder the canonical particle SoA and remap bonded topology to the
	/// new slots; forces a pairlist rebuild next force call. See dev_notes.
	void reorder_particles();

	/// The sorter driving the last reorder_particles(); its inverse map lets a
	/// parallel subsystem (e.g. RigidBodyManager) remap its own particle indices.
	ZOrderSort& reorder_sorter();

	/// Whether device bonded topology is populated (reorder needs it before remap).
	bool is_bonded_prepared() const {
		return bonded_device_data_prepared_;
	}
#endif

	/**
	 * @brief Update pairlist if particles have moved significantly
	 * @return Event for async pairlist update
	 */
	Event update_pairlist();

	/**
	 * @brief Check if pairlist needs updating based on particle displacement
	 * @param old_positions Previous particle positions
	 * @param skin_distance Skin distance for update criterion
	 * @return True if update is needed
	 */
	bool needs_pairlist_update(const DeviceBuffer<Vector3>& old_positions, float skin_distance);

	const Pairlist& get_pairlist() const {
		return *pairlist_;
	}

	Pairlist& get_pairlist() {
		return *pairlist_;
	}

	/**
	 * @brief Execute integration for particles in this patch
	 * @param dt Timestep from SimSystem
	 * @param temperature Temperature from SimSystem (position-dependent if gridded)
	 * @param integrator_type Integration algorithm from SimSystem
	 * @return Event for async integration
	 */
	Event
	execute_integration(float dt, const Temperature& temperature, IntegratorType integrator_type);

	/**
	 * @brief Copy particle data from host to this patch's device storage
	 * @param host_data Host particle data from SystemState
	 * @param start_idx Starting index in host data
	 * @param count Number of particles to copy
	 */
	void copy_particles_from_host(const HostParticleData& host_data, idx_t start_idx, idx_t count);

	/**
	 * @brief Copy particle data from this patch to host storage
	 * @param host_data Output host particle data
	 * @param start_idx Starting index in host data for output
	 * @param count Number of particles to copy
	 * @param need_energy Also copy/unpack ForceEnergy (skipped by default -
	 *        DCD/restart writers never read force or energy; only the
	 *        energy-output path needs it)
	 */
	void copy_particles_to_host(HostParticleData& host_data,
								idx_t start_idx,
								idx_t count,
								bool need_energy = false) const;

  private:
	/**
	 * @brief Initialize spatial acceleration structures
	 */
	void initialize_spatial_structures();

	/**
	 * @brief Lazily build device-side bonded topology (bonds/angles/dihedrals/exclusions/tables)
	 *
	 * Extracted so both calculate_bonded_forces() and calculate_nonbonded_forces()
	 * can call it - the latter needs exclusions ready before evaluating the
	 * pairwise term, but runs first each step (see SimManager::execute_force_calculation).
	 * Idempotent via bonded_device_data_prepared_.
	 */
	void ensure_bonded_topology_ready(const BondedInteractions& interactions,
									  const TablesRegistry& tables_registry,
									  size_t resource_idx);
	//================================================================================
	// Core Patch Properties
	//================================================================================
	patch_t patch_id_;					///< Unique patch identifier
	uint64_t base_seed_{0};				///< Run-wide RNG seed (see set_base_seed)
	bool deferred_kick_pending_{false}; ///< BAOAB closing half-kick owed (see finish_deferred_kick)
	idx_t capacity_;					///< Maximum particle storage capacity
	idx_t particle_count_{0};			///< Current number of active particles
	Resource resource_;					///< Computational resource (GPU/CPU)

	//================================================================================
	// Particle Data and Spatial Organization
	//================================================================================
	HostParticleData host_particles_;	 ///< used for staging on HOST.
	DeviceParticle particles_;			 ///< Local particle data in SoA format
	std::unique_ptr<Pairlist> pairlist_; ///< Pairlist for neighbor finding and spatial organization
	DeviceBuffer<int>
		pair_table_idx_; ///< Per-pair tabulated-table index, resolved each rebuild (-1 = skip)
	bool pairlist_built_{
		false}; ///< True once pairlist_ has been built; gates the displacement skip check
#ifdef ENABLE_ZORDER_REORDER
	std::unique_ptr<ZOrderSort>
		reorder_sorter_;		///< Morton sorter for canonical reorder (System mode)
	bool force_rebuild_{false}; ///< Set by reorder_particles(); forces next pairlist rebuild
#endif
	std::unique_ptr<DeviceParticleTypes> particle_types_; ///< Device buffers for all particle types

	float halo_thickness_{arbd_real(0.0)}; ///< Halo region thickness for ghost particles

	//================================================================================
	// Spatial Bounds and Geometry
	//================================================================================
	// min/max bounds are derived from periodic_box_ (see get_min_bounds/get_max_bounds) -
	// no separate storage, so they can never drift out of sync with the box.

	//================================================================================
	// Periodic wrapping box (owned by value - see set_periodic_box)
	//================================================================================
	PeriodicBox periodic_box_{}; ///< This patch's periodic wrapping box (patch-extent basis)
	DeviceBuffer<PeriodicBox> periodic_box_device_; ///< Device-resident copy for kernels that
													///< dereference PeriodicBox* on-device

	//================================================================================
	// Position-dependent force state (set in calculate_nonbonded_forces, consumed by
	// integrate_motion, which fuses the PMF/force-grid + uniform E term into the
	// integrator kernels - see Pmf.h / v1's compute_position_dependent_force).
	//================================================================================
	const BaseGridView<arbd_real>* pmf_grid_configs_{nullptr}; ///< PMF/force grids, nullptr = none
	Vector3 electric_field_{0.0f, 0.0f, 0.0f};				   ///< Uniform global E field
	int interpolation_scheme_{0};							   ///< 0=linear, 1=cubic

	DeviceBondedInteractions device_bonded_;
	bool bonded_device_data_prepared_{false};

	// Device-side pairwise nonbonded topology (type-pair -> table matrix)
	std::unique_ptr<DevicePairNonBondedInteractions> device_pair_nb_;
	bool pairwise_nb_device_data_prepared_{false};
};

/**
 * @brief Factory function to create patch decomposer
 * @param type Type of decomposer to create
 * @return Unique pointer to decomposer
 */
std::unique_ptr<PatchDecomposer> create_patch_decomposer(DecomposerType type);

} // namespace ARBD
