#pragma once
#include "Interactions/BondedInteraction.h"
#include "Interactions/Interactions.h"
#include "Interactions/NonBondedInteraction.h"
#include "Objects/DeviceParticle.h"
#include "Objects/ParticleProperties.h"
#include "SimParam.h"
#include "System/PeriodicBox.h"
#include "System/ZOrderSort.h"
#include "Types/Types.h"
#include <utility>

namespace ARBD {

class Patch {

  public:
	/**
	 * @brief Construct a patch with specified capacity and index
	 * @param patch_idx Unique identifier for this patch
	 * @param capacity Initial capacity for particle storage
	 */
	Patch(patch_t patch_idx = 0, idx_t capacity = 1024)
		: patch_idx_(patch_idx), capacity_(capacity), num_(0){};

	/**
	 * @brief Move constructor
	 */
	Patch(Patch&& other) noexcept
		: patch_idx_(other.patch_idx_), capacity_(other.capacity_), num_(other.num_),
		  num_replicas_(other.num_replicas_), gpu_id_(other.gpu_id_),
		  num_group_sites_(other.num_group_sites_), bounds_min_(other.bounds_min_),
		  bounds_max_(other.bounds_max_), particle_ids(std::move(other.particle_ids)),
		  particle_type_ids(std::move(other.particle_type_ids)),
		  particle_positions(std::move(other.particle_positions)),
		  particle_momenta(std::move(other.particle_momenta)),
		  particle_forces(std::move(other.particle_forces)),
		  particle_energies(std::move(other.particle_energies)),
		  particle_orientations(std::move(other.particle_orientations)),
		  particle_is_dummy(std::move(other.particle_is_dummy)),
		  particle_has_orientation(std::move(other.particle_has_orientation)),
		  particle_is_ghost(std::move(other.particle_is_ghost)),
		  d_cell_starts(std::move(other.d_cell_starts)), d_cell_ends(std::move(other.d_cell_ends)),
		  d_cell_neighbors(std::move(other.d_cell_neighbors)),
		  pmf_grids_(std::move(other.pmf_grids_)), density_grids_(std::move(other.density_grids_)),
		  force_grids_(std::move(other.force_grids_)), system_sim_box_(other.system_sim_box_) {
		// Reset moved object
		other.patch_idx_ = 0;
		other.capacity_ = 0;
		other.num_ = 0;
		other.system_sim_box_ = nullptr;
	}

	/**
	 * @brief Move assignment operator
	 */
	Patch& operator=(Patch&& other) noexcept {
		if (this != &other) {
			patch_idx_ = other.patch_idx_;
			capacity_ = other.capacity_;
			num_ = other.num_;
			num_replicas_ = other.num_replicas_;
			gpu_id_ = other.gpu_id_;
			num_group_sites_ = other.num_group_sites_;
			particle_ids = std::move(other.particle_ids);
			particle_type_ids = std::move(other.particle_type_ids);
			particle_positions = std::move(other.particle_positions);
			particle_momenta = std::move(other.particle_momenta);
			particle_forces = std::move(other.particle_forces);
			particle_energies = std::move(other.particle_energies);
			particle_orientations = std::move(other.particle_orientations);
			particle_is_dummy = std::move(other.particle_is_dummy);
			particle_has_orientation = std::move(other.particle_has_orientation);
			particle_is_ghost = std::move(other.particle_is_ghost);
			nonbonded_interactions_ = std::move(other.nonbonded_interactions_);
			bonded_interactions_ = std::move(other.bonded_interactions_);
			system_sim_box_ = other.system_sim_box_;

			// Reset moved object
			other.patch_idx_ = 0;
			other.capacity_ = 0;
			other.num_ = 0;
			other.system_sim_box_ = nullptr;
		}
		return *this;
	}

	~Patch() = default;

	/**
	 * @brief Set reference to system-wide simulation box (shared across all patches)
	 * @param system_box Pointer to system-wide simulation box
	 *
	 * Note: The patch does not own this pointer - it's managed by SimSystem/PatchManager
	 */
	void set_system_sim_box(const PeriodicBox* system_box) {
		system_sim_box_ = system_box;
	}

	/**
	 * @brief Get system-wide simulation box for boundary calculations
	 * @return Pointer to system simulation box (device-safe)
	 */
	const PeriodicBox* get_sim_box() const {
		return system_sim_box_;
	}

	/**
	 * @brief Get patch index
	 */
	patch_t get_patch_idx() const {
		return patch_idx_;
	}

	/**
	 * @brief Get current number of particles
	 */
	idx_t get_num() const {
		return num_;
	}

	/**
	 * @brief Get patch capacity
	 */
	idx_t get_capacity() const {
		return capacity_;
	}

	/**
	 * @brief Get particle types
	 */
	const std::vector<ParticleType>& get_types() const {
		return types_;
	}
	std::vector<ParticleType>& get_types() {
		return types_;
	}

	/**
	 * @brief Get nonbonded interactions
	 */
	const std::vector<NonBondedInteraction*>& get_nonbonded_interactions() const {
		return nonbonded_interactions_;
	}
	std::vector<NonBondedInteraction*>& get_nonbonded_interactions() {
		return nonbonded_interactions_;
	}

	/**
	 * @brief Get bonded interactions
	 */
	const std::vector<BondedInteraction*>& get_bonded_interactions() const {
		return bonded_interactions_;
	}
	std::vector<BondedInteraction*>& get_bonded_interactions() {
		return bonded_interactions_;
	}

	/**
	 * @brief Set the number of particles in this patch
	 * @param num New particle count
	 */
	void set_num(idx_t num) {
		if (num <= capacity_) {
			num_ = num;
		}
	}

	/**
	 * @brief Add a particle to this patch
	 * @return Index of the added particle, or -1 if patch is full
	 */
	idx_t add_particle() {
		if (num_ < capacity_) {
			return num_++;
		}
		return -1; // Patch is full
	}

	/**
	 * @brief Remove a particle from this patch
	 * @param index Index of particle to remove
	 * @return True if successful, false if index is invalid
	 */
	bool remove_particle(idx_t index) {
		if (index < num_) {

			num_--;
			return true;
		}
		return false;
	}

	/**
	 * @brief Check if patch has space for more particles
	 */
	bool has_space() const {
		return num_ < capacity_;
	}

	DEVICE void push_particle(const ParticleRead& particle) {
		if (has_space()) {
			particle_ids.push_back(particle.id);
			particle_type_ids.push_back(particle.type_id);
			particle_positions.push_back(particle.position);
			particle_momenta.push_back(particle.momentum);
			particle_forces.push_back(particle.force);
			particle_energies.push_back(particle.energy);
			particle_orientations.push_back(particle.orientation);
			num_++;
		}
	}

	/**
	 * @brief Get remaining capacity
	 */
	idx_t get_remaining_capacity() const {
		return capacity_ - num_;
	}

	/**
	 * @brief Set spatial bounds for this patch
	 * @param min Minimum corner of patch
	 * @param max Maximum corner of patch
	 */
	void set_bounds(const Vector3& min, const Vector3& max) {
		bounds_min_ = min;
		bounds_max_ = max;
	}

	/**
	 * @brief Get patch minimum bounds
	 */
	const Vector3& get_bounds_min() const {
		return bounds_min_;
	}

	/**
	 * @brief Get patch maximum bounds
	 */
	const Vector3& get_bounds_max() const {
		return bounds_max_;
	}

	/**
	 * @brief Get ghost region thickness from system simulation box
	 * @return Ghost region thickness
	 */
	float get_ghost_thickness() const {
		return ghost_thickness_;
	}
	void set_ghost_thickness(float ghost_thickness) {
		ghost_thickness_ = ghost_thickness;
	}

	/**
	 * @brief Check if a position is in the ghost region of this patch
	 * @param pos Position to check
	 * @return True if position is within ghost region
	 */
	bool is_in_ghost_region(const Vector3& pos) const {
		Vector3 r = pos - bounds_min_;
		if (pos.x > bounds_max_.x || pos.y > bounds_max_.y || pos.z > bounds_max_.z) {
			return true;
		} else if (r.x < 0.0f | r.y < 0.0f || r.z < 0.0f) {
			return true;
		}
		return false;
	}

	/**
	 * @brief Get resource for this patch
	 * @return Resource for this patch
	 */
	const Resource& get_resource() const {
		return resource_;
	}

	void set_resource(const Resource& resource) {
		resource_ = resource;
	}

	/**
	 * @brief Check if a particle position needs to migrate to a neighbor patch
	 * @param position Particle position to check
	 * @return Direction index if migration needed, -1 otherwise
	 *         Direction: 0=x-, 1=x+, 2=y-, 3=y+, 4=z-, 5=z+
	 */
	int check_particle_migration(const Vector3& position) const {
		if (position.x < bounds_min_.x)
			return 0; // x-
		if (position.x >= bounds_max_.x)
			return 1; // x+
		if (position.y < bounds_min_.y)
			return 2; // y-
		if (position.y >= bounds_max_.y)
			return 3; // y+
		if (position.z < bounds_min_.z)
			return 4; // z-
		if (position.z >= bounds_max_.z)
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
		switch (direction) {
		case 0: // x-
			return position.x >= bounds_min_.x && position.x < bounds_min_.x + halo_width;
		case 1: // x+
			return position.x >= bounds_max_.x - halo_width && position.x < bounds_max_.x;
		case 2: // y-
			return position.y >= bounds_min_.y && position.y < bounds_min_.y + halo_width;
		case 3: // y+
			return position.y >= bounds_max_.y - halo_width && position.y < bounds_max_.y;
		case 4: // z-
			return position.z >= bounds_min_.z && position.z < bounds_min_.z + halo_width;
		case 5: // z+
			return position.z >= bounds_max_.z - halo_width && position.z < bounds_max_.z;
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
		Vector3 halo_min = bounds_min_;
		Vector3 halo_max = bounds_max_;

		switch (direction) {
		case 0: // x-
			halo_max.x = bounds_min_.x;
			halo_min.x = bounds_min_.x - halo_width;
			break;
		case 1: // x+
			halo_min.x = bounds_max_.x;
			halo_max.x = bounds_max_.x + halo_width;
			break;
		case 2: // y-
			halo_max.y = bounds_min_.y;
			halo_min.y = bounds_min_.y - halo_width;
			break;
		case 3: // y+
			halo_min.y = bounds_max_.y;
			halo_max.y = bounds_max_.y + halo_width;
			break;
		case 4: // z-
			halo_max.z = bounds_min_.z;
			halo_min.z = bounds_min_.z - halo_width;
			break;
		case 5: // z+
			halo_min.z = bounds_max_.z;
			halo_max.z = bounds_max_.z + halo_width;
			break;
		}

		return {halo_min, halo_max};
	}

	void sort_particles(const DeviceBuffer<Vector3>& positions,
						size_t num_particles,
						const Vector3& box_min,
						const Vector3& box_max) {
		zorder_sorters_.sort_particles(positions, num_particles, box_min, box_max);
	}

	/**
	 * @brief Pack particles from this patch's boundary region for halo exchange
	 * This method also marks the packed particles as inactive (is_dummy = true)
	 * @param packed_data Output buffer for packed particle data (position + momentum)
	 * @param direction Direction index (0=x-, 1=x+, 2=y-, 3=y+, 4=z-, 5=z+)
	 * @param cutoff Cutoff radius for halo region calculation
	 * @return Number of particles packed and marked as inactive
	 */
	idx_t
	pack_boundary_particles(DeviceBuffer<float>& packed_data, int direction, const Length& cutoff);

	/**
	 * @brief Unpack received halo particles into this patch's halo storage
	 * @param packed_data Input buffer containing packed particle data (position + momentum)
	 * @param received_count Number of particles to unpack
	 */
	void unpack_halo_particles(DeviceBuffer<float>& packed_data, idx_t received_count);

  private:
	// Core patch properties
	short gpu_id_{-1};		///< GPU ID for this patch
	Resource resource_;		///< Resource for this patch
	patch_t patch_idx_;		///< Unique identifier for this patch
	idx_t capacity_;		///< Maximum number of particles this patch can hold
	idx_t num_replicas_{1}; ///< Number of replicas of this patch
	idx_t num_; ///< Current number of particles in this patch, num_*num_replicas_<=capacity_
	idx_t num_group_sites_{0}; ///< Number of group sites in this patch
	float ghost_thickness_{0.0f};
	ZOrderSort zorder_sorters_{resource_, capacity_, ZOrderOptimizationMode::Pairlist};
	idx_t num_ghost_particles_{0};

  protected:
	idx_t capacity;
	Vector3 lower_bound, upper_bound;

	static idx_t global_patch_idx; ///< Unique ID across ranks
	idx_t patch_idx;			   ///< Unique ID for this patch

	// Spatial bounds
	Vector3 bounds_min_{0.0f, 0.0f, 0.0f}; ///< Minimum corner of patch
	Vector3 bounds_max_{0.0f, 0.0f, 0.0f}; ///< Maximum corner of patch
	// Particle data
	std::vector<int> particle_ids;
	std::vector<int> particle_type_ids;
	std::vector<Vector3> particle_positions;
	std::vector<Vector3> particle_momenta;
	std::vector<Vector3> particle_forces;
	std::vector<float> particle_energies;
	std::vector<Vector3> particle_orientations;
	std::vector<bool> particle_is_dummy;
	std::vector<bool> particle_has_orientation;
	std::vector<bool> particle_is_ghost;

	DeviceBuffer<int> d_cell_starts;
	DeviceBuffer<int> d_cell_ends;
	DeviceBuffer<int> d_cell_neighbors;

	std::vector<BaseGrid<float>> pmf_grids_;
	std::vector<BaseGrid<float>> density_grids_;
	std::vector<BaseGrid<float>> force_grids_;
	// Interactions
	std::vector<NonBondedInteraction*> nonbonded_interactions_;
	std::vector<BondedInteraction*> bonded_interactions_;

	// System-wide simulation box
	const PeriodicBox* system_sim_box_{
		nullptr}; ///< Reference to system-wide simulation box (not owned)
};
}; // namespace ARBD
