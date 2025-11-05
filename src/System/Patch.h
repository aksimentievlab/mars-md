#pragma once
#include "Objects/DeviceParticle.h"
#include "Objects/ParticleProperties.h"
#include "SimParam.h"
#include "System/PeriodicBox.h"
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
		  bounds_max_(other.bounds_max_), particle_array_(std::move(other.particle_array_)),
		  types_(std::move(other.types_)),
		  nonbonded_interactions_(std::move(other.nonbonded_interactions_)),
		  bonded_interactions_(std::move(other.bonded_interactions_)),
		  system_sim_box_(other.system_sim_box_) {
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
			particle_array_ = std::move(other.particle_array_);
			types_ = std::move(other.types_);
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
	 * @brief Get particle array
	 */
	const ParticleSoA& get_particle_array() const {
		return device_particle_array_;
	}
	ParticleSoA& get_particle_array() {
		return device_particle_array_;
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
	const std::vector<NonbondedInteraction*>& get_nonbonded_interactions() const {
		return nonbonded_interactions_;
	}
	std::vector<NonbondedInteraction*>& get_nonbonded_interactions() {
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

	void push_particle(const ParticleAoS& particle) {
		if (has_space()) {
			particle_array_.push_back(particle);
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

  private:
	// Core patch properties
	short gpu_id_{-1};		///< GPU ID for this patch
	Resource resource_;		///< Resource for this patch
	patch_t patch_idx_;		///< Unique identifier for this patch
	idx_t capacity_;		///< Maximum number of particles this patch can hold
	idx_t num_replicas_{1}; ///< Number of replicas of this patch
	idx_t num_; ///< Current number of particles in this patch, num_*num_replicas_<=capacity_
	int num_group_sites_{0}; ///< Number of group sites in this patch
	float ghost_thickness_{0.0f};

  protected:
	idx_t capacity;
	idx_t num;
	Vector3 lower_bound, upper_bound;

	static idx_t global_patch_idx; ///< Unique ID across ranks
	idx_t patch_idx;			   ///< Unique ID for this patch

	// Spatial bounds
	Vector3 bounds_min_{0.0f, 0.0f, 0.0f}; ///< Minimum corner of patch
	Vector3 bounds_max_{0.0f, 0.0f, 0.0f}; ///< Maximum corner of patch

	// Particle data
	ParticleSoA device_particle_array_;		  ///< Array of particles in this patch
	std::vector<ParticleAoS> particle_array_; ///< Array of particles in this patch
	std::vector<ParticleType> types_;		  ///< Types of particles in this patch

	std::vector<BaseGrid<float>> grids_;
	// Interactions
	std::vector<NonbondedInteraction*> nonbonded_interactions_;
	std::vector<BondedInteraction*> bonded_interactions_;

	// System-wide simulation box
	const PeriodicBox* system_sim_box_{
		nullptr}; ///< Reference to system-wide simulation box (not owned)
};
}; // namespace ARBD
