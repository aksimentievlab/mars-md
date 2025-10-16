#pragma once
#include "Interactions/Interactions.h"
#include "Objects/ParticleProperties.h"
#include "SimParam.h"
#include "System/PeriodicBox.h"
#include "Types/Types.h"

namespace ARBD {

class Patch {

  public:
	/**
	 * @brief Construct a patch with specified capacity and index
	 * @param patch_idx Unique identifier for this patch
	 * @param capacity Initial capacity for particle storage
	 */
	Patch(patch_t patch_idx = 0, idx_t capacity = 1024)
		: patch_idx_(patch_idx), capacity_(capacity), num_(0), particle_array_{capacity} {};

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
		return particle_array_;
	}
	ParticleSoA& get_particle_array() {
		return particle_array_;
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
			// Move last particle to this position
			if (index < num_ - 1) {
				// Copy data from last particle to this position
				particle_array_.id[index] = particle_array_.id[num_ - 1];
				particle_array_.type_id[index] = particle_array_.type_id[num_ - 1];
				particle_array_.position[index] = particle_array_.position[num_ - 1];
				particle_array_.momentum[index] = particle_array_.momentum[num_ - 1];
				particle_array_.force[index] = particle_array_.force[num_ - 1];
				particle_array_.orientation[index] = particle_array_.orientation[num_ - 1];
				particle_array_.is_dummy[index] = particle_array_.is_dummy[num_ - 1];
				particle_array_.has_orientation[index] = particle_array_.has_orientation[num_ - 1];
			}
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
			particle_array_aos_.push_back(particle);
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
		if (system_sim_box_) {
		}
		return false;
	}

  private:
	// Core patch properties
	patch_t patch_idx_;		 ///< Unique identifier for this patch
	idx_t capacity_;		 ///< Maximum number of particles this patch can hold
	idx_t num_;				 ///< Current number of particles in this patch
	idx_t num_replicas_{1};	 ///< Number of replicas of this patch
	short gpu_id_{0};		 ///< GPU ID for this patch
	int num_group_sites_{0}; ///< Number of group sites in this patch
	float ghost_thickness_{0.0f};

	// Spatial bounds
	Vector3 bounds_min_{0.0f, 0.0f, 0.0f}; ///< Minimum corner of patch
	Vector3 bounds_max_{0.0f, 0.0f, 0.0f}; ///< Maximum corner of patch

	// Particle data
	ParticleSoA particle_array_;				  ///< Array of particles in this patch
	std::vector<ParticleAoS> particle_array_aos_; ///< Array of particles in this patch
	std::vector<ParticleType> types_;			  ///< Types of particles in this patch

	// Interactions
	std::vector<NonbondedInteraction*> nonbonded_interactions_;
	std::vector<BondedInteraction*> bonded_interactions_;

	// System-wide simulation box
	const PeriodicBox* system_sim_box_{nullptr}; ///< Reference to system-wide simulation box (not owned)
};
}; // namespace ARBD
