#pragma once
#include "Interactions/Interactions.h"
#include "ParticleProperties.h"
#include "Types/BaseGrid.h"
#include "Types/Types.h"

namespace ARBD {

class Patch {

  public:
	Patch() = default;
	Patch(const Patch& other) = default;
	Patch(patch_t patch_idx = 0, idx_t capacity = 2048)
		: capacity(capacity), patch_idx(patch_idx){};

	// Move assignment operator
	Patch& operator=(Patch&& other) {
		num = other.num;
		capacity = other.capacity;
		patch_idx = other.patch_idx;
		return *this;
	}

	~Patch() {}

	/**
	 * @brief Metadata for this patch.
	 */
  public:
	struct Metadata {
		idx_t num;		// number of objects in this patch
		idx_t capacity; // Capacity of the patch, will be larger than num
		Vector3 min;	// Minimum position of the patch
		Vector3 max;	// Maximum position of the patch
	} metadata;

	idx_t capacity;
	idx_t num{0};		   // number of particles in this patch
	idx_t num_replicas{1}; // number of replicas of this patch
	short thread_id;
	short gpu_id;
	idx_t patch_idx;			// This is now set by the kernel.
	int num_group_sites{0};		// number of group sites in this patch
	ParticleSoA particle_array; // array of particles in this patch

  private:
	Vector3 minimum; // minimum position of the patch
	Vector3 maximum;
	patch_t patch_id = 0;

	std::vector<ParticleType> types; // types of particles in this patch

	// Particle data
	size_t num_particles;

	std::vector<NonbondedInteraction*> nonbonded_interactions_;
	std::vector<BondedInteraction*> bonded_interactions_;
	std::vector<BaseGrid<float>> bonded_grids_;
	std::vector<BaseGrid<float>> nonbonded_grids_;
};
}; // namespace ARBD
