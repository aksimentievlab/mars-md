#pragma once
#include "../Patch/BasePatch.h"
#include "Backend/BackendTypes.h"
#include "Backend/Buffer.h"
#include "ParticleType.h"
#include "SimSystem.h"
#include "Types/Types.h"

namespace ARBD {
struct ParticleAoS {
	int id;
	int type_id;
	Vector3 position;
	Vector3 momentum;
	Vector3 force;
	Vector3 orientation;
	bool is_dummy = false;
	bool has_orientation = false;
	int colvars_group_id = -1;
	ParticleAoS& operator=(const ParticleAoS& src) {
		id = src.id;
		type_id = src.type_id;
		position = src.position;
		momentum = src.momentum;
		force = src.force;
		orientation = src.orientation;
		is_dummy = src.is_dummy;
		has_orientation = src.has_orientation;
		return *this;
	}
};

struct ParticleSoA { // Stored on GPU only
	Array<Vector3> id;
	Array<Vector3> type_id;
	Array<Vector3> position;
	Array<Vector3> momentum;
	Array<Vector3> force;
	Array<Vector3> orientation;
	Array<bool> is_dummy;
	Array<bool> has_orientation;
	Array<simd_int> colvars_group_id;
	size_t num_local_particles;
	size_t num_ghost_particles;
	size_t capacity; // All buffers share the same capacity
};
class NonbondedInteraction;
class BondedInteraction;

// Assign ParticleAoS and ParticleSoA
class ParticlePatch : public BasePatch {
	int num_replicas; // Number of replicas of this patch
	std::vector<NonbondedInteraction*> nonbonded_interactions_;
	std::vector<BondedInteraction*> bonded_interactions_;
	std::vector<BaseGrid<float>> bonded_grids_;
	std::vector<BaseGrid<float>> nonbonded_grids_;

  public:
	Patch(SimParameters* sim);

	colvars_group_id = src.colvars_group_id;
	return *this;
	ParticlePatch(const ParticlePatch& other) : BasePatch(other) {
		num_replicas = other.num_replicas;
	}
	ParticlePatch(ParticlePatch&& other) : BasePatch(std::move(other)) {
		num_replicas = std::move(other.num_replicas);
	}
	void compute();

  private:
	SimParameters* sim;

	Pos minimum;
	Pos maximum;

	std::vector<ParticleType> types;
	std::vector<ParticleType> types_d;

	// Particle data
	size_t num_particles;

	idx_t* global_idx; // global index of particle
	size_t* type_ids;
	Pos* pos;
	Force* force;

	size_t* type_ids_d;
	Pos* pos_d;
	Force* force_d;

  public:
	Patch() : BasePatch(), metadata() {
		LOGINFO("Creating Patch");
		initialize();
		LOGINFO("Done Creating Patch");
	}
	Patch(size_t capacity) : BasePatch(capacity) {
		initialize();
	}

	// Particle data arrays pointing to either CPU or GPU memory
	struct Data {
		Data(const size_t capacity = 0) {
			if (capacity == 0) {
				pos_force = nullptr;
				momentum = nullptr;
				particle_types = nullptr;
				particle_order = nullptr;
			} else {
				pos_force = std::make_unique<VecArray>(capacity);
				momentum = std::make_unique<VecArray>(capacity);
				particle_types = std::make_unique<std::vector<size_t>>(capacity);
				particle_order = std::make_unique<std::vector<size_t>>(capacity);
			}
		}
		std::unique_ptr<Array<Vector3>> pos_force;
		std::unique_ptr<Array<Vector3>> momentum;
		std::unique_ptr<Array<size_t>> particle_types;
		std::unique_ptr<Array<size_t>> particle_order;

		HOST DEVICE inline Vector3& get_pos(size_t i) {
			return (*pos_force)[i * 2];
		};
		HOST DEVICE inline Vector3& get_force(size_t i) {
			return (*pos_force)[i * 2 + 1];
		};
		HOST DEVICE inline Vector3& get_momentum(size_t i) {
			return (*momentum)[i];
		};
		HOST DEVICE inline size_t& get_type(size_t i) {
			return (*particle_types)[i];
		};
		HOST DEVICE inline size_t& get_order(size_t i) {
			return (*particle_order)[i];
		};

		// void deleteParticles(IndexList& p);
		// void addParticles(size_t n, size_t typ);
		// template<class T>
		// void add_compute(std::unique_ptr<T>&& p) {
		// 	std::unique_ptr<BasePatchOp> base_p =
		// static_cast<std::unique_ptr<BasePatchOp>>(p);
		// 	local_computes.emplace_back(p);
		// };

		// void add_compute(std::unique_ptr<BasePatchOp>&& p) {
		// 	local_computes.emplace_back(std::move(p));
		// };

		void add_point_particles(size_t num_added);
		void add_point_particles(size_t num_added, Vector3* positions, Vector3* momenta = nullptr);

		// TODO? emplace_point_particles
		void compute();

		// Communication
		// size_t send_particles(Proxy<Patch>* destination); // Same as send_children?
		// void send_particles_filtered( Proxy<Patch> destination,
		// std::function<bool(size_t, Patch::Data)> = [](size_t idx, Patch::Data
		// d)->bool { return true; } );

		// Replace with auto? Return number of particles sent?
		// template<typename T>
		// size_t send_particles_filtered( Proxy<Patch>& destination, T filter );
		// // [](size_t idx, Patch::Data d)->bool { return true; } );
		// size_t send_particles_filtered(Proxy<Patch>& destination,
		//							   std::function<bool(size_t, Data)> filter);
		// [](size_t idx, Patch::Data d)->bool { return true; } );

		void clear() {
			LOGWARN("Patch::clear() was called but is not implemented");
		}

		size_t test() {
			LOGWARN("Patch::test() was called but is not implemented");
			return 1;
		}

	  private:
		void initialize();

		void randomize_positions(size_t start = 0, size_t num = -1);
		Metadata metadata; // Usually associated with proxy, but can use it here too

		size_t num_group_sites;
	};
} // namespace ARBD
} // namespace ARBD
