#pragma once

#include <functional>
#include <memory> // std::make_unique
#include <vector> // std::vector

#include "ParticleType.h"
#include "System/SimSystem.h"
#include "Types/Types.h"

namespace ARBD {

template<typename Pos, typename Force>
class Patch {
	int num_replicas;

  public:
	Patch(SimParameters* sim);

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
};
// Storage class that should be
class Patch : public BasePatch {
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
		std::unique_ptr<VecArray> pos_force;
		std::unique_ptr<VecArray> momentum;
		std::unique_ptr<std::vector<size_t>> particle_types;
		std::unique_ptr<std::vector<size_t>> particle_order;

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

		// Replace with auto? Return number of particles sent?
		// template<typename T>
		// size_t send_particles_filtered( Proxy<Data>& destination, T filter ); //
		// = [](size_t idx, Patch::Data d)->bool { return true; } );
		size_t send_particles_filtered(
			Proxy<Data>& destination,
			std::function<bool(size_t, Data)> filter); // = [](size_t idx, Patch::Data
													   // d)->bool { return true; } );
	};

	// Metadata stored on host even if Data is on device

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

	Patch send_children(Resource location) const;

	// TODO deprecate copy_to_cuda
	Patch* copy_from_host(Patch* host_ptr = nullptr) const;
	static Patch copy_to_host(Patch* host_ptr, Patch* dest = nullptr);
	static void remove_from_host(Patch* host_ptr); // TODO: remove_from_host

	// TODO? emplace_point_particles
	void compute();

	// Communication
	size_t send_particles(Proxy<Patch>* destination); // Same as send_children?
	// void send_particles_filtered( Proxy<Patch> destination,
	// std::function<bool(size_t, Patch::Data)> = [](size_t idx, Patch::Data
	// d)->bool { return true; } );

	// Replace with auto? Return number of particles sent?
	// template<typename T>
	// size_t send_particles_filtered( Proxy<Patch>& destination, T filter );
	// // [](size_t idx, Patch::Data d)->bool { return true; } );
	size_t send_particles_filtered(Proxy<Patch>& destination,
								   std::function<bool(size_t, Data)> filter);
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

	// TODO: move computes to another object; this should simply be a dumb data
	// store std::vector<PatchProxy> neighbors and should use Event/Buffer system for streaming
	// Kernels
	// Potentially a CUDAGraph or SYCL graph ops.
	std::vector<std::unique_ptr<PatchOp>> local_computes;	 // Operations that will be performed on
															 // this patch each timestep
	std::vector<std::unique_ptr<PatchOp>> nonlocal_computes; // Operations that will be performed on
															 // this patch each timestep

	Metadata metadata; // Usually associated with proxy, but can use it here too

	size_t num_group_sites;
};
} // namespace ARBD
