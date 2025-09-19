#pragma once
// #include "Types/BaseGrid.h"
#include "Types/Types.h"
#include <string>

namespace ARBD {
class NonbondedInteraction;
class BondedInteraction;
/**
 * @brief Atom group definition for collective variables
 */
struct ColvarsGroup {
	int id;
	std::vector<int> particle_ids;
	std::string name;

	// Helper methods
	size_t size() const {
		return particle_ids.size();
	}
	bool empty() const {
		return particle_ids.empty();
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
	size_t num_local_particles;
	size_t num_ghost_particles;
	size_t capacity; // All buffers share the same capacity
};

struct ParticleAoS {
	int id;
	int type_id;
	Vector3 position;
	Vector3 momentum;
	Vector3 force;
	Vector3 orientation;
	float energy;
	bool is_dummy = false;
	bool has_orientation = false;
	std::vector<ColvarsGroup> colvars_groups;
	// colvars groups this particle belongs to. One particle can belong to multiple groups.
	ParticleAoS& operator=(const ParticleAoS& src) {
		id = src.id;
		type_id = src.type_id;
		position = src.position;
		momentum = src.momentum;
		force = src.force;
		orientation = src.orientation;
		energy = src.energy;
		is_dummy = src.is_dummy;
		has_orientation = src.has_orientation;
		return *this;
	}
};
using Particle = ParticleAoS;

class ParticleType {
  private:
	void clear();
	void copy(const ParticleType& src);

  public:
	void parse_type();
	std::string name;
	int id;
	int num; // number of particles of this type
	float mass;
	float charge;
	float radius; // vdw radius
	float eps;	  // vdw
	float diffusion;
	Vector3 transDamping;
	float mu; // for Nose-Hoover Langevin dynamics
	int numPartGridFiles;
	float* meanPmf;
	float* pmf_scale;

	// BaseGrid<float>** pmfGrid;
	// BaseGrid<float>* diffusionGrid; // uncommon
	// BaseGrid<Vector3>* forceGrid;

	ParticleType(const std::string& name) : name(name){};
	ParticleType(const ParticleType& src);
	ParticleType& operator=(const ParticleType& src);
	~ParticleType();
};

} // namespace ARBD
