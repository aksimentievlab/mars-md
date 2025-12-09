#pragma once
// #include "Types/BaseGrid.h"
#include "Objects/Grid.h"
#include "System/Reservoir.h"
#include "Types/BaseGrid.h"
#include "Types/Types.h"
#include <string>

namespace ARBD {

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

struct ParticleRead {
	int id;
	std::string type_name;
	Vector3 position;
	Vector3 momentum;
	Vector3 orientation;
	Vector3 force;
	float energy;
	bool has_orientation = false;
	int group_id = -1; // belongs to no one.
	// colvars groups this particle belongs to. One particle can belong to multiple groups.
	ParticleRead& operator=(const ParticleRead& src) {
		id = src.id;
		type_name = src.type_name;
		position = src.position;
		momentum = src.momentum;
		force = src.force;
		orientation = src.orientation;
		energy = src.energy;
		has_orientation = src.has_orientation;
		group_id = src.group_id;
		return *this;
	}
};

class ParticleType {
  public:
	std::string name;
	int id = -1; // Default to invalid, will be set after initialization
	int num = 0; // Count starts at 0

	// --- Physics Constants (Initialized to safe defaults) ---
	float mass = 1.0f; // Avoid divide-by-zero if missed
	float charge = 0.0f;
	float radius = 0.0;
	float eps = 0.0f;
	float diffusion = 0.0f;
	Vector3 transDamping = {0.0f, 0.0f, 0.0f};
	float mu = 0.0f;

	// --- PMF / SMD Control ---
	float meanPmf = 0.0f;
	float pmf_scale = 1.0f;
	float pmf_scale_slope = 0.0f;
	uint32_t pmf_smd_freq = 0;

	// --- Grid IDs ---
	int pmf_grid_id = -1;
	int diffusion_grid_id = -1;
	int3 force_grid_id = {-1, -1, -1}; // 3d grid

	std::unique_ptr<Reservoir> reservoir = nullptr;

	ParticleType(const std::string& name) : name(name) {}
	ParticleType(const ParticleType& src)
		: name(src.name), id(src.id), num(src.num), mass(src.mass), charge(src.charge),
		  radius(src.radius), eps(src.eps), diffusion(src.diffusion),
		  transDamping(src.transDamping), mu(src.mu), pmf_scale(src.pmf_scale),
		  pmf_scale_slope(src.pmf_scale_slope), pmf_smd_freq(src.pmf_smd_freq),
		  pmf_grid_id(src.pmf_grid_id), diffusion_grid_id(src.diffusion_grid_id),
		  force_grid_id(src.force_grid_id), reservoir(std::make_unique<Reservoir>(*src.reservoir)) {
	}
	ParticleType(const std::string& name,
				 float mass,
				 float charge,
				 float radius,
				 float eps,
				 float diffusion,
				 Vector3 transDamping,
				 float mu,
				 float pmf_scale,
				 float pmf_scale_slope,
				 float pmf_smd_freq,
				 int pmf_grid_id,
				 int diffusion_grid_id,
				 int3 force_grid_ids,
				 std::unique_ptr<Reservoir> reservoir)
		: name(name), mass(mass), charge(charge), radius(radius), eps(eps), diffusion(diffusion),
		  transDamping(transDamping), mu(mu), pmf_scale(pmf_scale),
		  pmf_scale_slope(pmf_scale_slope), pmf_smd_freq(pmf_smd_freq), pmf_grid_id(pmf_grid_id),
		  diffusion_grid_id(diffusion_grid_id), force_grid_id(force_grid_ids),
		  reservoir(std::make_unique<Reservoir>(*reservoir)) {}
};
// ============================================================================
// 4. HOST STORAGE (For IO / Init)
// ============================================================================
struct HostParticleData {
	std::vector<int> id;
	std::vector<int> type_id;
	std::vector<Vector3> pos;
	std::vector<Vector3> mom;
	std::vector<Vector3> force;
	std::vector<float> energy;
	std::vector<Vector3> orient;
	std::vector<uint32_t> flags;

	void resize(size_t n) {
		id.resize(n);
		type_id.resize(n);
		pos.resize(n);
		mom.resize(n);
		force.resize(n);
		energy.resize(n);
		orient.resize(n);
		flags.resize(n);
	}

	size_t size() const {
		return id.size();
	}
};

} // namespace ARBD
