#pragma once
// #include "Types/BaseGrid.h"
#include "Objects/Grid.h"
#include "System/Reservoir.h"
#include "Types/BaseGrid.h"
#include "Types/Types.h"
#include <string>

namespace ARBD {
enum ParticleFlags : uint32_t {
	FLAG_NONE = 0,
	FLAG_DUMMY = 1 << 0,
	FLAG_ACTIVE = 1 << 1,
};
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
	Vector3 diffusion = {0.0f, 0.0f, 0.0f};
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

	std::string pmf_grid_name = "";
	std::string diffusion_grid_name = "";
	std::array<std::string, 3> force_grid_names = {"", "", ""};

	std::unique_ptr<Reservoir> reservoir = nullptr;

	ParticleType(const std::string& name) : name(name) {}
	ParticleType(const ParticleType& src)
		: name(src.name), id(src.id), num(src.num), mass(src.mass), charge(src.charge),
		  radius(src.radius), eps(src.eps), diffusion(src.diffusion), mu(src.mu),
		  pmf_scale(src.pmf_scale), pmf_scale_slope(src.pmf_scale_slope),
		  pmf_smd_freq(src.pmf_smd_freq), pmf_grid_id(src.pmf_grid_id),
		  diffusion_grid_id(src.diffusion_grid_id), force_grid_id(src.force_grid_id),
		  reservoir(src.reservoir ? std::make_unique<Reservoir>(*src.reservoir) : nullptr) {}
	ParticleType(const std::string& name,
				 float mass,
				 float charge,
				 float radius,
				 float eps,
				 float diffusion,
				 float mu,
				 float pmf_scale,
				 float pmf_scale_slope,
				 float pmf_smd_freq,
				 int pmf_grid_id,
				 int diffusion_grid_id,
				 int3 force_grid_ids,
				 std::unique_ptr<Reservoir> reservoir)
		: name(name), mass(mass), charge(charge), radius(radius), eps(eps), diffusion(diffusion),
		  mu(mu), pmf_scale(pmf_scale), pmf_scale_slope(pmf_scale_slope),
		  pmf_smd_freq(pmf_smd_freq), pmf_grid_id(pmf_grid_id),
		  diffusion_grid_id(diffusion_grid_id), force_grid_id(force_grid_ids),
		  reservoir(reservoir ? std::make_unique<Reservoir>(*reservoir) : nullptr) {}
};
// ============================================================================
// 4. HOST STORAGE (For IO / Init)
// ============================================================================
struct ParticleRead {
	int id;
	std::string type_name;
	Vector3 position;
	Vector3 momentum;
	Vector3 orientation{1.0f, 0.0f, 0.0f};
	Vector3 force;
	float energy;
	int group_id = -1; // belongs to no one.
	int colvars_group_id = -1;
	int attached_rigid_body_id = -1;
	// colvars groups this particle belongs to. One particle can belong to multiple groups.
	ParticleRead& operator=(const ParticleRead& src) {
		id = src.id;
		type_name = src.type_name;
		position = src.position;
		momentum = src.momentum;
		force = src.force;
		orientation = src.orientation;
		energy = src.energy;
		group_id = src.group_id;
		return *this;
	}
};

struct HostParticleData {
	std::vector<int> global_id;
	std::vector<int> type_id;
	std::vector<Vector3> pos;
	std::vector<Vector3> mom;
	std::vector<Vector3> force;
	std::vector<float> energy;
	std::vector<Vector3> orient;
	std::vector<uint32_t> flags;

	std::vector<bool> is_active;
	std::vector<bool> is_dummy;
	std::vector<int> colvars_group_id;
	std::vector<int> group_id;
	std::vector<int> attached_rigid_body_id;

	size_t size() const {
		return global_id.size();
	}

	void pack_flags() {
		flags.resize(size());
		for (size_t i = 0; i < size(); ++i) {
			flags[i] = ParticleFlags::FLAG_NONE;
			if (is_dummy[i])
				flags[i] |= ParticleFlags::FLAG_DUMMY;
			if (is_active[i])
				flags[i] |= ParticleFlags::FLAG_ACTIVE;
		}
	}
	void unpack_flags() {
		is_dummy.resize(flags.size());
		for (size_t i = 0; i < flags.size(); ++i) {
			is_dummy[i] = (flags[i] & ParticleFlags::FLAG_DUMMY) != 0;
			is_active[i] = (flags[i] & ParticleFlags::FLAG_ACTIVE) != 0;
		}
	}
	void resize(size_t n) {
		global_id.resize(n);
		type_id.resize(n);
		pos.resize(n);
		mom.resize(n);
		force.resize(n);
		energy.resize(n);
		orient.resize(n);
		flags.resize(n);
		group_id.resize(n);
		is_active.resize(n);
		is_dummy.resize(n);
		colvars_group_id.resize(n);
		attached_rigid_body_id.resize(n);
	}

	void clear() {
		global_id.clear();
		type_id.clear();
		pos.clear();
		mom.clear();
		force.clear();
		energy.clear();
		orient.clear();
		// flags.clear();
		group_id.clear();
		is_active.clear();
		is_dummy.clear();
		colvars_group_id.clear();
		attached_rigid_body_id.clear();
	}
	void reserve(size_t n) {
		global_id.reserve(n);
		type_id.reserve(n);
		pos.reserve(n);
		mom.reserve(n);
		force.reserve(n);
		energy.reserve(n);
		orient.reserve(n);
	}

	void push_back(int global_id_,
				   int type_id_,
				   const Vector3& pos_,
				   const Vector3& mom_,
				   const Vector3& force_,
				   float energy_,
				   const Vector3& orient_) {
		global_id.push_back(global_id_);
		type_id.push_back(type_id_);
		pos.push_back(pos_);
		mom.push_back(mom_);
		force.push_back(force_);
		energy.push_back(energy_);
		orient.push_back(orient_);
		flags.push_back(ParticleFlags::FLAG_NONE);
		is_active.push_back(true);
		is_dummy.push_back(false);
		colvars_group_id.push_back(-1);
		group_id.push_back(-1);
		attached_rigid_body_id.push_back(-1);
	}

	void push_back(const HostParticleData& particle, size_t idx) {
		global_id.push_back(particle.global_id[idx]);
		type_id.push_back(particle.type_id[idx]);
		pos.push_back(particle.pos[idx]);
		mom.push_back(particle.mom[idx]);
		force.push_back(particle.force[idx]);
		energy.push_back(particle.energy[idx]);
		orient.push_back(particle.orient[idx]);
		flags.push_back(particle.flags[idx]);
		is_active.push_back(particle.is_active[idx]);
		is_dummy.push_back(particle.is_dummy[idx]);
		colvars_group_id.push_back(particle.colvars_group_id[idx]);
		group_id.push_back(particle.group_id[idx]);
		attached_rigid_body_id.push_back(particle.attached_rigid_body_id[idx]);
	}

	void push_back(const ParticleRead& particle,
				   std::unordered_map<std::string, int>& particle_type_name_to_id) {
		global_id.push_back(particle.id);
		type_id.push_back(particle_type_name_to_id.at(particle.type_name));
		pos.push_back(particle.position);
		mom.push_back(particle.momentum);
		force.push_back(particle.force);
		energy.push_back(particle.energy);
		orient.push_back(particle.orientation);
		flags.push_back(ParticleFlags::FLAG_NONE);
		is_active.push_back(true);
		is_dummy.push_back(false);
		colvars_group_id.push_back(particle.colvars_group_id);
		group_id.push_back(particle.group_id);
		attached_rigid_body_id.push_back(particle.attached_rigid_body_id);
	}

	void remove_swap(size_t idx) {
		if (idx >= size())
			return;

		const size_t last = size() - 1;
		if (idx < last) {
			// Swap with last element
			std::swap(global_id[idx], global_id[last]);
			std::swap(type_id[idx], type_id[last]);
			std::swap(pos[idx], pos[last]);
			std::swap(mom[idx], mom[last]);
			std::swap(force[idx], force[last]);
			std::swap(energy[idx], energy[last]);
			std::swap(orient[idx], orient[last]);

			// Swap flags and boolean vectors directly (O(1) instead of O(n))
			if (flags.size() > idx && flags.size() > last) {
				std::swap(flags[idx], flags[last]);
			}
			if (is_active.size() > idx && is_active.size() > last) {
				bool temp = is_active[idx];
				is_active[idx] = is_active[last];
				is_active[last] = temp;
			}
			if (is_dummy.size() > idx && is_dummy.size() > last) {
				bool temp = is_dummy[idx];
				is_dummy[idx] = is_dummy[last];
				is_dummy[last] = temp;
			}

			if (group_id.size() > idx && group_id.size() > last) {
				std::swap(group_id[idx], group_id[last]);
			}
			if (colvars_group_id.size() > idx && colvars_group_id.size() > last) {
				std::swap(colvars_group_id[idx], colvars_group_id[last]);
			}
			if (attached_rigid_body_id.size() > idx && attached_rigid_body_id.size() > last) {
				std::swap(attached_rigid_body_id[idx], attached_rigid_body_id[last]);
			}
		}

		// Pop last element (same as before)
		global_id.pop_back();
		type_id.pop_back();
		pos.pop_back();
		mom.pop_back();
		force.pop_back();
		energy.pop_back();
		orient.pop_back();

		if (!flags.empty())
			flags.pop_back();
		if (!is_active.empty())
			is_active.pop_back();
		if (!is_dummy.empty())
			is_dummy.pop_back();
		if (!group_id.empty())
			group_id.pop_back();
		if (!colvars_group_id.empty())
			colvars_group_id.pop_back();
		if (!attached_rigid_body_id.empty())
			attached_rigid_body_id.pop_back();
	}
};

} // namespace ARBD
