#pragma once
#include "Backend/Buffer.h"
#include "ParticleProperties.h"
#include "Types/Types.h"

namespace ARBD {
// ============================================================================
// 1. FLAGS (Replacing bool vectors)
// ============================================================================
enum ParticleFlags : uint32_t {
	FLAG_NONE = 0,
	FLAG_DUMMY = 1 << 0,
	FLAG_HAS_ORIENTATION = 1 << 1
	// Add more as needed
};

// ============================================================================
// 2. VIEW (Lightweight pointer wrapper for Kernels)
// ============================================================================
struct ParticleView {
	int* id;
	int* type_id;
	Vector3* pos;
	Vector3* mom;
	Vector3* ForceEnergy;
	Vector3* orient;
	uint32_t* flags; // Combined flags

	idx_t capacity;
	idx_t num_particles; // Pass current count too!
};

struct ConstParticleView {
	const int* id;
	const int* type_id;
	const Vector3* pos;
	const Vector3* mom;
	const Vector3* ForceEnergy;
	const Vector3* orient;
	const uint32_t* flags;

	idx_t capacity;
	idx_t num_particles;
};

// ============================================================================
// 3. DEVICE STORAGE (Owning the buffers)
// ============================================================================
class DeviceParticle {
  public:
	DeviceParticle(idx_t capacity, const Resource& resource)
		: capacity_(capacity), count_(0), resource_(resource), id_(capacity, resource),
		  type_id_(capacity, resource), pos_(capacity, resource), mom_(capacity, resource),
		  ForceEnergy_(capacity, resource), orient_(capacity, resource),
		  flags_(capacity, resource) {}

	// Get View for Kernels
	ParticleView view() {
		return {id_.data(),
				type_id_.data(),
				pos_.data(),
				mom_.data(),
				ForceEnergy_.data(),
				orient_.data(),
				flags_.data(),
				capacity_,
				count_};
	}

	ConstParticleView view() const {
		return {id_.data(),
				type_id_.data(),
				pos_.data(),
				mom_.data(),
				ForceEnergy_.data(),
				orient_.data(),
				flags_.data(),
				capacity_,
				count_};
	}

	// Getters for raw buffers (if needed for copy/reorder)
	DeviceBuffer<Vector3>& pos() {
		return pos_;
	}
	DeviceBuffer<int>& id() {
		return id_;
	}
	DeviceBuffer<uint32_t>& flags() {
		return flags_;
	}
	DeviceBuffer<Vector3>& ForceEnergy() {
		return ForceEnergy_;
	}
	DeviceBuffer<Vector3>& orient() {
		return orient_;
	}
	DeviceBuffer<int>& type_id() {
		return type_id_;
	}
	DeviceBuffer<Vector3>& mom() {
		return mom_;
	}

	// Management
	idx_t size() const {
		return count_;
	}
	idx_t capacity() const {
		return capacity_;
	}
	void set_size(idx_t n) {
		count_ = n;
	} // Be careful with this

	// Scratchpad Swap (For sorting)
	// Swaps internal pointers with another buffer set (efficient double buffering)
	void swap(DeviceParticle& other);

  private:
	idx_t capacity_;
	idx_t count_;
	Resource resource_;

	DeviceBuffer<int> id_;
	DeviceBuffer<int> type_id_;
	DeviceBuffer<Vector3> pos_;
	DeviceBuffer<Vector3> mom_;
	DeviceBuffer<Vector3> ForceEnergy_;
	DeviceBuffer<Vector3> orient_;
	DeviceBuffer<uint32_t> flags_; // Replaces 3 bool arrays
};

// ============================================================================
// 5. STATIC DATA (Particle Types)
// ============================================================================
// Keep this struct simple and packed (16-byte aligned if possible)
struct ParticleTypeView {
	float* mass;
	float* charge;
	float* radius;
	float* eps;
	float* diffusion;
	Vector3* transDamping;
	float* mu;
	float* pmf_scale;
	float* pmf_scale_slope;
	float* pmf_smd_freq;
	int* pmf_grid_id;
	int* diffusion_grid_id;
	int3* force_grid_id;
};

class DeviceParticleTypes {
  public:
	DeviceParticleTypes(const std::vector<ParticleType>& types, const Resource& res) {
		mass_ = DeviceBuffer<float>(types.size(), res);
		charge_ = DeviceBuffer<float>(types.size(), res);
		radius_ = DeviceBuffer<float>(types.size(), res);
		eps_ = DeviceBuffer<float>(types.size(), res);
		diffusion_ = DeviceBuffer<float>(types.size(), res);
		transDamping_ = DeviceBuffer<Vector3>(types.size(), res);
		mu_ = DeviceBuffer<float>(types.size(), res);
		pmf_scale_ = DeviceBuffer<float>(types.size(), res);
		pmf_scale_slope_ = DeviceBuffer<float>(types.size(), res);
		pmf_smd_freq_ = DeviceBuffer<float>(types.size(), res);
		pmf_grid_id_ = DeviceBuffer<int>(types.size(), res);
		diffusion_grid_id_ = DeviceBuffer<int>(types.size(), res);
		force_grid_id_ = DeviceBuffer<int3>(types.size(), res);
	};

  private:
	DeviceBuffer<float> mass_;
	DeviceBuffer<float> charge_;
	DeviceBuffer<float> radius_;
	DeviceBuffer<float> eps_;
	DeviceBuffer<float> diffusion_;
	DeviceBuffer<Vector3> transDamping_;
	DeviceBuffer<float> mu_;
	DeviceBuffer<float> pmf_scale_;
	DeviceBuffer<float> pmf_scale_slope_;
	DeviceBuffer<float> pmf_smd_freq_;
	DeviceBuffer<int> pmf_grid_id_;
	DeviceBuffer<int> diffusion_grid_id_;
	DeviceBuffer<int3> force_grid_id_;
};

} // namespace ARBD
