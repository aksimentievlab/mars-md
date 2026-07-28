#pragma once
#include "Backend/Buffer.h"
#include "DeviceParticle.h"
#include "ParticleProperties.h"
#include "Types/Types.h"

namespace ARBD {

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
				flags_.data()};
	}

	ConstParticleView view() const {
		return {id_.data(),
				type_id_.data(),
				pos_.data(),
				mom_.data(),
				ForceEnergy_.data(),
				orient_.data(),
				flags_.data()};
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

	// NOTE: DeviceBuffer::clear() deallocates the underlying storage - it does not
	// zero it. These clear_*() methods are called every step (e.g. before force
	// accumulation) and must zero in place instead, or the buffer would be freed
	// after the first call and every subsequent access would read/write a dangling
	// pointer.
	void clear_forces() {
		ForceEnergy_.fill(Vector3(0.0f, 0.0f, 0.0f));
	}
	void clear_orientations() {
		orient_.fill(Vector3(0.0f, 0.0f, 0.0f));
	}
	void clear_flags() {
		flags_.fill(0u);
	}

	// Bulk Copy Helper: Host -> Device (Structure of Arrays)
	void copy_from_host(const HostParticleData& host, idx_t count) {
		if (count > capacity_) {
			throw Exception(
				ExceptionType::RuntimeError,
				SourceLocation(),
				"DeviceParticle overflow: attempting to copy {} particles into capacity {}",
				count,
				capacity_);
		}

		// Assuming HostParticleData vectors are at least 'count' size
		id_.copy_from_host(host.global_id.data(), count);
		type_id_.copy_from_host(host.type_id.data(), count);
		pos_.copy_from_host(host.pos.data(), count);
		mom_.copy_from_host(host.mom.data(), count);

		// ForceEnergy is usually computed on device, but we copy for restart/debug
		// Assuming host.force maps to Force part of ForceEnergy, need careful handling if
		// packed For now assuming direct mapping or skipping force copy if not needed for init

		// Orient
		if (!host.orient.empty())
			orient_.copy_from_host(host.orient.data(), count);

		// Flags
		if (!host.flags.empty())
			flags_.copy_from_host(host.flags.data(), count);

		count_ = count;
	}

	// Bulk Copy Helper: Device -> Host
	void copy_to_host(HostParticleData& host, idx_t count) const {
		// Ensure host vectors are sized appropriately
		if (host.size() < count)
			host.resize(count);

		id_.copy_to_host(host.global_id.data(), count);
		type_id_.copy_to_host(host.type_id.data(), count);
		pos_.copy_to_host(host.pos.data(), count);
		mom_.copy_to_host(host.mom.data(), count);
		orient_.copy_to_host(host.orient.data(), count);
		flags_.copy_to_host(host.flags.data(), count);

		// ForceEnergy_ packs force (xyz) and energy (t) together; unpack
		// into host.force/host.energy separately. This used to be a
		// separate copy_force_energy_to_host() that nothing ever called, so
		// host.force silently stayed at whatever it was default-constructed
		// to (zero) regardless of the actual on-device force.
		std::vector<Vector3> temp(count);
		ForceEnergy_.copy_to_host(temp.data(), count);
		for (idx_t i = 0; i < count; ++i) {
			host.force[i].x = temp[i].x;
			host.force[i].y = temp[i].y;
			host.force[i].z = temp[i].z;
			host.force[i].t = 0.0f;
			host.energy[i] = temp[i].t;
		}
	}

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

class DeviceParticleTypes {
  public:
	DeviceParticleTypes(const std::vector<ParticleType>& types, const Resource& res) {
		mass_ = DeviceBuffer<float>(types.size(), res);
		charge_ = DeviceBuffer<float>(types.size(), res);
		radius_ = DeviceBuffer<float>(types.size(), res);
		eps_ = DeviceBuffer<float>(types.size(), res);
		diffusion_ = DeviceBuffer<Vector3>(types.size(), res);
		trans_damping_ = DeviceBuffer<Vector3>(types.size(), res);
		mu_ = DeviceBuffer<float>(types.size(), res);
		pmf_scale_ = DeviceBuffer<float>(types.size(), res);
		pmf_scale_slope_ = DeviceBuffer<float>(types.size(), res);
		pmf_smd_freq_ = DeviceBuffer<float>(types.size(), res);
		pmf_grid_id_ = DeviceBuffer<int>(types.size(), res);
		diffusion_grid_id_ = DeviceBuffer<int>(types.size(), res);
		force_grid_id_ = DeviceBuffer<int3>(types.size(), res);
		copy_from_host(types);
	}

	void copy_from_host(const std::vector<ParticleType>& types) {
		std::vector<float> mass(types.size());
		std::vector<float> charge(types.size());
		std::vector<float> radius(types.size());
		std::vector<float> eps(types.size());
		std::vector<Vector3> diffusion(types.size());
		std::vector<Vector3> trans_damping(types.size());
		std::vector<float> mu(types.size());
		std::vector<float> pmf_scale(types.size());
		std::vector<float> pmf_scale_slope(types.size());
		std::vector<float> pmf_smd_freq(types.size());
		std::vector<int> pmf_grid_id(types.size());
		std::vector<int> diffusion_grid_id(types.size());
		std::vector<int3> force_grid_id(types.size());
		for (size_t i = 0; i < types.size(); i++) {
			mass[i] = types[i].mass;
			charge[i] = types[i].charge;
			radius[i] = types[i].radius;
			eps[i] = types[i].eps;
			diffusion[i] = types[i].diffusion;
			trans_damping[i] = types[i].trans_damping;
			mu[i] = types[i].mu;
			pmf_scale[i] = types[i].pmf_scale;
			pmf_scale_slope[i] = types[i].pmf_scale_slope;
			pmf_smd_freq[i] = types[i].pmf_smd_freq;
			pmf_grid_id[i] = types[i].pmf_grid_id;
			diffusion_grid_id[i] = types[i].diffusion_grid_id;
			force_grid_id[i] = types[i].force_grid_id;
		}
		mass_.copy_from_host(mass.data(), types.size());
		charge_.copy_from_host(charge.data(), types.size());
		radius_.copy_from_host(radius.data(), types.size());
		eps_.copy_from_host(eps.data(), types.size());
		diffusion_.copy_from_host(diffusion.data(), types.size());
		trans_damping_.copy_from_host(trans_damping.data(), types.size());
		mu_.copy_from_host(mu.data(), types.size());
		pmf_scale_.copy_from_host(pmf_scale.data(), types.size());
		pmf_scale_slope_.copy_from_host(pmf_scale_slope.data(), types.size());
		pmf_smd_freq_.copy_from_host(pmf_smd_freq.data(), types.size());
		pmf_grid_id_.copy_from_host(pmf_grid_id.data(), types.size());
		diffusion_grid_id_.copy_from_host(diffusion_grid_id.data(), types.size());
		force_grid_id_.copy_from_host(force_grid_id.data(), types.size());
	}
	DeviceBuffer<float>& mass() {
		return mass_;
	}
	DeviceBuffer<float>& charge() {
		return charge_;
	}
	DeviceBuffer<float>& radius() {
		return radius_;
	}
	DeviceBuffer<Vector3>& trans_damping() {
		return trans_damping_;
	}
	DeviceBuffer<Vector3>& diffusion() {
		return diffusion_;
	}
	DeviceBuffer<float>& mu() {
		return mu_;
	}
	DeviceBuffer<float>& pmf_scale() {
		return pmf_scale_;
	}
	DeviceBuffer<float>& pmf_scale_slope() {
		return pmf_scale_slope_;
	}
	DeviceBuffer<float>& pmf_smd_freq() {
		return pmf_smd_freq_;
	}
	DeviceBuffer<int>& pmf_grid_id() {
		return pmf_grid_id_;
	}
	DeviceBuffer<int>& diffusion_grid_id() {
		return diffusion_grid_id_;
	}
	DeviceBuffer<int3>& force_grid_id() {
		return force_grid_id_;
	}

	// Get View for Kernels
	ParticleTypeView view() {
		return {mass_.data(),
				charge_.data(),
				radius_.data(),
				eps_.data(),
				diffusion_.data(),
				trans_damping_.data(),
				mu_.data(),
				pmf_scale_.data(),
				pmf_scale_slope_.data(),
				pmf_smd_freq_.data(),
				pmf_grid_id_.data(),
				diffusion_grid_id_.data(),
				force_grid_id_.data()};
	}

	const ParticleTypeView view() const {
		return {const_cast<float*>(mass_.data()),
				const_cast<float*>(charge_.data()),
				const_cast<float*>(radius_.data()),
				const_cast<float*>(eps_.data()),
				const_cast<Vector3*>(diffusion_.data()),
				const_cast<Vector3*>(trans_damping_.data()),
				const_cast<float*>(mu_.data()),
				const_cast<float*>(pmf_scale_.data()),
				const_cast<float*>(pmf_scale_slope_.data()),
				const_cast<float*>(pmf_smd_freq_.data()),
				const_cast<int*>(pmf_grid_id_.data()),
				const_cast<int*>(diffusion_grid_id_.data()),
				const_cast<int3*>(force_grid_id_.data())};
	}

  private:
	DeviceBuffer<float> mass_;
	DeviceBuffer<float> charge_;
	DeviceBuffer<float> radius_;
	DeviceBuffer<float> eps_;
	DeviceBuffer<Vector3> diffusion_;
	DeviceBuffer<Vector3> trans_damping_;
	DeviceBuffer<float> mu_;
	DeviceBuffer<float> pmf_scale_;
	DeviceBuffer<float> pmf_scale_slope_;
	DeviceBuffer<float> pmf_smd_freq_;
	DeviceBuffer<int> pmf_grid_id_;
	DeviceBuffer<int> diffusion_grid_id_;
	DeviceBuffer<int3> force_grid_id_;
};

} // namespace ARBD
