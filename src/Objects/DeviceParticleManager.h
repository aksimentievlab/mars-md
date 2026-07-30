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
	// `need_energy` gates the ForceEnergy_ device->host copy and its
	// per-particle unpack loop: host.force is never read anywhere in the
	// codebase, and host.energy is only read by SimManager::write_energy_output.
	// DCD-frame and restart-only gathers (the common case) don't need either,
	// so skip this field-copy + O(N) unpack loop for them.
	void copy_to_host(HostParticleData& host, idx_t count, bool need_energy = false) const {
		// Ensure host vectors are sized appropriately
		if (host.size() < count)
			host.resize(count);

		id_.copy_to_host(host.global_id.data(), count);
		type_id_.copy_to_host(host.type_id.data(), count);
		pos_.copy_to_host(host.pos.data(), count);
		mom_.copy_to_host(host.mom.data(), count);
		orient_.copy_to_host(host.orient.data(), count);
		flags_.copy_to_host(host.flags.data(), count);

		if (!need_energy)
			return;
		// write energy to host
		std::vector<Vector3> temp(count);
		ForceEnergy_.copy_to_host(temp.data(), count);
		for (idx_t i = 0; i < count; ++i) {
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
		pmf_smd_freq_ = DeviceBuffer<uint32_t>(types.size(), res);
		pmf_grid_offset_ = DeviceBuffer<int>(types.size(), res);
		pmf_grid_count_ = DeviceBuffer<int>(types.size(), res);

		// Zero terms is legitimate (no type declared a gridFile); Buffer's ctor
		// skips allocation for count 0 and the kernel's per-type count is 0 too,
		// so the null data() is never reached.
		size_t total_pmf_terms = 0;
		for (const auto& t : types) {
			total_pmf_terms += t.pmf_grids.size();
		}
		pmf_grid_terms_ = DeviceBuffer<GridTerm>(total_pmf_terms, res);
		diffusion_grid_id_ = DeviceBuffer<int>(types.size(), res);
		force_grid_id_ = DeviceBuffer<int3>(types.size(), res);
		force_grid_scale_ = DeviceBuffer<Vector3>(types.size(), res);
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
		std::vector<uint32_t> pmf_smd_freq(types.size());
		std::vector<int> pmf_grid_offset(types.size());
		std::vector<int> pmf_grid_count(types.size());
		std::vector<GridTerm> pmf_grid_terms;
		pmf_grid_terms.reserve(pmf_grid_terms_.size());
		std::vector<int> diffusion_grid_id(types.size());
		std::vector<int3> force_grid_id(types.size());
		std::vector<Vector3> force_grid_scale(types.size());
		for (size_t i = 0; i < types.size(); i++) {
			mass[i] = types[i].mass;
			charge[i] = types[i].charge;
			radius[i] = types[i].radius;
			eps[i] = types[i].eps;
			diffusion[i] = types[i].diffusion;
			trans_damping[i] = types[i].trans_damping;
			mu[i] = types[i].mu;
			pmf_smd_freq[i] = types[i].pmf_smd_freq;

			// Flatten this type's PMF grid list into the shared term table; the
			// kernel walks [offset, offset + count).
			pmf_grid_offset[i] = static_cast<int>(pmf_grid_terms.size());
			pmf_grid_count[i] = static_cast<int>(types[i].pmf_grids.size());
			pmf_grid_terms.insert(pmf_grid_terms.end(),
								  types[i].pmf_grids.begin(),
								  types[i].pmf_grids.end());

			diffusion_grid_id[i] = types[i].diffusion_grid_id;
			force_grid_id[i] = types[i].force_grid_id;
			force_grid_scale[i] = types[i].force_grid_scale;
		}
		mass_.copy_from_host(mass.data(), types.size());
		charge_.copy_from_host(charge.data(), types.size());
		radius_.copy_from_host(radius.data(), types.size());
		eps_.copy_from_host(eps.data(), types.size());
		diffusion_.copy_from_host(diffusion.data(), types.size());
		trans_damping_.copy_from_host(trans_damping.data(), types.size());
		mu_.copy_from_host(mu.data(), types.size());
		pmf_smd_freq_.copy_from_host(pmf_smd_freq.data(), types.size());
		pmf_grid_offset_.copy_from_host(pmf_grid_offset.data(), types.size());
		pmf_grid_count_.copy_from_host(pmf_grid_count.data(), types.size());
		if (!pmf_grid_terms.empty()) {
			pmf_grid_terms_.copy_from_host(pmf_grid_terms.data(), pmf_grid_terms.size());
		}
		diffusion_grid_id_.copy_from_host(diffusion_grid_id.data(), types.size());
		force_grid_id_.copy_from_host(force_grid_id.data(), types.size());
		force_grid_scale_.copy_from_host(force_grid_scale.data(), types.size());
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
	DeviceBuffer<uint32_t>& pmf_smd_freq() {
		return pmf_smd_freq_;
	}
	DeviceBuffer<int>& pmf_grid_offset() {
		return pmf_grid_offset_;
	}
	DeviceBuffer<int>& pmf_grid_count() {
		return pmf_grid_count_;
	}
	DeviceBuffer<GridTerm>& pmf_grid_terms() {
		return pmf_grid_terms_;
	}
	DeviceBuffer<int>& diffusion_grid_id() {
		return diffusion_grid_id_;
	}
	DeviceBuffer<int3>& force_grid_id() {
		return force_grid_id_;
	}
	DeviceBuffer<Vector3>& force_grid_scale() {
		return force_grid_scale_;
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
				pmf_smd_freq_.data(),
				pmf_grid_offset_.data(),
				pmf_grid_count_.data(),
				pmf_grid_terms_.data(),
				diffusion_grid_id_.data(),
				force_grid_id_.data(),
				force_grid_scale_.data()};
	}

	const ParticleTypeView view() const {
		return {const_cast<float*>(mass_.data()),
				const_cast<float*>(charge_.data()),
				const_cast<float*>(radius_.data()),
				const_cast<float*>(eps_.data()),
				const_cast<Vector3*>(diffusion_.data()),
				const_cast<Vector3*>(trans_damping_.data()),
				const_cast<float*>(mu_.data()),
				const_cast<uint32_t*>(pmf_smd_freq_.data()),
				const_cast<int*>(pmf_grid_offset_.data()),
				const_cast<int*>(pmf_grid_count_.data()),
				const_cast<GridTerm*>(pmf_grid_terms_.data()),
				const_cast<int*>(diffusion_grid_id_.data()),
				const_cast<int3*>(force_grid_id_.data()),
				const_cast<Vector3*>(force_grid_scale_.data())};
	}

	idx_t size() const {
		return mass_.size();
	}

  private:
	DeviceBuffer<float> mass_;
	DeviceBuffer<float> charge_;
	DeviceBuffer<float> radius_;
	DeviceBuffer<float> eps_;
	DeviceBuffer<Vector3> diffusion_;
	DeviceBuffer<Vector3> trans_damping_;
	DeviceBuffer<float> mu_;
	DeviceBuffer<uint32_t> pmf_smd_freq_;
	// offset+count per type into the flat pmf_grid_terms_ table (same layout as
	// DeviceRigidBodyTypes' grid_terms_).
	DeviceBuffer<int> pmf_grid_offset_;
	DeviceBuffer<int> pmf_grid_count_;
	DeviceBuffer<GridTerm> pmf_grid_terms_;
	DeviceBuffer<int> diffusion_grid_id_;
	DeviceBuffer<int3> force_grid_id_;
	DeviceBuffer<Vector3> force_grid_scale_;
};

} // namespace ARBD
