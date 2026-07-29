#pragma once
#include "Backend/Buffer.h"
#include "DeviceRigidBody.h"
#include "RigidBodyProperties.h"
#include "Types/Types.h"

namespace ARBD {

class DeviceRigidBody {
  public:
	DeviceRigidBody(idx_t capacity, const Resource& resource)
		: capacity_(capacity), count_(0), resource_(resource), id_(capacity, resource),
		  type_id_(capacity, resource), position_(capacity, resource),
		  orientation_(capacity, resource), momentum_(capacity, resource),
		  angular_momentum_(capacity, resource), force_(capacity, resource),
		  torque_(capacity, resource) {}

	// Get View for Kernels
	RigidBodyView view() {
		return {id_.data(),
				type_id_.data(),
				position_.data(),
				orientation_.data(),
				momentum_.data(),
				angular_momentum_.data(),
				force_.data(),
				torque_.data()};
	}

	ConstRigidBodyView view() const {
		return {id_.data(),
				type_id_.data(),
				position_.data(),
				orientation_.data(),
				momentum_.data(),
				angular_momentum_.data(),
				force_.data(),
				torque_.data()};
	}

	// Getters for raw buffers (if needed for copy/reorder)
	DeviceBuffer<Vector3>& position() {
		return position_;
	}
	DeviceBuffer<int>& id() {
		return id_;
	}
	DeviceBuffer<int>& type_id() {
		return type_id_;
	}
	DeviceBuffer<Matrix3>& orientation() {
		return orientation_;
	}
	DeviceBuffer<Vector3>& momentum() {
		return momentum_;
	}
	DeviceBuffer<Vector3>& angular_momentum() {
		return angular_momentum_;
	}
	DeviceBuffer<Vector3>& force() {
		return force_;
	}
	DeviceBuffer<Vector3>& torque() {
		return torque_;
	}

	// Management
	idx_t size() const {
		return count_;
	}
	idx_t capacity() const {
		return capacity_;
	}

	// See DeviceParticle::clear_forces() for why this fills in place rather
	// than calling DeviceBuffer::clear() (which deallocates).
	void clear_forces() {
		force_.fill(Vector3(0.0f, 0.0f, 0.0f));
		torque_.fill(Vector3(0.0f, 0.0f, 0.0f));
	}

	// Bulk Copy Helper: Host -> Device (Structure of Arrays)
	void copy_from_host(const HostRigidBodyData& host, idx_t count) {
		if (count > capacity_) {
			throw Exception(
				ExceptionType::RuntimeError,
				SourceLocation(),
				"DeviceRigidBody overflow: attempting to copy {} rigid bodies into capacity {}",
				count,
				capacity_);
		}

		id_.copy_from_host(host.global_id.data(), count);
		type_id_.copy_from_host(host.type_id.data(), count);
		position_.copy_from_host(host.position.data(), count);
		orientation_.copy_from_host(host.orientation.data(), count);
		momentum_.copy_from_host(host.momentum.data(), count);
		angular_momentum_.copy_from_host(host.angular_momentum.data(), count);
		force_.copy_from_host(host.force.data(), count);
		torque_.copy_from_host(host.torque.data(), count);

		count_ = count;
	}

	// Bulk Copy Helper: Device -> Host
	void copy_to_host(HostRigidBodyData& host, idx_t count) const {
		if (host.size() < count)
			host.resize(count);

		id_.copy_to_host(host.global_id.data(), count);
		type_id_.copy_to_host(host.type_id.data(), count);
		position_.copy_to_host(host.position.data(), count);
		orientation_.copy_to_host(host.orientation.data(), count);
		momentum_.copy_to_host(host.momentum.data(), count);
		angular_momentum_.copy_to_host(host.angular_momentum.data(), count);
		force_.copy_to_host(host.force.data(), count);
		torque_.copy_to_host(host.torque.data(), count);
	}

	void swap(DeviceRigidBody& other);

  private:
	idx_t capacity_;
	idx_t count_;
	Resource resource_;

	DeviceBuffer<int> id_;
	DeviceBuffer<int> type_id_;
	DeviceBuffer<Vector3> position_;
	DeviceBuffer<Matrix3> orientation_;
	DeviceBuffer<Vector3> momentum_;
	DeviceBuffer<Vector3> angular_momentum_;
	DeviceBuffer<Vector3> force_;
	DeviceBuffer<Vector3> torque_;
};

class DeviceRigidBodyTypes {
  public:
	DeviceRigidBodyTypes(const std::vector<RigidBodyType>& types, const Resource& res) {
		const idx_t n = types.size();
		mass_ = DeviceBuffer<float>(n, res);
		inertia_ = DeviceBuffer<Vector3>(n, res);
		trans_damping_ = DeviceBuffer<Vector3>(n, res);
		rot_damping_ = DeviceBuffer<Vector3>(n, res);
		trans_force_coeff_ = DeviceBuffer<Vector3>(n, res);
		rot_torque_coeff_ = DeviceBuffer<Vector3>(n, res);
		rot_diffusivity_ = DeviceBuffer<float>(n, res);
		rot_damping_coefficient_ = DeviceBuffer<float>(n, res);
		potential_grid_offset_ = DeviceBuffer<int>(n, res);
		potential_grid_count_ = DeviceBuffer<int>(n, res);
		density_grid_offset_ = DeviceBuffer<int>(n, res);
		density_grid_count_ = DeviceBuffer<int>(n, res);
		pmf_grid_offset_ = DeviceBuffer<int>(n, res);
		pmf_grid_count_ = DeviceBuffer<int>(n, res);

		idx_t total_grid_ids = 0;
		for (const auto& t : types) {
			total_grid_ids +=
				t.potential_grid_ids.size() + t.density_grid_ids.size() + t.pmf_grid_ids.size();
		}
		// A size-0 DeviceBuffer isn't a meaningful case to support here, so
		// always allocate at least 1 entry.
		const idx_t grid_id_capacity = total_grid_ids > 0 ? total_grid_ids : 1;
		grid_ids_ = DeviceBuffer<int>(grid_id_capacity, res);
		grid_scales_ = DeviceBuffer<float>(grid_id_capacity, res);

		copy_from_host(types);
	}

	void copy_from_host(const std::vector<RigidBodyType>& types) {
		const size_t n = types.size();
		std::vector<float> mass(n);
		std::vector<Vector3> inertia(n);
		std::vector<Vector3> trans_damping(n);
		std::vector<Vector3> rot_damping(n);
		std::vector<Vector3> trans_force_coeff(n);
		std::vector<Vector3> rot_torque_coeff(n);
		std::vector<float> rot_diffusivity(n);
		std::vector<float> rot_damping_coefficient(n);
		std::vector<int> potential_grid_offset(n);
		std::vector<int> potential_grid_count(n);
		std::vector<int> density_grid_offset(n);
		std::vector<int> density_grid_count(n);
		std::vector<int> pmf_grid_offset(n);
		std::vector<int> pmf_grid_count(n);
		std::vector<int> grid_ids;
		std::vector<float> grid_scales;
		grid_ids.reserve(grid_ids_.size());
		grid_scales.reserve(grid_ids_.size());

		// One scale value per grid id, in the same order (a missing/short
		// scale vector defaults to 1.0, i.e. unscaled).
		auto scale_or_one = [](const std::vector<float>& scales, size_t j) {
			return j < scales.size() ? scales[j] : 1.0f;
		};

		int offset = 0;
		for (size_t i = 0; i < n; i++) {
			const RigidBodyType& t = types[i];
			mass[i] = t.mass;
			inertia[i] = t.inertia;
			trans_damping[i] = t.trans_damping;
			rot_damping[i] = t.rot_damping;
			trans_force_coeff[i] = t.trans_force_coeff;
			rot_torque_coeff[i] = t.rot_torque_coeff;
			rot_diffusivity[i] = t.rot_diffusivity;
			rot_damping_coefficient[i] = t.rot_damping_coefficient;

			potential_grid_offset[i] = offset;
			potential_grid_count[i] = static_cast<int>(t.potential_grid_ids.size());
			for (size_t j = 0; j < t.potential_grid_ids.size(); j++) {
				grid_ids.push_back(static_cast<int>(t.potential_grid_ids[j]));
				grid_scales.push_back(scale_or_one(t.potential_grid_scale, j));
			}
			offset += potential_grid_count[i];

			density_grid_offset[i] = offset;
			density_grid_count[i] = static_cast<int>(t.density_grid_ids.size());
			for (size_t j = 0; j < t.density_grid_ids.size(); j++) {
				grid_ids.push_back(static_cast<int>(t.density_grid_ids[j]));
				grid_scales.push_back(scale_or_one(t.density_grid_scale, j));
			}
			offset += density_grid_count[i];

			pmf_grid_offset[i] = offset;
			pmf_grid_count[i] = static_cast<int>(t.pmf_grid_ids.size());
			for (size_t j = 0; j < t.pmf_grid_ids.size(); j++) {
				grid_ids.push_back(static_cast<int>(t.pmf_grid_ids[j]));
				grid_scales.push_back(scale_or_one(t.pmf_grid_scale, j));
			}
			offset += pmf_grid_count[i];
		}

		mass_.copy_from_host(mass.data(), n);
		inertia_.copy_from_host(inertia.data(), n);
		trans_damping_.copy_from_host(trans_damping.data(), n);
		rot_damping_.copy_from_host(rot_damping.data(), n);
		trans_force_coeff_.copy_from_host(trans_force_coeff.data(), n);
		rot_torque_coeff_.copy_from_host(rot_torque_coeff.data(), n);
		rot_diffusivity_.copy_from_host(rot_diffusivity.data(), n);
		rot_damping_coefficient_.copy_from_host(rot_damping_coefficient.data(), n);
		potential_grid_offset_.copy_from_host(potential_grid_offset.data(), n);
		potential_grid_count_.copy_from_host(potential_grid_count.data(), n);
		density_grid_offset_.copy_from_host(density_grid_offset.data(), n);
		density_grid_count_.copy_from_host(density_grid_count.data(), n);
		pmf_grid_offset_.copy_from_host(pmf_grid_offset.data(), n);
		pmf_grid_count_.copy_from_host(pmf_grid_count.data(), n);
		if (!grid_ids.empty()) {
			grid_ids_.copy_from_host(grid_ids.data(), grid_ids.size());
			grid_scales_.copy_from_host(grid_scales.data(), grid_scales.size());
		}
	}

	DeviceBuffer<float>& mass() {
		return mass_;
	}
	DeviceBuffer<Vector3>& inertia() {
		return inertia_;
	}
	DeviceBuffer<Vector3>& trans_damping() {
		return trans_damping_;
	}
	DeviceBuffer<Vector3>& rot_damping() {
		return rot_damping_;
	}
	DeviceBuffer<Vector3>& trans_force_coeff() {
		return trans_force_coeff_;
	}
	DeviceBuffer<Vector3>& rot_torque_coeff() {
		return rot_torque_coeff_;
	}
	DeviceBuffer<float>& rot_diffusivity() {
		return rot_diffusivity_;
	}
	DeviceBuffer<float>& rot_damping_coefficient() {
		return rot_damping_coefficient_;
	}
	DeviceBuffer<int>& grid_ids() {
		return grid_ids_;
	}
	DeviceBuffer<float>& grid_scales() {
		return grid_scales_;
	}

	// Get View for Kernels
	RigidBodyTypeView view() {
		return {mass_.data(),
				inertia_.data(),
				trans_damping_.data(),
				rot_damping_.data(),
				trans_force_coeff_.data(),
				rot_torque_coeff_.data(),
				rot_diffusivity_.data(),
				rot_damping_coefficient_.data(),
				potential_grid_offset_.data(),
				potential_grid_count_.data(),
				density_grid_offset_.data(),
				density_grid_count_.data(),
				pmf_grid_offset_.data(),
				pmf_grid_count_.data(),
				grid_ids_.data(),
				grid_scales_.data()};
	}

	const RigidBodyTypeView view() const {
		return {const_cast<float*>(mass_.data()),
				const_cast<Vector3*>(inertia_.data()),
				const_cast<Vector3*>(trans_damping_.data()),
				const_cast<Vector3*>(rot_damping_.data()),
				const_cast<Vector3*>(trans_force_coeff_.data()),
				const_cast<Vector3*>(rot_torque_coeff_.data()),
				const_cast<float*>(rot_diffusivity_.data()),
				const_cast<float*>(rot_damping_coefficient_.data()),
				const_cast<int*>(potential_grid_offset_.data()),
				const_cast<int*>(potential_grid_count_.data()),
				const_cast<int*>(density_grid_offset_.data()),
				const_cast<int*>(density_grid_count_.data()),
				const_cast<int*>(pmf_grid_offset_.data()),
				const_cast<int*>(pmf_grid_count_.data()),
				const_cast<int*>(grid_ids_.data()),
				const_cast<float*>(grid_scales_.data())};
	}

	idx_t size() const {
		return mass_.size();
	}

  private:
	DeviceBuffer<float> mass_;
	DeviceBuffer<Vector3> inertia_;
	DeviceBuffer<Vector3> trans_damping_;
	DeviceBuffer<Vector3> rot_damping_;
	DeviceBuffer<Vector3> trans_force_coeff_;
	DeviceBuffer<Vector3> rot_torque_coeff_;
	DeviceBuffer<float> rot_diffusivity_;
	DeviceBuffer<float> rot_damping_coefficient_;
	DeviceBuffer<int> potential_grid_offset_;
	DeviceBuffer<int> potential_grid_count_;
	DeviceBuffer<int> density_grid_offset_;
	DeviceBuffer<int> density_grid_count_;
	DeviceBuffer<int> pmf_grid_offset_;
	DeviceBuffer<int> pmf_grid_count_;
	// Flat, shared across all types: laid out per type as [potential
	// ids][density ids][pmf ids] contiguously (see RigidBodyTypeView).
	// grid_scales_ is the parallel per-grid scale factor at the same index.
	DeviceBuffer<int> grid_ids_;
	DeviceBuffer<float> grid_scales_;
};

} // namespace ARBD
