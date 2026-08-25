#pragma once
#include "Backend/Buffer.h"
#include "DeviceRigidBody.h"
#include "RigidBodyProperties.h"
#include "Types/Types.h"

namespace MARS {

class DeviceRigidBody {
  public:
	DeviceRigidBody(idx_t capacity, const Resource& resource)
		: capacity_(capacity), count_(0), resource_(resource), id_(capacity, resource),
		  type_id_(capacity, resource), position_(capacity, resource),
		  orientation_(capacity, resource), momentum_(capacity, resource),
		  angular_momentum_(capacity, resource), force_(capacity, resource),
		  torque_(capacity, resource), external_force_(capacity, resource),
		  external_torque_(capacity, resource) {
		// Zeroed once here, not per step: an external load persists until its
		// owner (e.g. the scuff-em coupling) overwrites it, so a system with no
		// external source must read zeros rather than uninitialized device memory.
		external_force_.fill(Vector3(0.0f));
		external_torque_.fill(Vector3(0.0f));
	}

	// Get View for Kernels
	//
	// Every pointer in RigidBodyView must be supplied here: the view is an
	// aggregate, so any field left off the end is value-initialized to nullptr
	// and the first kernel to touch it faults on the device. external_force/
	// external_torque are read unconditionally by RBLangevinForceKernel.
	RigidBodyView view() {
		return {id_.data(),
				type_id_.data(),
				position_.data(),
				orientation_.data(),
				momentum_.data(),
				angular_momentum_.data(),
				force_.data(),
				torque_.data(),
				external_force_.data(),
				external_torque_.data()};
	}

	ConstRigidBodyView view() const {
		return {id_.data(),
				type_id_.data(),
				position_.data(),
				orientation_.data(),
				momentum_.data(),
				angular_momentum_.data(),
				force_.data(),
				torque_.data(),
				external_force_.data(),
				external_torque_.data()};
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
	// Written by whatever supplies an external load (scuff-em plasmonic forces);
	// deliberately not cleared by clear_forces(), so a value written here holds
	// until it is overwritten.
	DeviceBuffer<Vector3>& external_force() {
		return external_force_;
	}
	DeviceBuffer<Vector3>& external_torque() {
		return external_torque_;
	}

	// Management
	idx_t size() const {
		return count_;
	}
	idx_t capacity() const {
		return capacity_;
	}

	// See DeviceParticle::clear_forces() for why these fills are synchronous
	// and operate in place rather than calling DeviceBuffer::clear().
	void clear_forces() {
		force_.fill(Vector3(0.0f, 0.0f, 0.0f), true);
		torque_.fill(Vector3(0.0f, 0.0f, 0.0f), true);
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
	// Persistent per-body external load (e.g. plasmonic forces from scuff-em).
	// Kept separate from force_/torque_ precisely because those are zeroed every
	// step by clear_forces(), whereas an external load must persist until
	// whoever owns it rewrites it. Zero-initialized, so systems with no external
	// source behave exactly as before.
	DeviceBuffer<Vector3> external_force_;
	DeviceBuffer<Vector3> external_torque_;
};

class DeviceRigidBodyTypes {
  public:
	DeviceRigidBodyTypes(const std::vector<RigidBodyType>& types, const Resource& res) {
		const idx_t n = types.size();
		mass_ = DeviceBuffer<mars_real>(n, res);
		inertia_ = DeviceBuffer<Vector3>(n, res);
		trans_damping_ = DeviceBuffer<Vector3>(n, res);
		rot_damping_ = DeviceBuffer<Vector3>(n, res);
		trans_force_coeff_ = DeviceBuffer<Vector3>(n, res);
		rot_torque_coeff_ = DeviceBuffer<Vector3>(n, res);
		rot_diffusivity_ = DeviceBuffer<mars_real>(n, res);
		rot_damping_coefficient_ = DeviceBuffer<mars_real>(n, res);
		potential_grid_offset_ = DeviceBuffer<int>(n, res);
		potential_grid_count_ = DeviceBuffer<int>(n, res);
		density_grid_offset_ = DeviceBuffer<int>(n, res);
		density_grid_count_ = DeviceBuffer<int>(n, res);
		pmf_grid_offset_ = DeviceBuffer<int>(n, res);
		pmf_grid_count_ = DeviceBuffer<int>(n, res);

		idx_t total_grid_terms = 0;
		for (const auto& t : types) {
			total_grid_terms += t.potential_grids.size() + t.density_grids.size() + t.pmf_grids.size();
		}
		// A size-0 DeviceBuffer isn't a meaningful case to support here, so
		// always allocate at least 1 entry.
		const idx_t grid_term_capacity = total_grid_terms > 0 ? total_grid_terms : 1;
		grid_terms_ = DeviceBuffer<GridTerm>(grid_term_capacity, res);

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
		std::vector<GridTerm> grid_terms;
		grid_terms.reserve(grid_terms_.size());

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
			potential_grid_count[i] = static_cast<int>(t.potential_grids.size());
			grid_terms.insert(grid_terms.end(), t.potential_grids.begin(), t.potential_grids.end());
			offset += potential_grid_count[i];

			density_grid_offset[i] = offset;
			density_grid_count[i] = static_cast<int>(t.density_grids.size());
			grid_terms.insert(grid_terms.end(), t.density_grids.begin(), t.density_grids.end());
			offset += density_grid_count[i];

			pmf_grid_offset[i] = offset;
			pmf_grid_count[i] = static_cast<int>(t.pmf_grids.size());
			grid_terms.insert(grid_terms.end(), t.pmf_grids.begin(), t.pmf_grids.end());
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
		if (!grid_terms.empty()) {
			grid_terms_.copy_from_host(grid_terms.data(), grid_terms.size());
		}
	}

	DeviceBuffer<mars_real>& mass() {
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
	DeviceBuffer<mars_real>& rot_diffusivity() {
		return rot_diffusivity_;
	}
	DeviceBuffer<mars_real>& rot_damping_coefficient() {
		return rot_damping_coefficient_;
	}
	DeviceBuffer<GridTerm>& grid_terms() {
		return grid_terms_;
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
				grid_terms_.data()};
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
				const_cast<GridTerm*>(grid_terms_.data())};
	}

	idx_t size() const {
		return mass_.size();
	}

  private:
	DeviceBuffer<mars_real> mass_;
	DeviceBuffer<Vector3> inertia_;
	DeviceBuffer<Vector3> trans_damping_;
	DeviceBuffer<Vector3> rot_damping_;
	DeviceBuffer<Vector3> trans_force_coeff_;
	DeviceBuffer<Vector3> rot_torque_coeff_;
	DeviceBuffer<mars_real> rot_diffusivity_;
	DeviceBuffer<mars_real> rot_damping_coefficient_;
	DeviceBuffer<int> potential_grid_offset_;
	DeviceBuffer<int> potential_grid_count_;
	DeviceBuffer<int> density_grid_offset_;
	DeviceBuffer<int> density_grid_count_;
	DeviceBuffer<int> pmf_grid_offset_;
	DeviceBuffer<int> pmf_grid_count_;
	// Flat, shared across all types: laid out per type as [potential
	// terms][density terms][pmf terms] contiguously (see RigidBodyTypeView).
	DeviceBuffer<GridTerm> grid_terms_;
};

} // namespace MARS
