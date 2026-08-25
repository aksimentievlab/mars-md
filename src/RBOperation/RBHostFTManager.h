#pragma once
#include "MARSException.h"
#include "ApplyHostForce.h"
#include "Backend/Buffer.h"
#include "Backend/Events.h"
#include "Backend/KernelConfig.h"
#include "Backend/Kernels.h"
#include "Backend/Resource.h"
#include "Constants.h"
#include "Header.h"
#include "Objects/DeviceRigidBody.h"
#include "Types/Types.h"

#include <algorithm>
#include <span>
#include <vector>
namespace MARS {
/**
 * @brief Pushes host-computed per-body external force/torque onto the device.
 *
 * Only the listed bodies are transferred; ApplyExternalForcesKernel scatters
 * them into the full external_force/external_torque SoA by rigid-body index, so
 * untouched bodies keep whatever they already held.
 * @see RBHostFTManager.md
 */
class RBHostFTManager {
  public:
	RBHostFTManager(idx_t capacity, const Resource& resource)
		: resource_(resource), capacity_(capacity), id_(capacity, resource),
		  force_(capacity, resource), torque_(capacity, resource),
		  baseline_force_(capacity, Vector3(0.0f)), baseline_torque_(capacity, Vector3(0.0f)) {}

	idx_t capacity() const {
		return capacity_;
	}

	/**
	 * @brief Sets the always-on contribution every push is added to.
	 * @param force Per-body constant force, indexed by device SoA slot.
	 * @param torque Per-body constant torque, same indexing.
	 * @details constantForce/constantTorque are static for the run, so they are
	 *          held here rather than re-sent each publish. Without this, a push
	 *          would assign over them and the constant would vanish.
	 */
	void set_baseline(std::span<const Vector3> force, std::span<const Vector3> torque) {
		if (force.size() != torque.size()) {
			MARS_Exception(ExceptionType::ValueError,
						   "RBHostFTManager: %zu baseline forces but %zu torques",
						   force.size(),
						   torque.size());
		}
		const idx_t n = std::min(capacity_, static_cast<idx_t>(force.size()));
		for (idx_t i = 0; i < n; ++i) {
			baseline_force_[i] = force[i];
			baseline_torque_[i] = torque[i];
		}
	}

	/**
	 * @brief Pushes baseline + this producer's contribution for the listed bodies.
	 * @see push() for the argument contract.
	 */
	Event push_with_baseline(std::span<const int> rigid_body_index,
							 std::span<const Vector3> force,
							 std::span<const Vector3> torque,
							 RigidBodyView rb) {
		const idx_t n = static_cast<idx_t>(rigid_body_index.size());
		if (force.size() != n || torque.size() != n) {
			MARS_Exception(ExceptionType::ValueError,
						   "RBHostFTManager: %zu indices but %zu forces and %zu torques",
						   n,
						   force.size(),
						   torque.size());
		}
		staged_force_.resize(n);
		staged_torque_.resize(n);
		for (idx_t i = 0; i < n; ++i) {
			const idx_t slot = static_cast<idx_t>(rigid_body_index[i]);
			if (slot >= capacity_) {
				MARS_Exception(ExceptionType::ValueError,
							   "RBHostFTManager: body index %zu out of range [0,%zu)",
							   slot,
							   capacity_);
			}
			staged_force_[i] = force[i] + baseline_force_[slot];
			staged_torque_[i] = torque[i] + baseline_torque_[slot];
		}
		return push(rigid_body_index, staged_force_, staged_torque_, rb);
	}

	/**
	 * @brief Pushes the baseline alone, for every body.
	 * @details Call once after set_baseline() so constants reach the device even
	 *          when no dynamic producer ever pushes.
	 */
	Event push_baseline(RigidBodyView rb) {
		staged_id_.resize(capacity_);
		for (idx_t i = 0; i < capacity_; ++i) {
			staged_id_[i] = static_cast<int>(i);
		}
		return push(staged_id_, baseline_force_, baseline_torque_, rb);
	}

	/**
	 * @brief Sets external force/torque on the listed bodies.
	 * @param rigid_body_index Device SoA slot per entry.
	 * @param force Lab-frame force, kcal/mol/Angstrom.
	 * @param torque Lab-frame torque, kcal/mol.
	 * @param rb Target view.
	 * @throws Exception on a length mismatch or an over-capacity push.
	 */
	Event push(std::span<const int> rigid_body_index,
			   std::span<const Vector3> force,
			   std::span<const Vector3> torque,
			   RigidBodyView rb) {
		const idx_t n = static_cast<idx_t>(rigid_body_index.size());
		if (force.size() != n || torque.size() != n) {
			MARS_Exception(ExceptionType::ValueError,
						   "RBHostFTManager: %zu indices but %zu forces and %zu torques",
						   n,
						   force.size(),
						   torque.size());
		}
		if (n == 0) {
			return Event(nullptr, resource_);
		}
		if (n > capacity_) {
			MARS_Exception(ExceptionType::RuntimeError,
						   "RBHostFTManager overflow: %zu entries into capacity %zu",
						   n,
						   capacity_);
		}

		id_.copy_from_host(rigid_body_index.data(), n);
		force_.copy_from_host(force.data(), n);
		torque_.copy_from_host(torque.data(), n);

		ExternalForceView view{id_.data(), force_.data(), torque_.data()};
		KernelConfig config = KernelConfig::for_1d(n, resource_);
		return launch_kernel(resource_, config, ApplyExternalForcesKernel(rb, view, n));
	}

  private:
	Resource resource_;
	idx_t capacity_;
	DeviceBuffer<int> id_;
	DeviceBuffer<Vector3> force_;
	DeviceBuffer<Vector3> torque_;

	// Full-length, indexed by device SoA slot.
	std::vector<Vector3> baseline_force_;
	std::vector<Vector3> baseline_torque_;
	// Scratch for push_with_baseline/push_baseline, kept as members so a
	// per-step publish does not reallocate.
	std::vector<Vector3> staged_force_;
	std::vector<Vector3> staged_torque_;
	std::vector<int> staged_id_;
};
} // namespace MARS
