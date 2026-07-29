// RigidBodyManager.h (2026)
// Phase 4 of the rigid-body suite: orchestration on top of Phase 2's SoA
// device storage (DeviceRigidBody/DeviceRigidBodyTypes) and Phase 3's
// type-level force-pair list (RigidBodyForcePairList).
#pragma once
#include "ARBDException.h"
#include "Backend/Events.h"
#include "Backend/Kernels.h"
#include "Backend/KernelConfig.h"
#include "Backend/Resource.h"
#include "Objects/DeviceRigidBodyManager.h"
#include "Objects/RigidBodyForcePairs.h"
#include "PatchOperation/Integrator/RBDLM.h"
#include "System/PeriodicBox.h"
#include <memory>
#include <vector>

namespace ARBD {

/**
 * @brief Owns all rigid-body device state and drives its per-step physics.
 *
 * Global and non-spatially-decomposed - not a Patch subtype (architecture
 * decision #2): force-pairing is by grid-key match, not proximity, and an
 * RB's density grid sometimes spans multiple patches. Lives beside
 * PatchManager rather than inside it.
 *
 * Takes a resource vector + compute_resource_idx even though only
 * resources.size() == 1 is exercised today (architecture decision #3) - the
 * grid-grid math itself stays centralized on compute_resource(); a second
 * resource would only need position/orientation broadcast for grid-particle
 * forces, via broadcast_state_to_resources().
 *
 * Batched grid-grid dispatch (Phase 4.1), particle-RB grids (4.3), and
 * attached-particle position sync (needs Phase 5's config-parsed per-particle
 * body-frame offsets, which don't exist yet) are deliberately not part of
 * this skeleton.
 */
class RigidBodyManager {
  public:
	RigidBodyManager(std::vector<Resource> resources, size_t compute_resource_idx = 0)
		: resources_(std::move(resources)), compute_resource_idx_(compute_resource_idx) {
		if (resources_.empty()) {
			throw_value_error("RigidBodyManager: at least one Resource is required");
		}
		if (compute_resource_idx_ >= resources_.size()) {
			throw_value_error("RigidBodyManager: compute_resource_idx %zu out of range for %zu "
							   "resource(s)",
							   compute_resource_idx_,
							   resources_.size());
		}
	}

	/**
	 * @brief (Re)build device type/instance data and the Phase 3 force-pair
	 *        list from host-side state.
	 * @param types Rigid body types (grid ids/keys already resolved)
	 * @param host_rigid_bodies Initial per-instance state (SoA)
	 * @param grid_format grid_id -> GridFormat lookup, e.g.
	 *        `[&](int id){ return grid_manager.get_grid_format(id); }`
	 * @param grid_grid_update_period legacy: rigidBodyGridGridPeriod
	 */
	void initialize(const std::vector<RigidBodyType>& types,
					const HostRigidBodyData& host_rigid_bodies,
					const std::function<GridFormat(int)>& grid_format,
					int grid_grid_update_period = 1) {
		const idx_t count = static_cast<idx_t>(host_rigid_bodies.size());
		types_ = std::make_unique<DeviceRigidBodyTypes>(types, compute_resource());
		bodies_ = std::make_unique<DeviceRigidBody>(count, compute_resource());
		bodies_->copy_from_host(host_rigid_bodies, count);
		force_pairs_.build(types, grid_format, grid_grid_update_period);
	}

	/**
	 * @brief Port of legacy RigidBody::addLangevin, batched across all RBs.
	 */
	Event add_langevin_forces(float dt, float kT, uint64_t base_seed, size_t step) {
		ensure_initialized();
		const idx_t n = bodies_->size();
		KernelConfig config = KernelConfig::for_1d(n, compute_resource());
		RBAddLangevinKernel<float> kernel(
			bodies_->view(), types_->view(), dt, kT, n, base_seed, step);
		return launch_kernel(compute_resource(), config, kernel);
	}

	/**
	 * @brief Port of legacy RigidBody::integrateDLM, batched across all RBs.
	 *
	 * Three substeps (half-kick, drift+rotate, half-kick) launched back to
	 * back on the same stream - stream ordering makes each substep see the
	 * previous one's writes with no explicit event/sync between them.
	 */
	Event integrate_motion(float dt, const PeriodicBox& sim_box) {
		ensure_initialized();
		const idx_t n = bodies_->size();
		KernelConfig config = KernelConfig::for_1d(n, compute_resource());
		Event evt;
		for (int substep = 0; substep < 3; ++substep) {
			RBIntegrateDLMKernel kernel(bodies_->view(), types_->view(), sim_box, dt, n, substep);
			evt = launch_kernel(compute_resource(), config, kernel);
		}
		return evt;
	}

	/**
	 * @brief Broadcast RB position/orientation to non-compute resources.
	 *
	 * No-op while resources_.size() == 1 (architecture decision #3) - real
	 * work lands once a second resource needs RB state for grid-particle
	 * forces (Phase 4.3).
	 */
	void broadcast_state_to_resources() {}

	DeviceRigidBody& bodies() {
		ensure_initialized();
		return *bodies_;
	}
	const DeviceRigidBody& bodies() const {
		ensure_initialized();
		return *bodies_;
	}
	DeviceRigidBodyTypes& types() {
		ensure_initialized();
		return *types_;
	}
	const DeviceRigidBodyTypes& types() const {
		ensure_initialized();
		return *types_;
	}
	const RigidBodyForcePairList& force_pairs() const {
		return force_pairs_;
	}

	idx_t size() const {
		return bodies_ ? bodies_->size() : 0;
	}

	const Resource& compute_resource() const {
		return resources_[compute_resource_idx_];
	}

  private:
	void ensure_initialized() const {
		if (!bodies_ || !types_) {
			throw Exception(ExceptionType::RuntimeError,
							 SourceLocation(),
							 "RigidBodyManager: initialize() must be called before use");
		}
	}

	std::vector<Resource> resources_;
	size_t compute_resource_idx_;
	std::unique_ptr<DeviceRigidBody> bodies_;
	std::unique_ptr<DeviceRigidBodyTypes> types_;
	RigidBodyForcePairList force_pairs_;
};

} // namespace ARBD
