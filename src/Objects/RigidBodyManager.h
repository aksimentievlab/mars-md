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
#include "Interactions/Nonbonded/RigidBodyGridBatch.h"
#include "Interactions/Nonbonded/RigidBodyParticleGridBatch.h"
#include "Objects/DeviceParticle.h"
#include "Objects/DeviceRigidBodyManager.h"
#include "Objects/Grid.h"
#include "Objects/RigidBodyForcePairs.h"
#include "PatchOperation/Integrator/RBDLM.h"
#include "System/PeriodicBox.h"
#include <algorithm>
#include <memory>
#include <unordered_map>
#include <utility>
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
 * Batched grid-grid dispatch (Phase 4.1) is implemented via
 * prepare_grid_grid_dispatch()/compute_grid_grid_forces() (see
 * Interactions/Nonbonded/RigidBodyGridBatch.h). Particle-RB grids (4.3) and
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
		host_type_id_ = host_rigid_bodies.type_id;
		grid_grid_dispatch_ready_ = false;
		particle_grid_dispatch_ready_ = false;
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
	 * @brief Precompute the static instance-level candidate list and
	 *        worklist capacity for batched grid-grid dispatch (Phase 4.1).
	 *
	 * The type-level force pairs (Phase 3) are expanded here into concrete
	 * RB-instance pairs - static for the run, since RB counts/types don't
	 * change - so only the per-step distance cull needs to run on-device
	 * (RBGridCullKernel). Must be called once after initialize() and after
	 * grid_manager.build_device_arrays(); call again if either changes.
	 *
	 * Worklist capacity is set to exactly num_candidates: culling only
	 * removes candidates, never adds them, so this is an exact upper bound
	 * with no wasted allocation and (in the intended usage) no possible
	 * overflow - RBGridCullKernel's overflow flag exists as a safety net,
	 * exercised directly in RigidBodyGridBatch's own tests.
	 *
	 * @param grid_manager Supplies host-visible grid sizes (for block-count
	 *        sizing) and, later, the device grid views compute_grid_grid_forces()
	 *        reads from.
	 * @param grid_resource_idx grid_manager's resource index to read grid
	 *        sizes/views from - independent of compute_resource_idx_, since
	 *        the two managers may be indexed differently until Phase 5 wires
	 *        them through the same SimSystem resource list.
	 * @param threads_per_block Kernel B's block size; matches legacy NUMTHREADS.
	 */
	void prepare_grid_grid_dispatch(GridManager& grid_manager,
									size_t grid_resource_idx = 0,
									idx_t threads_per_block = 128) {
		ensure_initialized();
		grid_grid_threads_per_block_ = threads_per_block;

		const auto& pairs = force_pairs_.dense_pairs();
		const idx_t num_pairs = static_cast<idx_t>(pairs.size());
		device_grid_pairs_ =
			DeviceBuffer<RigidBodyGridPair>(num_pairs > 0 ? num_pairs : 1, compute_resource());
		if (!pairs.empty()) {
			device_grid_pairs_.copy_from_host(pairs.data(), pairs.size());
		}

		std::unordered_map<int, std::vector<int>> instances_by_type;
		for (size_t i = 0; i < host_type_id_.size(); ++i) {
			instances_by_type[host_type_id_[i]].push_back(static_cast<int>(i));
		}

		std::vector<int2> candidates;
		std::vector<int> candidate_pair_idx;
		idx_t total_blocks_capacity = 0;

		for (size_t p = 0; p < pairs.size(); ++p) {
			const RigidBodyGridPair& gp = pairs[p];
			const idx_t rho_size = grid_manager.get_dense_grid(gp.grid_id_rho).size();
			const idx_t blocks_per_candidate = (rho_size + threads_per_block - 1) / threads_per_block;

			const auto& type_i_instances = instances_by_type[gp.type_i];
			if (gp.is_pmf) {
				for (int rb_a : type_i_instances) {
					candidates.push_back(int2(rb_a, rb_a));
					candidate_pair_idx.push_back(static_cast<int>(p));
					total_blocks_capacity += blocks_per_candidate;
				}
				continue;
			}

			// Ordered instance pairs, both directions: rb_a's density through
			// rb_b's potential and rb_b's density through rb_a's potential are
			// generally distinct contributions (different grids), not a
			// double-count - see RigidBodyGridBatch.h's RBGridWork doc.
			const auto& type_j_instances = instances_by_type[gp.type_j];
			for (int rb_a : type_i_instances) {
				for (int rb_b : type_j_instances) {
					if (rb_a == rb_b)
						continue;
					candidates.push_back(int2(rb_a, rb_b));
					candidate_pair_idx.push_back(static_cast<int>(p));
					total_blocks_capacity += blocks_per_candidate;
				}
			}
		}

		num_candidates_ = static_cast<idx_t>(candidates.size());
		max_total_blocks_ = total_blocks_capacity;

		const idx_t capacity = num_candidates_ > 0 ? num_candidates_ : 1;
		candidate_pairs_ = DeviceBuffer<int2>(capacity, compute_resource());
		candidate_pair_idx_ = DeviceBuffer<int>(capacity, compute_resource());
		grid_grid_work_ = DeviceBuffer<RBGridWork>(capacity, compute_resource());
		grid_grid_work_count_ = DeviceBuffer<unsigned int>(1, compute_resource());
		grid_grid_total_blocks_ = DeviceBuffer<unsigned int>(1, compute_resource());
		grid_grid_overflow_ = DeviceBuffer<unsigned int>(1, compute_resource());
		if (!candidates.empty()) {
			candidate_pairs_.copy_from_host(candidates.data(), candidates.size());
			candidate_pair_idx_.copy_from_host(candidate_pair_idx.data(), candidate_pair_idx.size());
		}

		grid_grid_dispatch_ready_ = true;
	}

	/**
	 * @brief Batched RB-RB / RB-PMF grid-grid forces+torques (Phase 4.1).
	 *
	 * Three kernels back to back on one stream (cull -> prefix sum ->
	 * batched force), no host sync between them: Kernel B's launch grid is
	 * sized to the worklist *capacity* (an exact, host-known upper bound -
	 * see prepare_grid_grid_dispatch), with blocks beyond the real,
	 * device-computed total_blocks doing a cheap early-return no-op, so
	 * total_blocks never needs to make a D2H trip back to the host.
	 *
	 * @param grid_manager Must already have build_device_arrays() called.
	 * @param grid_resource_idx Must match what prepare_grid_grid_dispatch() used.
	 * @param step Current simulation step, for per-pair update_period gating.
	 * @param cutoff RB-RB distance cutoff for the broad-phase cull (not
	 *        applied to type-PMF terms, which always evaluate - an external
	 *        field has no "distance" to the body it acts on).
	 * @param scheme Interpolation order (1=Linear, 3=Cubic; InterpolationOrder).
	 */
	Event compute_grid_grid_forces(const GridManager& grid_manager,
								   size_t grid_resource_idx,
								   size_t step,
								   float cutoff,
								   int scheme = 1) {
		ensure_initialized();
		if (!grid_grid_dispatch_ready_) {
			throw Exception(ExceptionType::RuntimeError,
							 SourceLocation(),
							 "RigidBodyManager: prepare_grid_grid_dispatch() must be called "
							 "before compute_grid_grid_forces()");
		}
		if (num_candidates_ == 0) {
			return Event(nullptr, compute_resource());
		}

		const unsigned int zero = 0;
		grid_grid_work_count_.copy_from_host(&zero, 1, true);
		grid_grid_overflow_.copy_from_host(&zero, 1, true);

		const BaseGridView<float>* grid_views = grid_manager.get_device_grid_views(grid_resource_idx).data();

		// All three kernels share the GridCompute stream (architecture
		// decision, todo.md Phase 4.2): they already saturate the GPU, so
		// splitting further buys little, and it lets this whole pipeline
		// overlap with the nonbonded/bonded path on the Compute stream.
		void* grid_stream = compute_resource().get_stream(StreamType::GridCompute);

		RBGridCullKernel cull{std::as_const(*bodies_).view(),
							 candidate_pairs_.data(),
							 candidate_pair_idx_.data(),
							 device_grid_pairs_.data(),
							 grid_views,
							 num_candidates_,
							 cutoff * cutoff,
							 step,
							 grid_grid_threads_per_block_,
							 scheme,
							 grid_grid_work_.data(),
							 grid_grid_work_count_.data(),
							 num_candidates_,
							 grid_grid_overflow_.data()};
		KernelConfig cull_config = KernelConfig::for_1d(num_candidates_, compute_resource());
		cull_config.explicit_queue = grid_stream;
		launch_kernel(compute_resource(), cull_config, cull);

		RBGridPrefixSumKernel scan{
			grid_grid_work_.data(), grid_grid_work_count_.data(), grid_grid_total_blocks_.data()};
		KernelConfig scan_config = KernelConfig::for_1d(1, compute_resource());
		scan_config.explicit_queue = grid_stream;
		launch_kernel(compute_resource(), scan_config, scan);

		RBGridBatchedForceKernel force{bodies_->view(),
									   grid_grid_work_.data(),
									   grid_grid_work_count_.data(),
									   grid_grid_total_blocks_.data(),
									   grid_views,
									   grid_grid_threads_per_block_};
		KernelConfig force_config;
		force_config.dim = 1;
		force_config.block_size = {grid_grid_threads_per_block_, 1, 1};
		force_config.grid_size = {std::max<idx_t>(max_total_blocks_, 1), 1, 1};
		force_config.problem_size = {force_config.grid_size.x * force_config.block_size.x, 1, 1};
		force_config.shared_memory = 2 * grid_grid_threads_per_block_ * sizeof(Vector3);
		force_config.explicit_queue = grid_stream;

		return launch_kernel_with_workitem(compute_resource(), force_config, force);
	}

	/**
	 * @brief Whether the last compute_grid_grid_forces() call overflowed the
	 *        worklist. Should never happen given prepare_grid_grid_dispatch()'s
	 *        exact capacity sizing; D2H sync, so call this periodically (e.g.
	 *        every N steps or at output time), not every step.
	 */
	bool grid_grid_worklist_overflowed() const {
		if (!grid_grid_dispatch_ready_) {
			return false;
		}
		unsigned int flag = 0;
		grid_grid_overflow_.copy_to_host(&flag, 1, true);
		return flag != 0;
	}

	/**
	 * @brief Precompute the static (RB instance, potential-grid) candidate
	 *        list for batched particle-RB dispatch (Phase 4.3).
	 *
	 * Unlike prepare_grid_grid_dispatch(), there's no distance-based culling
	 * here (see RigidBodyParticleGridBatch.h) and every candidate shares the
	 * same num_particles, so block counts are uniform across candidates -
	 * the whole worklist reduces to plain arithmetic, no worklist buffer or
	 * prefix sum needed.
	 *
	 * @param types Same RigidBodyType vector passed to initialize() -
	 *        re-passed here rather than cached, since RigidBodyManager has
	 *        no other need for host-side RigidBodyType data once initialize()
	 *        uploads it to device buffers.
	 * @param num_particles Total particle count sampling these grids. RBs
	 *        don't own particles (architecture decision #4 - particles stay
	 *        in Patch/DeviceParticle), so this comes from whichever Patch(es)
	 *        hold them; single-patch assumption, matching the rest of this
	 *        codebase today.
	 */
	void prepare_particle_grid_dispatch(const std::vector<RigidBodyType>& types,
										idx_t num_particles,
										idx_t threads_per_block = 128) {
		ensure_initialized();
		particle_grid_threads_per_block_ = threads_per_block;
		particle_grid_num_particles_ = num_particles;
		particle_grid_blocks_per_candidate_ =
			(num_particles + threads_per_block - 1) / threads_per_block;

		std::unordered_map<int, std::vector<int>> instances_by_type;
		for (size_t i = 0; i < host_type_id_.size(); ++i) {
			instances_by_type[host_type_id_[i]].push_back(static_cast<int>(i));
		}

		std::vector<int> cand_rb_id;
		std::vector<int> cand_grid_id;
		std::vector<float> cand_scale;
		for (size_t t = 0; t < types.size(); ++t) {
			const auto& instances = instances_by_type[static_cast<int>(t)];
			for (const GridTerm& term : types[t].potential_grids) {
				for (int rb_id : instances) {
					cand_rb_id.push_back(rb_id);
					cand_grid_id.push_back(term.grid_id);
					cand_scale.push_back(term.scale);
				}
			}
		}

		particle_grid_num_candidates_ = static_cast<idx_t>(cand_rb_id.size());

		const idx_t capacity = particle_grid_num_candidates_ > 0 ? particle_grid_num_candidates_ : 1;
		particle_grid_candidate_rb_id_ = DeviceBuffer<int>(capacity, compute_resource());
		particle_grid_candidate_grid_id_ = DeviceBuffer<int>(capacity, compute_resource());
		particle_grid_candidate_scale_ = DeviceBuffer<float>(capacity, compute_resource());
		particle_grid_work_ = DeviceBuffer<RBParticleGridWork>(capacity, compute_resource());
		if (!cand_rb_id.empty()) {
			particle_grid_candidate_rb_id_.copy_from_host(cand_rb_id.data(), cand_rb_id.size(), true);
			particle_grid_candidate_grid_id_.copy_from_host(
				cand_grid_id.data(), cand_grid_id.size(), true);
			particle_grid_candidate_scale_.copy_from_host(cand_scale.data(), cand_scale.size(), true);
		}

		particle_grid_dispatch_ready_ = true;
	}

	/**
	 * @brief Batched particle-RB grid forces (Phase 4.3): rebuild per-candidate
	 *        transforms, then the batched force kernel - two kernels back to
	 *        back on the GridCompute stream, no host sync between them.
	 *
	 * @param grid_manager Must already have build_device_arrays() called.
	 * @param grid_resource_idx Must match what prepare_particle_grid_dispatch()
	 *        (indirectly, via the types passed to it) assumes.
	 * @param particles Particle view from whichever Patch owns them; this
	 *        kernel atomically accumulates into its ForceEnergy buffer, so it
	 *        must not be cleared or read again until the returned Event
	 *        completes (see todo.md Phase 4.2's correctness constraint).
	 * @param scheme Interpolation order (1=Linear, 3=Cubic; InterpolationOrder).
	 */
	Event compute_particle_rb_forces(const GridManager& grid_manager,
									 size_t grid_resource_idx,
									 ParticleView particles,
									 int scheme = 1) {
		ensure_initialized();
		if (!particle_grid_dispatch_ready_) {
			throw Exception(ExceptionType::RuntimeError,
							 SourceLocation(),
							 "RigidBodyManager: prepare_particle_grid_dispatch() must be called "
							 "before compute_particle_rb_forces()");
		}
		if (particle_grid_num_candidates_ == 0 || particle_grid_blocks_per_candidate_ == 0) {
			return Event(nullptr, compute_resource());
		}

		const BaseGridView<float>* grid_views =
			grid_manager.get_device_grid_views(grid_resource_idx).data();
		void* grid_stream = compute_resource().get_stream(StreamType::GridCompute);

		RBParticleGridBuildKernel build{std::as_const(*bodies_).view(),
									   particle_grid_candidate_rb_id_.data(),
									   particle_grid_candidate_grid_id_.data(),
									   particle_grid_candidate_scale_.data(),
									   particle_grid_num_candidates_,
									   grid_views,
									   scheme,
									   particle_grid_work_.data()};
		KernelConfig build_config =
			KernelConfig::for_1d(particle_grid_num_candidates_, compute_resource());
		build_config.explicit_queue = grid_stream;
		launch_kernel(compute_resource(), build_config, build);

		RBParticleGridForceKernel force{bodies_->view(),
										particles,
										particle_grid_work_.data(),
										grid_views,
										particle_grid_num_particles_,
										particle_grid_blocks_per_candidate_,
										particle_grid_threads_per_block_};
		const idx_t total_blocks = particle_grid_num_candidates_ * particle_grid_blocks_per_candidate_;
		KernelConfig force_config;
		force_config.dim = 1;
		force_config.block_size = {particle_grid_threads_per_block_, 1, 1};
		force_config.grid_size = {total_blocks, 1, 1};
		force_config.problem_size = {total_blocks * particle_grid_threads_per_block_, 1, 1};
		force_config.shared_memory = 2 * particle_grid_threads_per_block_ * sizeof(Vector3);
		force_config.explicit_queue = grid_stream;

		return launch_kernel_with_workitem(compute_resource(), force_config, force);
	}

	/**
	 * @brief Broadcast RB position/orientation to non-compute resources.
	 *
	 * No-op while resources_.size() == 1 (architecture decision #3) - real
	 * work lands once a second resource needs RB state for grid-particle
	 * forces.
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
	std::vector<int> host_type_id_; // per-RB-instance type id, for candidate expansion

	// Phase 4.1 batched grid-grid dispatch state (see prepare_grid_grid_dispatch).
	bool grid_grid_dispatch_ready_{false};
	idx_t grid_grid_threads_per_block_{128};
	idx_t num_candidates_{0};
	idx_t max_total_blocks_{0};
	DeviceBuffer<RigidBodyGridPair> device_grid_pairs_;
	DeviceBuffer<int2> candidate_pairs_;
	DeviceBuffer<int> candidate_pair_idx_;
	DeviceBuffer<RBGridWork> grid_grid_work_;
	DeviceBuffer<unsigned int> grid_grid_work_count_;
	DeviceBuffer<unsigned int> grid_grid_total_blocks_;
	DeviceBuffer<unsigned int> grid_grid_overflow_;

	// Phase 4.3 batched particle-RB dispatch state (see prepare_particle_grid_dispatch).
	bool particle_grid_dispatch_ready_{false};
	idx_t particle_grid_threads_per_block_{128};
	idx_t particle_grid_num_particles_{0};
	idx_t particle_grid_blocks_per_candidate_{0};
	idx_t particle_grid_num_candidates_{0};
	DeviceBuffer<int> particle_grid_candidate_rb_id_;
	DeviceBuffer<int> particle_grid_candidate_grid_id_;
	DeviceBuffer<float> particle_grid_candidate_scale_;
	DeviceBuffer<RBParticleGridWork> particle_grid_work_;
};

} // namespace ARBD
