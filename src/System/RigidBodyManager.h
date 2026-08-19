// RigidBodyManager.h (2026)
// Phase 4 of the rigid-body suite: orchestration on top of Phase 2's SoA
// device storage (DeviceRigidBody/DeviceRigidBodyTypes) and Phase 3's
// type-level force-pair list (RigidBodyForcePairList).
#pragma once
#include "ARBDException.h"
#include "Backend/Events.h"
#include "Backend/KernelConfig.h"
#include "Backend/Kernels.h"
#include "Backend/Resource.h"
#include "Interactions/Nonbonded/RigidBodyAttachedParticles.h"
#include "Interactions/Nonbonded/RigidBodyGridBatch.h"
#include "Interactions/Nonbonded/RigidBodyParticleGridBatch.h"
#include "Objects/DeviceParticle.h"
#include "Objects/DeviceRigidBodyManager.h"
#include "Objects/Grid.h"
#include "Objects/RigidBodyCosmeticsKernel.h"
#include "Objects/RigidBodyForcePairs.h"
#include "PatchOperation/Integrator/RBBD.h"
#include "PatchOperation/Integrator/RBDLM.h"
#include "RBOperation/RBHostFTManager.h"
#include "System/PeriodicBox.h"
#include <algorithm>
#include <memory>
#include <span>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ARBD {

/**
 * @brief Owns all rigid-body device state and drives its per-step physics.
 * @see RigidBodyManager.md
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
		RBLangevinForceKernel<float>
			kernel(bodies_->view(), types_->view(), dt, kT, n, base_seed, step);
		return launch_kernel(compute_resource(), config, kernel);
	}

	/**
	 * @brief DLM half-kick then drift (substeps 0-1). Run before the force phase.
	 * @param dt Timestep.
	 * @param sim_box Boundary conditions.
	 * @see dev_notes.md
	 */
	Event integrate_drift(float dt, const PeriodicBox& sim_box) {
		return integrate_motion(dt, sim_box, 0, 1);
	}

	/**
	 * @brief DLM trailing half-kick (substep 2). Run after the force phase.
	 * @param dt Timestep.
	 * @param sim_box Boundary conditions.
	 */
	Event integrate_kick(float dt, const PeriodicBox& sim_box) {
		return integrate_motion(dt, sim_box, 2, 2);
	}

	/**
	 * @brief Port of legacy RigidBody::integrateDLM, batched across all RBs.
	 * @param dt Timestep.
	 * @param sim_box Boundary conditions.
	 * @param first First substep to run, inclusive: 0/2 half-kick, 1 drift.
	 * @param last Last substep to run, inclusive.
	 * @see dev_notes.md - running 0-2 in one call is not velocity Verlet.
	 */
	Event integrate_motion(float dt, const PeriodicBox& sim_box, int first = 0, int last = 2) {
		ensure_initialized();
		const idx_t n = bodies_->size();
		KernelConfig config = KernelConfig::for_1d(n, compute_resource());
		Event evt;
		for (int substep = first; substep <= last; ++substep) {
			RBIntegrateDLMKernel kernel(bodies_->view(), types_->view(), sim_box, dt, n, substep);
			evt = launch_kernel(compute_resource(), config, kernel);
		}
		return evt;
	}

	/**
	 * @brief Setup-time upload of per-instance constantForce/constantTorque.
	 * @param force Per-body constant force, indexed by device SoA slot.
	 * @param torque Per-body constant torque, same indexing.
	 * @see RBHostFTManager.md
	 */
	void set_external_loads(std::span<const Vector3> force, std::span<const Vector3> torque) {
		ensure_initialized();
		if (bodies_->size() == 0) {
			return;
		}
		host_forces().set_baseline(force, torque);
		host_forces().push_baseline(bodies_->view()).wait();
	}

	/**
	 * @brief The single path for host-computed external force/torque.
	 * @details Holds the constantForce/constantTorque baseline, so a dynamic
	 *          producer must use push_with_baseline() or it will assign over it.
	 */
	RBHostFTManager& host_forces() {
		ensure_initialized();
		if (!host_forces_) {
			host_forces_ = std::make_unique<RBHostFTManager>(bodies_->size(), compute_resource());
		}
		return *host_forces_;
	}

	/**
	 * @brief Port of legacy RigidBody::integrate - overdamped Brownian, one
	 *        full step, no momentum.
	 */
	Event integrate_brownian(float dt,
							 float kT,
							 uint64_t base_seed,
							 size_t step,
							 const PeriodicBox& sim_box) {
		ensure_initialized();
		const idx_t n = bodies_->size();
		KernelConfig config = KernelConfig::for_1d(n, compute_resource());
		RBIntegrateBDKernel<float>
			kernel(bodies_->view(), types_->view(), sim_box, dt, kT, n, base_seed, step);
		return launch_kernel(compute_resource(), config, kernel);
	}

	/**
	 * @brief Precompute the instance-level candidate list and worklist capacity
	 *        for batched grid-grid dispatch (Phase 4.1).
	 *
	 * Call once after initialize() and grid_manager.build_device_arrays();
	 * call again if either changes. See dev_notes.md.
	 *
	 * @param grid_manager Supplies grid sizes and device grid views.
	 * @param grid_resource_idx grid_manager's resource index, independent of
	 *        compute_resource_idx_.
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
			const idx_t blocks_per_candidate =
				(rho_size + threads_per_block - 1) / threads_per_block;

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
			candidate_pair_idx_.copy_from_host(candidate_pair_idx.data(),
											   candidate_pair_idx.size());
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

		const BaseGridView<arbd_real>* grid_views =
			grid_manager.get_device_grid_views(grid_resource_idx).data();

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

		RBGridPrefixSumKernel scan{grid_grid_work_.data(),
								   grid_grid_work_count_.data(),
								   grid_grid_total_blocks_.data()};
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
	 *        list for batched particle-RB dispatch (Phase 4.3)
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

		const idx_t capacity =
			particle_grid_num_candidates_ > 0 ? particle_grid_num_candidates_ : 1;
		particle_grid_candidate_rb_id_ = DeviceBuffer<int>(capacity, compute_resource());
		particle_grid_candidate_grid_id_ = DeviceBuffer<int>(capacity, compute_resource());
		particle_grid_candidate_scale_ = DeviceBuffer<arbd_real>(capacity, compute_resource());
		particle_grid_work_ = DeviceBuffer<RBParticleGridWork>(capacity, compute_resource());
		if (!cand_rb_id.empty()) {
			particle_grid_candidate_rb_id_.copy_from_host(cand_rb_id.data(),
														  cand_rb_id.size(),
														  true);
			particle_grid_candidate_grid_id_.copy_from_host(cand_grid_id.data(),
															cand_grid_id.size(),
															true);
			particle_grid_candidate_scale_.copy_from_host(cand_scale.data(),
														  cand_scale.size(),
														  true);
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

		const BaseGridView<arbd_real>* grid_views =
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
		const idx_t total_blocks =
			particle_grid_num_candidates_ * particle_grid_blocks_per_candidate_;
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
	 * @brief Build the static attached-particle table (see
	 *        Interactions/Nonbonded/RigidBodyAttachedParticles.h).
	 *
	 * Attached particles are ordinary particles that happen to be rigidly
	 * slaved to a body: they sit in the patch's particle array and take part in
	 * every force path unchanged. All this table records is which particle
	 * belongs to which body and where on that body it sits - static for the run
	 * (bond breaking/formation is a separate, future feature), so it is built
	 * once here rather than rebuilt per step.
	 *
	 * @param types Rigid body types, for each type's attached-particle template
	 * @param bodies Per-instance state carrying the attached_start/attached_count
	 *        ranges ConfigParser assigned during its fold-in pass
	 * @param num_patches Patch count, purely to reject the multi-patch case:
	 *        particle_index entries are indices into one patch's arrays. This is
	 *        the same single-patch assumption compute_particle_rb_forces()
	 *        already makes. Generalizing means promoting attached_rigid_body_id
	 *        and the body-frame offset into DeviceParticle/ParticleView so each
	 *        patch can sync and reduce its own share.
	 */
	void prepare_attached_particles(const std::vector<RigidBodyType>& types,
									const std::vector<RigidBodyIO>& bodies,
									size_t num_patches = 1,
									idx_t threads_per_block = 128) {
		ensure_initialized();
		attached_threads_per_block_ = threads_per_block;

		std::vector<RBAttachedParticle> attached;
		std::vector<int> range_start;
		std::vector<int> range_count;
		std::vector<int> block_rb_id;

		for (const RigidBodyIO& rb : bodies) {
			if (rb.attached_count <= 0) {
				continue;
			}
			if (rb.type_id < 0 || static_cast<size_t>(rb.type_id) >= types.size()) {
				throw_value_error("RigidBodyManager: rigid body %d has attached particles but an "
								  "out-of-range type_id %d",
								  rb.id,
								  rb.type_id);
			}
			const auto& templ = types[rb.type_id].attached_particle;
			if (static_cast<int>(templ.size()) != rb.attached_count) {
				throw_value_error("RigidBodyManager: rigid body %d claims %d attached particle(s) "
								  "but its type declares %zu",
								  rb.id,
								  rb.attached_count,
								  templ.size());
			}

			range_start.push_back(static_cast<int>(attached.size()));
			range_count.push_back(rb.attached_count);
			block_rb_id.push_back(rb.id);
			for (int k = 0; k < rb.attached_count; ++k) {
				RBAttachedParticle a{};
				a.body_offset = templ[k].position; // body-frame, from the PDB template
				a.particle_index = rb.attached_start + k;
				a.rb_id = rb.id;
				attached.push_back(a);
			}
		}

		num_attached_ = static_cast<idx_t>(attached.size());
		num_attached_blocks_ = static_cast<idx_t>(block_rb_id.size());
		if (num_attached_ == 0) {
			return;
		}

		if (num_patches != 1) {
			throw Exception(ExceptionType::NotImplementedError,
							SourceLocation(),
							"RigidBodyManager: rigid-body attached particles currently require a "
							"single patch (got %zu) - their particle indices are patch-local",
							num_patches);
		}

		attached_ = DeviceBuffer<RBAttachedParticle>(num_attached_, compute_resource());
		attached_.copy_from_host(attached.data(), attached.size());
		attached_range_start_ = DeviceBuffer<int>(num_attached_blocks_, compute_resource());
		attached_range_start_.copy_from_host(range_start.data(), range_start.size());
		attached_range_count_ = DeviceBuffer<int>(num_attached_blocks_, compute_resource());
		attached_range_count_.copy_from_host(range_count.data(), range_count.size());
		attached_block_rb_id_ = DeviceBuffer<int>(num_attached_blocks_, compute_resource());
		attached_block_rb_id_.copy_from_host(block_rb_id.data(), block_rb_id.size());
	}

	/** @brief Whether any rigid body has attached particles. */
	bool has_attached_particles() const {
		return num_attached_ > 0;
	}

	/**
	 * @brief Build the static table of visualization-only template atoms.
	 *
	 * The complement of prepare_attached_particles(): template atoms whose
	 * CosmeticParticle::attached_particle_index is negative, i.e. those that
	 * never became real particles. They carry no physics, so all that is stored
	 * is where they sit on their body.
	 *
	 * Ordered instance-major, then template order - the order they occupy in the
	 * trajectory, and the order a PSF describing that trajectory must use.
	 */
	void prepare_cosmetic_atoms(const std::vector<RigidBodyType>& types,
								const std::vector<RigidBodyIO>& bodies) {
		ensure_initialized();

		std::vector<Vector3> body_offset;
		std::vector<int> rb_id;
		for (const RigidBodyIO& rb : bodies) {
			if (rb.type_id < 0 || static_cast<size_t>(rb.type_id) >= types.size()) {
				continue;
			}
			for (const CosmeticParticle& c : types[rb.type_id].template_particles) {
				if (c.attached_particle_index >= 0) {
					continue; // real particle; the trajectory already has it
				}
				body_offset.push_back(c.body_frame_position);
				rb_id.push_back(rb.id);
			}
		}

		num_cosmetic_ = static_cast<idx_t>(body_offset.size());
		if (num_cosmetic_ == 0) {
			return;
		}
		cosmetic_body_offset_ = DeviceBuffer<Vector3>(num_cosmetic_, compute_resource());
		cosmetic_body_offset_.copy_from_host(body_offset.data(), body_offset.size());
		cosmetic_rb_id_ = DeviceBuffer<int>(num_cosmetic_, compute_resource());
		cosmetic_rb_id_.copy_from_host(rb_id.data(), rb_id.size());
		cosmetic_positions_ = DeviceBuffer<Vector3>(num_cosmetic_, compute_resource());
	}

	/** @brief Number of visualization-only template atoms across all bodies. */
	idx_t num_cosmetic_atoms() const {
		return num_cosmetic_;
	}

	/**
	 * @brief Recompute cosmetic atom positions from current body transforms.
	 *
	 * Launched on StreamType::Optional: this is output-only work, so it may
	 * overlap the next step's physics rather than serializing behind it. Wait
	 * on the returned Event before calling copy_cosmetic_positions_to_host().
	 */
	Event compute_cosmetic_positions() {
		ensure_initialized();
		if (num_cosmetic_ == 0) {
			return Event(nullptr, compute_resource());
		}
		RBCosmeticParticleView view{cosmetic_body_offset_.data(), cosmetic_rb_id_.data()};
		RBCosmeticPositionsKernel kernel{std::as_const(*bodies_).view(),
										 view,
										 cosmetic_positions_.data(),
										 num_cosmetic_};
		KernelConfig config = KernelConfig::for_1d(num_cosmetic_, compute_resource());
		config.explicit_queue = compute_resource().get_stream(StreamType::Optional);
		return launch_kernel(compute_resource(), config, kernel);
	}

	/** @brief Drain the last compute_cosmetic_positions() result to the host. */
	void copy_cosmetic_positions_to_host(std::vector<Vector3>& out) const {
		if (num_cosmetic_ == 0) {
			return;
		}
		const size_t offset = out.size();
		out.resize(offset + static_cast<size_t>(num_cosmetic_));
		cosmetic_positions_.copy_to_host(out.data() + offset,
										 static_cast<size_t>(num_cosmetic_),
										 true);
	}

	/**
	 * @brief Rewrite attached-particle positions from their parent bodies.
	 *
	 * Must run before the step's force calculation: the pairlist and every
	 * force kernel read these positions.
	 */
	Event sync_attached_particle_positions(ParticleView particles) {
		ensure_initialized();
		if (num_attached_ == 0) {
			return Event(nullptr, compute_resource());
		}
		RBSyncAttachedPositionsKernel kernel{std::as_const(*bodies_).view(),
											 particles,
											 attached_.data(),
											 num_attached_};
		KernelConfig config = KernelConfig::for_1d(num_attached_, compute_resource());
		return launch_kernel(compute_resource(), config, kernel);
	}

	/**
	 * @brief Reduce attached-particle forces into their parent bodies' net
	 *        force and torque.
	 *
	 * Must run after all particle forces are complete (nonbonded *and* bonded)
	 * and before the rigid-body integration that consumes force/torque.
	 */
	Event reduce_attached_particle_forces(ConstParticleView particles) {
		ensure_initialized();
		if (num_attached_ == 0) {
			return Event(nullptr, compute_resource());
		}
		RBReduceAttachedForcesKernel kernel{bodies_->view(),
											particles,
											attached_.data(),
											attached_range_start_.data(),
											attached_range_count_.data(),
											attached_block_rb_id_.data(),
											attached_threads_per_block_};
		KernelConfig config;
		config.dim = 1;
		config.block_size = {attached_threads_per_block_, 1, 1};
		config.grid_size = {num_attached_blocks_, 1, 1};
		config.problem_size = {num_attached_blocks_ * attached_threads_per_block_, 1, 1};
		config.shared_memory = 2 * attached_threads_per_block_ * sizeof(Vector3);
		return launch_kernel_with_workitem(compute_resource(), config, kernel);
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

	/**
	 * @brief Refresh a host SoA from the device instance state.
	 * @param host Destination, grown by DeviceRigidBody::copy_to_host if short.
	 */
	void gather_to_host(HostRigidBodyData& host) const {
		const idx_t count = size();
		if (count == 0) {
			return;
		}
		bodies_->copy_to_host(host, count);
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
	// Lazily built: its staging capacity needs bodies_->size().
	std::unique_ptr<RBHostFTManager> host_forces_;
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

	// Attached-particle state (see prepare_attached_particles). Static for the
	// run: which particle belongs to which body never changes.
	idx_t attached_threads_per_block_{128};
	idx_t num_attached_{0};		   ///< total attached particles across all bodies
	idx_t num_attached_blocks_{0}; ///< == number of bodies that have any
	DeviceBuffer<RBAttachedParticle> attached_;
	DeviceBuffer<int> attached_range_start_;
	DeviceBuffer<int> attached_range_count_;
	DeviceBuffer<int> attached_block_rb_id_;

	// Visualization-only template atoms (see prepare_cosmetic_atoms).
	idx_t num_cosmetic_{0};
	DeviceBuffer<Vector3> cosmetic_body_offset_;
	DeviceBuffer<int> cosmetic_rb_id_;
	DeviceBuffer<Vector3> cosmetic_positions_;

	// Phase 4.3 batched particle-RB dispatch state (see prepare_particle_grid_dispatch).
	bool particle_grid_dispatch_ready_{false};
	idx_t particle_grid_threads_per_block_{128};
	idx_t particle_grid_num_particles_{0};
	idx_t particle_grid_blocks_per_candidate_{0};
	idx_t particle_grid_num_candidates_{0};
	DeviceBuffer<int> particle_grid_candidate_rb_id_;
	DeviceBuffer<int> particle_grid_candidate_grid_id_;
	DeviceBuffer<arbd_real> particle_grid_candidate_scale_;
	DeviceBuffer<RBParticleGridWork> particle_grid_work_;
};

} // namespace ARBD
