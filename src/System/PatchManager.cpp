// src/System/PatchManager.cpp

#include "System/PatchManager.h"
#include "ARBDLogger.h"
#include "Objects/DeviceParticleManager.h"
#include "System/SimSystem.h"
#include "System/SystemState.h"
#include <algorithm>
#include <cmath>

namespace ARBD {

//================================================================================
// Constructor
//================================================================================

PatchManager::PatchManager(SimSystem& system)
#ifdef USE_MPI
	: sim_system_(system), mpi_manager_(MPI::Manager::instance())
#else
	: sim_system_(system)
#endif
{
}

//================================================================================
// Initialization and Setup
//================================================================================

void PatchManager::initialize_from_plan(const DecompositionPlan& plan,
										idx_t estimated_particles_per_patch) {
	if (!plan.is_valid()) {
		throw Exception(ExceptionType::ValueError,
						SourceLocation(),
						"PatchManager::initialize_from_plan received invalid decomposition plan");
	}

	const Length cutoff = sim_system_.get_cutoff();

#ifdef USE_MPI
	if (!mpi_manager_.is_initialized()) {
		throw Exception(ExceptionType::RuntimeError,
						SourceLocation(),
						"PatchManager requires initialized MPI::Manager");
	}

	const int mpi_size = mpi_manager_.get_size();
	const size_t total_patches = plan.total_patches();
	if (static_cast<size_t>(mpi_size) != total_patches) {
		throw Exception(ExceptionType::ValueError,
						SourceLocation(),
						"PatchManager: MPI size ({}) must match total patches ({})",
						mpi_size,
						total_patches);
	}
#endif

	// Configure spatial grid from plan
	setup_spatial_grid(plan);

	const size_t num_patches = plan.total_patches();
	patches_.clear();
	patch_metadata_.clear();
	neighbor_comm_.clear();

	patches_.reserve(num_patches);
	patch_metadata_.reserve(num_patches);

	for (size_t pid = 0; pid < num_patches; ++pid) {
		const Resource& res = plan.patch_resources[pid];
		const Vector3& min_b = plan.patch_min_bounds[pid];
		const Vector3& max_b = plan.patch_max_bounds[pid];

		patch_metadata_.emplace_back(static_cast<patch_t>(pid), res);
		auto& meta = patch_metadata_.back();
		meta.capacity = estimated_particles_per_patch;
		meta.min_bounds = min_b;
		meta.max_bounds = max_b;

		// Grid coordinates derived from patch id assuming row-major ordering
		meta.grid_coords = patch_id_to_grid_coords(static_cast<patch_t>(pid));

#ifdef USE_MPI
		meta.mpi_rank = static_cast<int>(pid);
		if (mpi_manager_.get_rank() == meta.mpi_rank) {
			local_patch_id_ = static_cast<patch_t>(pid);
		}
#endif

		// Create patch instance
		auto patch =
			std::make_unique<Patch>(static_cast<patch_t>(pid), estimated_particles_per_patch, res);
		patch->set_bounds(min_b, max_b);
		patch->set_periodic_box(&sim_system_.get_boundary_conditions());
		patch->set_halo_thickness(static_cast<float>(cutoff));

		// Create and set device particle types for this patch
		const auto& particle_types = sim_system_.get_particle_types();
		if (!particle_types.empty()) {
			auto device_types = std::make_unique<DeviceParticleTypes>(particle_types, res);
			device_types->copy_from_host(particle_types);
			patch->set_particle_types(std::move(device_types));
		}

		patches_.push_back(std::move(patch));
	}

	// Discover neighbors and allocate communication buffers
	discover_neighbors();
	initialize_communication_buffers();

	LOGINFO("PatchManager: Initialized from plan with {} patches", num_patches);
}

void PatchManager::initialize_local_patches(idx_t estimated_particles, const Length& cutoff) {
	// Single-patch helper used by SimSystem::create_single_patch_manager
	patches_.clear();
	patch_metadata_.clear();
	neighbor_comm_.clear();

	const auto& resources = sim_system_.get_resources();
	if (resources.empty()) {
		throw Exception(ExceptionType::RuntimeError,
						SourceLocation(),
						"PatchManager::initialize_local_patches called with no resources");
	}

	const Resource& res = resources.front();

	// Use full simulation box as single patch bounds
	const Vector3 box = sim_system_.get_box_size();
	Vector3 min_bounds(0.0f, 0.0f, 0.0f);
	Vector3 max_bounds(box.x, box.y, box.z);

	patch_metadata_.emplace_back(0, res);
	auto& meta = patch_metadata_.back();
	meta.capacity = estimated_particles;
	meta.min_bounds = min_bounds;
	meta.max_bounds = max_bounds;
	meta.grid_coords = {0, 0, 0};
	meta.mpi_rank = 0;

	auto patch = std::make_unique<Patch>(0, estimated_particles, res);
	patch->set_bounds(min_bounds, max_bounds);
	patch->set_periodic_box(&sim_system_.get_boundary_conditions());
	patch->set_halo_thickness(static_cast<float>(cutoff));

	// Create and set device particle types for this patch
	const auto& particle_types = sim_system_.get_particle_types();
	if (!particle_types.empty()) {
		auto device_types = std::make_unique<DeviceParticleTypes>(particle_types, res);
		device_types->copy_from_host(particle_types);
		patch->set_particle_types(std::move(device_types));
	}

	patches_.push_back(std::move(patch));

	// Configure trivial grid
	setup_spatial_grid(1, 1, 1, true, true, true);
	discover_neighbors();
	initialize_communication_buffers();

	LOGINFO("PatchManager: Initialized single local patch with capacity {}", estimated_particles);
}

void PatchManager::setup_spatial_grid(int nx, int ny, int nz, bool px, bool py, bool pz) {
	grid_info_.dimensions = {nx, ny, nz};
	grid_info_.periodic = {px, py, pz};

	// Derive grid spacing from simulation box (if available)
	Vector3 box = sim_system_.get_box_size();
	grid_info_.grid_spacing = Vector3(box.x / static_cast<float>(nx),
									  box.y / static_cast<float>(ny),
									  box.z / static_cast<float>(nz));
	grid_info_.system_origin = Vector3(0.0f, 0.0f, 0.0f);

	LOGINFO("PatchManager: Setup spatial grid {}x{}x{} with periodicity [{},{},{}]",
			nx,
			ny,
			nz,
			px,
			py,
			pz);
}

void PatchManager::distribute_particles_from_state(SystemState& state) {
	const HostParticleData& global_particles = state.get_global_particles();

	if (patches_.empty()) {
		LOGWARN("PatchManager: No patches available for particle distribution");
		return;
	}

	// Single-patch optimization
	if (patches_.size() == 1) {
		auto& patch = *patches_.front();
		patch.copy_particles_from_host(global_particles, 0, global_particles.size());
		patch.set_particle_count(static_cast<idx_t>(global_particles.size()));

		auto& meta = patch_metadata_.front();
		meta.particle_count = patch.get_particle_count();

		LOGINFO("PatchManager: Distributed {} particles to single patch", global_particles.size());
		return;
	}

	// Simple multi-patch distribution based on spatial bounds
	std::vector<idx_t> offset_per_patch(patches_.size(), 0);
	for (size_t i = 0; i < patches_.size(); ++i) {
		offset_per_patch[i] = 0;
	}

	// First pass: count particles per patch
	std::vector<idx_t> counts(patches_.size(), 0);
	for (idx_t i = 0; i < static_cast<idx_t>(global_particles.size()); ++i) {
		const Vector3& pos = global_particles.pos[i];
		const patch_t pid = find_patch_for_position(pos);
		if (pid >= 0 && static_cast<size_t>(pid) < patches_.size()) {
			++counts[static_cast<size_t>(pid)];
		}
	}

	// Second pass: copy particles per patch (contiguous ranges in a scratch buffer
	// would be better, but for now we do a simple per-patch copy)
	for (size_t pid = 0; pid < patches_.size(); ++pid) {
		auto& patch = *patches_[pid];
		patch.set_particle_count(0);
		patch_metadata_[pid].particle_count = 0;
	}

	for (idx_t i = 0; i < static_cast<idx_t>(global_particles.size()); ++i) {
		const Vector3& pos = global_particles.pos[i];
		const patch_t pid = find_patch_for_position(pos);
		if (pid < 0 || static_cast<size_t>(pid) >= patches_.size())
			continue;

		auto& patch = *patches_[static_cast<size_t>(pid)];
		idx_t current_count = patch.get_particle_count();
		if (current_count >= patch.get_capacity()) {
			LOGWARN("PatchManager: Patch {} capacity exceeded during distribution", pid);
			continue;
		}

		patch.copy_particles_from_host(global_particles, i, 1);
		patch.set_particle_count(current_count + 1);
		patch_metadata_[static_cast<size_t>(pid)].particle_count = patch.get_particle_count();
	}

	LOGINFO("PatchManager: Distributed {} particles across {} patches",
			global_particles.size(),
			patches_.size());
}

void PatchManager::gather_particles_to_state(SystemState& state) {
	state.clear_global_arrays();

	if (patches_.empty()) {
		LOGWARN("PatchManager: No patches available for particle gathering");
		return;
	}

	HostParticleData all_particles;

	// Pre-compute total particle count
	size_t total = 0;
	for (const auto& meta : patch_metadata_) {
		total += static_cast<size_t>(meta.particle_count);
	}
	all_particles.reserve(total);

	for (size_t pid = 0; pid < patches_.size(); ++pid) {
		const auto& patch = *patches_[pid];
		const idx_t count = patch.get_particle_count();
		if (count == 0)
			continue;

		HostParticleData tmp;
		tmp.reserve(count);
		patch.copy_particles_to_host(tmp, 0, count);
		all_particles.reserve(all_particles.size() + count);

		for (idx_t i = 0; i < count; ++i) {
			all_particles.push_back(tmp, i);
		}
	}

	state.set_from_host_particle_data(all_particles);
	state.mark_synced();

	LOGINFO("PatchManager: Gathered {} particles from {} patches",
			all_particles.size(),
			patches_.size());
}

//================================================================================
// Patch Access and Management
//================================================================================

Patch& PatchManager::get_patch(patch_t patch_id) {
	if (patch_id < 0 || static_cast<size_t>(patch_id) >= patches_.size()) {
		throw Exception(ExceptionType::ValueError,
						SourceLocation(),
						"PatchManager::get_patch invalid patch id {}",
						patch_id);
	}
	return *patches_[static_cast<size_t>(patch_id)];
}

const Patch& PatchManager::get_patch(patch_t patch_id) const {
	if (patch_id < 0 || static_cast<size_t>(patch_id) >= patches_.size()) {
		throw Exception(ExceptionType::ValueError,
						SourceLocation(),
						"PatchManager::get_patch (const) invalid patch id {}",
						patch_id);
	}
	return *patches_[static_cast<size_t>(patch_id)];
}

Patch& PatchManager::get_local_patch(int rank) {
#ifdef USE_MPI
	const patch_t pid = static_cast<patch_t>(rank);
	return get_patch(pid);
#else
	(void)rank;
	return get_patch(0);
#endif
}

const Patch& PatchManager::get_local_patch(int rank) const {
#ifdef USE_MPI
	const patch_t pid = static_cast<patch_t>(rank);
	return get_patch(pid);
#else
	(void)rank;
	return get_patch(0);
#endif
}

//================================================================================
// Spatial Bounds and Queries
//================================================================================

const Vector3& PatchManager::get_local_min_bounds() const {
#ifdef USE_MPI
	const patch_t pid = local_patch_id_;
#else
	const patch_t pid = 0;
#endif
	if (patch_metadata_.empty() || pid < 0 || static_cast<size_t>(pid) >= patch_metadata_.size()) {
		static Vector3 zero(0.0f, 0.0f, 0.0f);
		return zero;
	}
	return patch_metadata_[static_cast<size_t>(pid)].min_bounds;
}

const Vector3& PatchManager::get_local_max_bounds() const {
#ifdef USE_MPI
	const patch_t pid = local_patch_id_;
#else
	const patch_t pid = 0;
#endif
	if (patch_metadata_.empty() || pid < 0 || static_cast<size_t>(pid) >= patch_metadata_.size()) {
		static Vector3 zero(0.0f, 0.0f, 0.0f);
		return zero;
	}
	return patch_metadata_[static_cast<size_t>(pid)].max_bounds;
}

Vector3 PatchManager::get_global_min_bounds() const {
	if (patch_metadata_.empty()) {
		return Vector3(0.0f, 0.0f, 0.0f);
	}

	Vector3 global_min = patch_metadata_.front().min_bounds;
	for (const auto& meta : patch_metadata_) {
		global_min.x = std::min(global_min.x, meta.min_bounds.x);
		global_min.y = std::min(global_min.y, meta.min_bounds.y);
		global_min.z = std::min(global_min.z, meta.min_bounds.z);
	}
	return global_min;
}

Vector3 PatchManager::get_global_max_bounds() const {
	if (patch_metadata_.empty()) {
		return Vector3(0.0f, 0.0f, 0.0f);
	}

	Vector3 global_max = patch_metadata_.front().max_bounds;
	for (const auto& meta : patch_metadata_) {
		global_max.x = std::max(global_max.x, meta.max_bounds.x);
		global_max.y = std::max(global_max.y, meta.max_bounds.y);
		global_max.z = std::max(global_max.z, meta.max_bounds.z);
	}
	return global_max;
}

patch_t PatchManager::find_patch_for_position(const Vector3& position) const {
	for (const auto& meta : patch_metadata_) {
		if (position.x >= meta.min_bounds.x && position.x < meta.max_bounds.x &&
			position.y >= meta.min_bounds.y && position.y < meta.max_bounds.y &&
			position.z >= meta.min_bounds.z && position.z < meta.max_bounds.z) {
			return meta.patch_id;
		}
	}
	return static_cast<patch_t>(-1);
}

//================================================================================
// Communication and Halo Exchange
//================================================================================

std::vector<Event> PatchManager::exchange_halo_particles(const Length& cutoff) {
	(void)cutoff;

	// TODO: Implement GPU/MPI halo exchange.
	// For now we return an empty event list so that callers can proceed.
	std::vector<Event> events;
	return events;
}

std::vector<Event> PatchManager::migrate_particles() {
	// TODO: Implement particle migration between patches.
	std::vector<Event> events;
	return events;
}

void PatchManager::synchronize_communication() {
#ifdef USE_MPI
	MPI_Barrier(MPI_COMM_WORLD);
#endif
}

#ifdef USE_MPI
void PatchManager::exchange_metadata_mpi() {
	// Placeholder for future MPI metadata exchange. Currently handled via plan.
}

idx_t PatchManager::gather_global_particle_count(idx_t local_count) {
	idx_t global_count = local_count;
	MPI_Allreduce(&local_count, &global_count, 1, MPI_UNSIGNED_LONG, MPI_SUM, MPI_COMM_WORLD);
	return global_count;
}
#endif

//================================================================================
// Force Computation and Integration
//================================================================================

std::vector<Event>
PatchManager::compute_nonbonded_forces(const NonBondedInteractions& interactions,
									   const DeviceParticleTypes& particle_types,
									   const GridManager& grid_manager,
									   float electric_field,
									   int interpolation_scheme) {
	std::vector<Event> events;
	events.reserve(patches_.size());

	const auto& resources = sim_system_.get_resources();
	for (auto& patch_ptr : patches_) {
		if (!patch_ptr)
			continue;

		size_t resource_idx = 0;
		for (size_t i = 0; i < resources.size(); ++i) {
			if (resources[i] == patch_ptr->get_resource()) {
				resource_idx = i;
				break;
			}
		}

		events.push_back(patch_ptr->calculate_nonbonded_forces(
			interactions,
			particle_types,
			grid_manager.get_device_grid_views(resource_idx),
			electric_field,
			interpolation_scheme));
	}
	return events;
}

std::vector<Event> PatchManager::compute_bonded_forces(const BondedInteractions& interactions,
													   const DeviceParticleTypes& particle_types,
													   const TablesRegistry& tables_registry) {
	std::vector<Event> events;
	events.reserve(patches_.size());
	const auto& resources = sim_system_.get_resources();

	for (auto& patch_ptr : patches_) {
		if (!patch_ptr)
			continue;

		size_t resource_idx = 0;
		for (size_t i = 0; i < resources.size(); ++i) {
			if (resources[i] == patch_ptr->get_resource()) {
				resource_idx = i;
				break;
			}
		}

		events.push_back(patch_ptr->calculate_bonded_forces(interactions,
															particle_types,
															tables_registry,
															resource_idx));
	}
	return events;
}

std::vector<Event> PatchManager::integrate_motion(float dt,
												  const Temperature& temperature,
												  IntegratorType integrator_type) {
	std::vector<Event> events;
	events.reserve(patches_.size());

	for (auto& patch_ptr : patches_) {
		if (!patch_ptr)
			continue;
		events.push_back(patch_ptr->integrate_motion(dt, temperature, integrator_type));
	}
	return events;
}

//================================================================================
// Load Balancing and Redecomposition
//================================================================================

bool PatchManager::needs_rebalancing(float imbalance_threshold) const {
	const LoadStats stats = get_load_statistics();
	return stats.imbalance_factor > imbalance_threshold;
}

PatchManager::LoadStats PatchManager::get_load_statistics() const {
	LoadStats stats;
	if (patch_metadata_.empty()) {
		return stats;
	}

	stats.patch_loads.reserve(patch_metadata_.size());
	for (const auto& meta : patch_metadata_) {
		const float load = static_cast<float>(meta.particle_count);
		stats.patch_loads.push_back(load);
	}

	stats.min_load = *std::min_element(stats.patch_loads.begin(), stats.patch_loads.end());
	stats.max_load = *std::max_element(stats.patch_loads.begin(), stats.patch_loads.end());

	float sum = 0.0f;
	for (float v : stats.patch_loads) {
		sum += v;
	}
	stats.avg_load = sum / static_cast<float>(stats.patch_loads.size());

	if (stats.avg_load > 0.0f) {
		stats.imbalance_factor = (stats.max_load - stats.avg_load) / stats.avg_load;
	} else {
		stats.imbalance_factor = 0.0f;
	}

	return stats;
}

void PatchManager::rebalance_patches(SystemState& state) {
	LOGINFO("PatchManager: Rebalancing patches (delegated to SimSystem::rebalance_system)");
	sim_system_.rebalance_system(state);
}

//================================================================================
// Neighbor Management
//================================================================================

void PatchManager::discover_neighbors() {
	const auto dims = grid_info_.dimensions;
	const bool px = grid_info_.periodic[0];
	const bool py = grid_info_.periodic[1];
	const bool pz = grid_info_.periodic[2];

	const size_t num_patches = patches_.size();
	neighbor_comm_.clear();
	neighbor_comm_.resize(num_patches);

	for (size_t pid = 0; pid < num_patches; ++pid) {
		auto& meta = patch_metadata_[pid];
		std::vector<NeighborComm>& neighbors = neighbor_comm_[pid];
		meta.neighbor_ranks.clear();

		const int i = meta.grid_coords[0];
		const int j = meta.grid_coords[1];
		const int k = meta.grid_coords[2];

		const int dirs[6][3] =
			{{-1, 0, 0}, {1, 0, 0}, {0, -1, 0}, {0, 1, 0}, {0, 0, -1}, {0, 0, 1}};

		for (int d = 0; d < 6; ++d) {
			int ni = i + dirs[d][0];
			int nj = j + dirs[d][1];
			int nk = k + dirs[d][2];

			if (dirs[d][0] != 0 && px) {
				if (ni < 0)
					ni = dims[0] - 1;
				else if (ni >= dims[0])
					ni = 0;
			}
			if (dirs[d][1] != 0 && py) {
				if (nj < 0)
					nj = dims[1] - 1;
				else if (nj >= dims[1])
					nj = 0;
			}
			if (dirs[d][2] != 0 && pz) {
				if (nk < 0)
					nk = dims[2] - 1;
				else if (nk >= dims[2])
					nk = 0;
			}

			if (ni < 0 || ni >= dims[0] || nj < 0 || nj >= dims[1] || nk < 0 || nk >= dims[2]) {
				continue;
			}

			const patch_t neighbor_pid = grid_coords_to_patch_id(ni, nj, nk);
			if (neighbor_pid < 0 || static_cast<size_t>(neighbor_pid) >= num_patches)
				continue;

			const Resource& res = patch_metadata_[pid].resource;
			NeighborComm comm(res, 0); // buffer sizes will be set later
			comm.neighbor_patch_id = neighbor_pid;
			comm.direction = {dirs[d][0], dirs[d][1], dirs[d][2]};
			comm.active = true;

#ifdef USE_MPI
			comm.neighbor_rank = patch_metadata_[static_cast<size_t>(neighbor_pid)].mpi_rank;
			meta.neighbor_ranks.push_back(comm.neighbor_rank);
#endif

			neighbors.push_back(std::move(comm));
		}
	}
}

std::vector<patch_t> PatchManager::get_patch_neighbors(patch_t patch_id) const {
	std::vector<patch_t> result;
	if (patch_id < 0 || static_cast<size_t>(patch_id) >= neighbor_comm_.size()) {
		return result;
	}
	for (const auto& comm : neighbor_comm_[static_cast<size_t>(patch_id)]) {
		if (comm.active) {
			result.push_back(static_cast<patch_t>(comm.neighbor_patch_id));
		}
	}
	return result;
}

//================================================================================
// Metadata and Information
//================================================================================

const PatchManager::PatchMetadata& PatchManager::get_patch_metadata(patch_t patch_id) const {
	if (patch_id < 0 || static_cast<size_t>(patch_id) >= patch_metadata_.size()) {
		throw Exception(ExceptionType::ValueError,
						SourceLocation(),
						"PatchManager::get_patch_metadata invalid patch id {}",
						patch_id);
	}
	return patch_metadata_[static_cast<size_t>(patch_id)];
}

void PatchManager::populate_metadata() {
	// In the new architecture, patch_metadata_ is maintained locally and does not
	// require additional population here. This method is kept for API completeness.
}

void PatchManager::print_status() const {
	LOGINFO("PatchManager: {} patches, grid {}x{}x{}",
			patches_.size(),
			grid_info_.dimensions[0],
			grid_info_.dimensions[1],
			grid_info_.dimensions[2]);
	for (const auto& meta : patch_metadata_) {
		LOGINFO("  Patch {}: bounds min=({}, {}, {}), max=({}, {}, {}), particles={}, capacity={}",
				meta.patch_id,
				meta.min_bounds.x,
				meta.min_bounds.y,
				meta.min_bounds.z,
				meta.max_bounds.x,
				meta.max_bounds.y,
				meta.max_bounds.z,
				meta.particle_count,
				meta.capacity);
	}
}

bool PatchManager::validate_patch_setup() const {
	if (patches_.size() != patch_metadata_.size()) {
		LOGERROR("PatchManager: mismatch between patches ({}) and metadata ({})",
				 patches_.size(),
				 patch_metadata_.size());
		return false;
	}
	return true;
}

//================================================================================
// Private Helper Methods
//================================================================================

void PatchManager::setup_spatial_grid(const DecompositionPlan& plan) {
	setup_spatial_grid(plan.grid_dimensions[0],
					   plan.grid_dimensions[1],
					   plan.grid_dimensions[2],
					   plan.periodicity[0],
					   plan.periodicity[1],
					   plan.periodicity[2]);
}

void PatchManager::initialize_communication_buffers() {
	if (patches_.empty())
		return;

	const Length cutoff = sim_system_.get_cutoff();
	const idx_t max_halo = calculate_max_halo_particles(cutoff);

	for (size_t pid = 0; pid < neighbor_comm_.size(); ++pid) {
		for (auto& comm : neighbor_comm_[pid]) {
			if (!comm.active)
				continue;
			const Resource& res = patch_metadata_[pid].resource;
			comm.send_buffer = DeviceBuffer<float>(max_halo * 8, res);
			comm.recv_buffer = DeviceBuffer<float>(max_halo * 8, res);
			comm.max_halo_particles = max_halo;
		}
	}
}

idx_t PatchManager::calculate_max_halo_particles(const Length& cutoff) const {
	if (patch_metadata_.empty()) {
		return 0;
	}

	// Use first patch as representative for volume estimate
	const auto& meta = patch_metadata_.front();
	const float cutoff_val = cutoff;
	const float dx = meta.max_bounds.x - meta.min_bounds.x;
	const float dy = meta.max_bounds.y - meta.min_bounds.y;
	const float dz = meta.max_bounds.z - meta.min_bounds.z;

	const float volume = dx * dy * dz;
	if (volume <= 0.0f) {
		return meta.capacity / 10;
	}

	const float halo_volume =
		2.0f * cutoff_val * (dx * dy + dx * dz + dy * dz); // slab approximation
	const float density = static_cast<float>(meta.capacity) / volume;

	const idx_t estimate = static_cast<idx_t>(halo_volume * density * 1.5f); // 1.5 safety factor
	return std::max<idx_t>(estimate, 1);
}

patch_t PatchManager::grid_coords_to_patch_id(int i, int j, int k) const {
	const auto dims = grid_info_.dimensions;
	if (i < 0 || i >= dims[0] || j < 0 || j >= dims[1] || k < 0 || k >= dims[2]) {
		return static_cast<patch_t>(-1);
	}
	const int id = k * dims[0] * dims[1] + j * dims[0] + i;
	return static_cast<patch_t>(id);
}

std::array<int, 3> PatchManager::patch_id_to_grid_coords(patch_t patch_id) const {
	const auto dims = grid_info_.dimensions;
	const int id = static_cast<int>(patch_id);
	const int nx = dims[0];
	const int ny = dims[1];

	const int k = id / (nx * ny);
	const int rem = id % (nx * ny);
	const int j = rem / nx;
	const int i = rem % nx;

	return {i, j, k};
}

bool PatchManager::are_spatial_neighbors(patch_t patch_id1, patch_t patch_id2) const {
	if (patch_id1 < 0 || patch_id2 < 0)
		return false;
	if (static_cast<size_t>(patch_id1) >= patch_metadata_.size() ||
		static_cast<size_t>(patch_id2) >= patch_metadata_.size())
		return false;

	const auto c1 = patch_metadata_[static_cast<size_t>(patch_id1)].grid_coords;
	const auto c2 = patch_metadata_[static_cast<size_t>(patch_id2)].grid_coords;

	const int di = std::abs(c1[0] - c2[0]);
	const int dj = std::abs(c1[1] - c2[1]);
	const int dk = std::abs(c1[2] - c2[2]);

	// Direct face neighbors only
	return (di + dj + dk == 1);
}

} // namespace ARBD
