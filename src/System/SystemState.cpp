#include "System/SystemState.h"
#include <algorithm>
#include <numeric>

namespace MARS {

void SystemState::initialize_system_objects() {
	LOGINFO("SystemState: Initializing system objects");

	LOGINFO("SystemState: System objects initialized");
}

//================================================================================
// Private Methods
//================================================================================

void SystemState::sort_particles_by_id() {
	if (global_particle_data_.size() == 0) {
		return;
	}

	// Create index array sorted by particle ID
	std::vector<size_t> indices(global_particle_data_.size());
	std::iota(indices.begin(), indices.end(), 0);

	// Sort indices based on global_id
	std::sort(indices.begin(), indices.end(), [this](size_t a, size_t b) {
		return global_particle_data_.global_id[a] < global_particle_data_.global_id[b];
	});

	// Check if already sorted
	bool already_sorted = true;
	for (size_t i = 0; i < indices.size(); ++i) {
		if (indices[i] != i) {
			already_sorted = false;
			break;
		}
	}

	if (already_sorted) {
		return; // Already in sorted order
	}

	// Create temporary sorted arrays
	HostParticleData sorted_data;
	sorted_data.resize(global_particle_data_.size());

	for (size_t i = 0; i < indices.size(); ++i) {
		size_t src_idx = indices[i];
		sorted_data.global_id[i] = global_particle_data_.global_id[src_idx];
		sorted_data.type_id[i] = global_particle_data_.type_id[src_idx];
		sorted_data.pos[i] = global_particle_data_.pos[src_idx];
		sorted_data.mom[i] = global_particle_data_.mom[src_idx];
		sorted_data.force[i] = global_particle_data_.force[src_idx];
		sorted_data.energy[i] = global_particle_data_.energy[src_idx];
		sorted_data.orient[i] = global_particle_data_.orient[src_idx];
		sorted_data.is_active[i] = global_particle_data_.is_active[src_idx];
		sorted_data.is_dummy[i] = global_particle_data_.is_dummy[src_idx];
		sorted_data.colvars_group_id[i] = global_particle_data_.colvars_group_id[src_idx];
		sorted_data.group_id[i] = global_particle_data_.group_id[src_idx];
		if (src_idx < global_particle_data_.flags.size()) {
			sorted_data.flags[i] = global_particle_data_.flags[src_idx];
		}
		if (src_idx < global_particle_data_.attached_rigid_body_id.size()) {
			sorted_data.attached_rigid_body_id[i] =
				global_particle_data_.attached_rigid_body_id[src_idx];
		}
	}

	// Replace original data with sorted data
	global_particle_data_ = std::move(sorted_data);

	LOGTRACE("SystemState: Sorted {} particles by global ID", global_particle_data_.size());
}

void SystemState::cleanup_gpu_resources() {
	// TODO: In real implementation, this would clean up GPU resources
	// if (temperature_grid_) {
	//     delete_device_grid(temperature_grid_);
	//     temperature_grid_ = nullptr;
	// }
	// if (force_grid_) {
	//     delete_device_grid(force_grid_);
	//     force_grid_ = nullptr;
	// }
	// if (bond_list_) {
	//     delete_device_array(bond_list_);
	//     bond_list_ = nullptr;
	// }

	LOGINFO("SystemState: GPU resources cleaned up");
}

} // namespace MARS
