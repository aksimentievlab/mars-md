#include "System/SystemState.h"
#include "PatchOperation/Patch.h"

namespace ARBD {

//================================================================================
// SystemState Implementation
//================================================================================

SystemState::SystemState() {
	LOGINFO("SystemState: Initializing runtime state");

	// Initialize system objects
	initialize_system_objects();

	LOGINFO("SystemState: Runtime state initialized");
}

SystemState::~SystemState() {
	cleanup_gpu_resources();
}

//================================================================================
// Particle State Management
//================================================================================

std::vector<Vector3> SystemState::get_particle_positions() const {}

//================================================================================
// System Object Management
//================================================================================

void SystemState::initialize_system_objects() {
	LOGINFO("SystemState: Initializing system objects");

	// Initialize based on configuration
	// These would be set based on actual system requirements
	has_bonds_ = false;			  // Set based on actual bond data
	has_external_forces_ = false; // Set based on grid/field configuration

	// Initialize GPU grids if needed
	// if (temperature_format_ == 1 && temperature_grid_) {
	//     // Copy temperature grid to device
	//     // temperature_grid_ = copy_grid_to_device(temperature_grid_);
	// }

	LOGINFO("SystemState: System objects initialized");
}

//================================================================================
// Private Methods
//================================================================================

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

void SystemState::gather_from_patches(PatchManager& patch_manager) {
	// Clear previous state
	global_positions_.clear();
	global_momentum_.clear();
	global_particle_ids_.clear();
	global_particle_types_.clear();

#ifdef USE_MPI
	// In MPI mode: gather from local patch + exchange with other ranks
	const Patch& local_patch = patch_manager.get_local_patch();

	// Collect from local patch
	idx_t local_num = local_patch.get_num();
	for (idx_t i = 0; i < local_num; ++i) {
		HostParticleData particle_data = local_patch.get_particle_data();
		global_positions_.push_back(particle_data.pos[i]);
		global_momentum_.push_back(particle_data.mom[i]); // Assuming momentum = velocity * mass
		global_particle_ids_.push_back(particle_data.id[i]);
		global_particle_types_.push_back(particle_data.type_id[i]);
	}

	// TODO: MPI_Gatherv to collect from all ranks and assemble in global order
	// This requires coordination with PatchManager's MPI communication

#else
	// Non-MPI: collect from all local patches
	const auto& patches = patch_manager.get_all_patches();
	for (const auto& patch : patches) {
		idx_t num = patch.get_num();
		for (idx_t i = 0; i < num; ++i) {
			global_positions_.push_back(patch.particle_positions[i]);
			global_velocities_.push_back(patch.particle_momenta[i]);
			global_particle_ids_.push_back(patch.particle_ids[i]);
			global_particle_types_.push_back(patch.particle_type_ids[i]);
		}
	}
#endif

	global_num_particles_ = global_positions_.size();
	state_synced_ = true;
}

} // namespace ARBD
