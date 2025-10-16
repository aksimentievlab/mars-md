#include "System/SystemState.h"
#include "Configuration.h"
#include <algorithm>
#include <cstring>

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

void SystemState::set_particle_positions(const std::vector<Vector3>& positions) {
	if (positions.size() != num_particles_) {
		throw Exception(ExceptionType::ValueError,
						SourceLocation(),
						"Position vector size ({}) does not match number of particles ({})",
						positions.size(),
						num_particles_);
	}

	objects_.particles = positions;

	LOGINFO("SystemState: Updated particle positions");
}

std::vector<Vector3> SystemState::get_particle_positions() const {
	// TODO: In real implementation, this would copy from GPU buffers
	// std::vector<Vector3> positions(num_particles_);
	// positions_buffer.copy_to_host(positions.data(), num_particles_);
	// return positions;

	return particle_positions_;
}

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

} // namespace ARBD
