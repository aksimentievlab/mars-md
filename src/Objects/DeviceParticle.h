#pragma once
#include "Backend/Buffer.h"
#include "Types/Types.h"

namespace ARBD {

/* for reference only. This is not used in the code. */
struct ParticleSoA { // Stored on GPU only
	DeviceBuffer<int> id;
	DeviceBuffer<int> type_id;
	DeviceBuffer<Vector3> position;
	DeviceBuffer<Vector3> momentum;
	DeviceBuffer<Vector3> force;
	DeviceBuffer<float> energy;
	DeviceBuffer<Vector3> orientation;
	DeviceBuffer<bool> is_dummy;
	DeviceBuffer<bool> has_orientation;
	DeviceBuffer<bool> is_ghost;
	size_t num_local_particles = 0;
	size_t num_ghost_particles = 0;
	size_t capacity; // All buffers share the same capacity
};
} // namespace ARBD
