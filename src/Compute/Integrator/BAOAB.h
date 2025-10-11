#pragma once
#include "Header.h"
#include "Objects/Particles/ParticleType.h"
#include "Random/Random.h"
#include "SimSystem.h"
#include "Types/IndexList.h"
#include "Types/Types.h"
#include "Types/Vector3.h"

namespace ARBD {
struct BAOABIntegrateParams {
	Vector3* positions;
	Vector3* momenta;
	const Vector3* forces;
	const int* types;
	const ParticleType* particle_types;
};
template<typename T>
DEVICE void baoab_integrate_device(idx_t idx,
								   Vector3* positions,
								   Vector3* momenta,
								   const Vector3* forces,
								   const int* types,
								   const ParticleType* particle_types,
								   float timestep,
								   float kT,
								   bool first_step) {
	if (idx >= num_particles)
		return;

	Vector3 pos = positions[idx];
	Vector3 mom = momenta[idx];
	Vector3 force = forces[idx];
	int type = types[idx];

	const ParticleType& pt = particle_types[type];
	float mass = pt.mass;
	float gamma = pt.transDamping.x;

	// BAOAB integration scheme
	// B: momentum update (half step)
	if (first_step) {
		mom += 0.5f * timestep * force;
	}

	// A: position update (half step)
	pos += 0.5f * timestep * mom / mass;

	// O: Ornstein-Uhlenbeck process
	float c = exp(-gamma * timestep);
	float noise_scale = sqrt(kT * mass * (1.0f - c * c));
	Vector3 random_force = rng_state->gaussian_vector(idx) * noise_scale;
	mom = c * mom + random_force;

	// A: position update (half step)
	pos += 0.5f * timestep * mom / mass;

	// Apply periodic boundary conditions
	pos = wrap_position(pos);

	// B: momentum update (half step) - done in next step

	positions[idx] = pos;
	momenta[idx] = mom;
}

// Launch function for different backends
Event launch_baoab_integrate(const Resource& resource,
							 DeviceBuffer<Vector3>& positions,
							 DeviceBuffer<Vector3>& momenta,
							 const DeviceBuffer<Vector3>& forces,
							 const DeviceBuffer<int>& types,
							 float timestep,
							 float temperature,
							 bool first_step) {
	KernelConfig config;

#ifdef USE_CUDA
	if (resource.type == ResourceType::CUDA) {
		return launch_cuda_kernel(resource,
								  config,
								  baoab_integrate_device<float>,
								  positions.data(),
								  momenta.data(),
								  forces.data(),
								  types.data(),
								  /* particle_types, */
								  timestep,
								  temperature * kB,
								  rng_state,
								  first_step);
	}
#endif

#ifdef USE_SYCL
	if (resource.type == ResourceType::SYCL) {
		return launch_sycl_kernel(resource,
								  config,
								  baoab_integrate_device<float>,
								  positions.data(),
								  momenta.data(),
								  forces.data(),
								  types.data(),
								  /* particle_types, */
								  timestep,
								  temperature * kB,
								  rng_state,
								  first_step);
	}
#endif

#ifdef USE_METAL
	if (resource.type == ResourceType::METAL) {
		return launch_metal_kernel(resource,
								   "baoab_integrate", // Kernel name in Metal shader
								   config,
								   positions,
								   momenta,
								   forces,
								   types,
								   timestep,
								   temperature * kB,
								   first_step);
	}
#endif

	return Event{};
}
} // namespace ARBD
