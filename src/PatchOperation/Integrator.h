/*********************************************************************
 * @file  Integrator.h
 *
 * @brief Declaration of Integrator class with factory-like method
 * GetIntegrator()
 *
 * Defines hashable Integrator::Conf struct that allows operator
 * resuse. Common CPU/GPU kernels implemented in Integrator/kernels.h
 * with various BD/MD integrators implemented on different backends in
 * Integrator/CUDA.h and Integrator/CPU.h
 *********************************************************************/
#pragma once

#include "Backend/Kernels.h"
#include "Header.h"
#include "Integrator/BAOAB.h"
#include "Integrator/BD.h"
#include "System/SimSystem.h"
#include "Types/Types.h"
#include "Types/Vector3.h"

namespace ARBD {

template<typename T>
Event launch_BD(const Resource& resource,
				DeviceBuffer<Vector3>& positions,
				const DeviceBuffer<Vector3>& forces,
				const DeviceBuffer<int>& types,
				const ParticleType* particle_types,
				float timestep,
				float kT,
				const Vector3& box_size,
				size_t num_particles,
				uint64_t base_seed,
				uint32_t base_ctr) {
	KernelConfig config;

#ifdef USE_CUDA
	if (resource.type() == ResourceType::CUDA) {
		return launch_cuda_kernel(resource,
								  config,
								  bd_integrate_device<float>,
								  positions.data(),
								  forces.data(),
								  types.data(),
								  particle_types,
								  timestep,
								  kT,
								  box_size,
								  num_particles,
								  base_seed,
								  base_ctr);
	}
#elif defined(USE_SYCL)
	if (resource.type() == ResourceType::SYCL) {
		return launch_sycl_kernel(resource,
								  config,
								  BDIntegrate(),
								  positions,
								  forces,
								  types,
								  particle_types,
								  box_size,
								  timestep,
								  kT,
								  num_particles,
								  base_seed,
								  base_ctr);
	}
#elif defined(USE_METAL)
	if (resource.type() == ResourceType::METAL) {
		return launch_metal_kernel(resource,
								   config,
								   BDIntegrate(),
								   positions,
								   forces,
								   types,
								   particle_types,
								   box_size,
								   timestep,
								   kT,
								   num_particles,
								   base_seed,
								   base_ctr);
	}
#else
	LOGINFO("BD is not supported for this backend");
	return Event(nullptr, resource);
#endif
}
// Launch function for different backends
template<typename T>
Event launch_BAOAB(const Resource& resource,
				   DeviceBuffer<Vector3>& positions,
				   DeviceBuffer<Vector3>& momenta,
				   const DeviceBuffer<Vector3>& forces,
				   const DeviceBuffer<int>& types,
				   const ParticleType* particle_types,
				   const Vector3& box_size,
				   float timestep,
				   float kT,
				   size_t num_particles,
				   uint64_t base_seed,
				   uint32_t base_ctr) {
	KernelConfig config;

#ifdef USE_CUDA
	if (resource.type() == ResourceType::CUDA) {
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
#elif defined(USE_SYCL)
	if (resource.type() == ResourceType::SYCL) {
		return launch_sycl_kernel(resource,
								  config,
								  BAOABIntegrate(),
								  positions,
								  momenta,
								  forces,
								  types,
								  particle_types,
								  box_size,
								  timestep,
								  kT,
								  num_particles,
								  base_seed,
								  base_ctr);
	}
#elif defined(USE_METAL)
	if (resource.type() == ResourceType::METAL) {
		return launch_metal_kernel(resource,
								   config,
								   BAOABIntegrate(),
								   positions,
								   momenta,
								   forces,
								   types,
								   particle_types,
								   box_size,
								   timestep,
								   kT,
								   num_particles,
								   base_seed,
								   base_ctr);
	}
#else
	LOGINFO("BAOAB is not supported for this backend");
	return Event(nullptr, resource);
#endif

}; // namespace ARBD
} // namespace ARBD
