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
				ParticleView particle_view,
				const ParticleTypeView particle_types,
				float timestep,
				size_t current_step,
				float kT,
				idx_t num_particles,
				const PeriodicBox& sim_box,
				uint64_t base_seed,
				uint32_t base_ctr) {
	KernelConfig config = KernelConfig::for_1d(num_particles, resource);
	BDIntegrate<T> bd_integrate(particle_view,
								particle_types,
								sim_box,
								timestep,
								current_step,
								kT,
								num_particles,
								base_seed,
								base_ctr);
	Event evt = launch_kernel(resource, config, bd_integrate);
	return evt;
}
// Launch function for different backends
template<typename T>
Event launch_BAOAB(const Resource& resource,
				   ParticleView particle_view,
				   const ParticleTypeView particle_types,
				   const PeriodicBox& sim_box,
				   float timestep,
				   size_t current_step,
				   float kT,
				   idx_t num_particles,
				   uint64_t base_seed,
				   uint32_t base_ctr) {
	KernelConfig config = KernelConfig::for_1d(num_particles, resource);
	BAOABIntegrate<T> baoab_integrate(particle_view,
									  particle_types,
									  sim_box,
									  timestep,
									  current_step,
									  kT,
									  num_particles,
									  base_seed,
									  base_ctr);
	Event evt = launch_kernel(resource, config, baoab_integrate);
	return evt;
}

} // namespace ARBD
