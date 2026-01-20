#pragma once

#include "../Random/philox.h"
#include "Header.h"
#include "Objects/DeviceParticle.h"
#include "Types/Types.h"

namespace ARBD {
template<typename TemperatureType = float>
struct BDIntegrate {
	ParticleView particle_view; // Pass by value (pointers are copied, data is shared)
	const ParticleTypeView particle_types;
	Vector3 box_size; // Pass by value for device compatibility
	float timestep;
	TemperatureType kT;
	idx_t num_particles;
	uint64_t base_seed;
	uint32_t base_ctr;
	size_t current_step;

	// Constructor for proper initialization
	BDIntegrate(ParticleView pv,
				const ParticleTypeView pt,
				const Vector3& box,
				float dt,
				size_t current_step,
				TemperatureType temp,
				idx_t n,
				uint64_t seed,
				uint32_t ctr)
		: particle_view(pv), particle_types(pt), box_size(box), timestep(dt), kT(temp),
		  num_particles(n), base_seed(seed), base_ctr(ctr), current_step(current_step) {}

	KERNEL_FUNC void operator()(idx_t idx) const {
		if (idx >= num_particles)
			return;

		Vector3 pos = particle_view.pos[idx];
		Vector3 force = particle_view.ForceEnergy[idx];
		int type = particle_view.type_id[idx];

		// Get particle type properties
		float mass = particle_types.mass[type];
		Vector3 diffusion = particle_types.diffusion[type];

		// Generate Gaussian random numbers
		openrand::Philox rng(base_seed + current_step, base_ctr + static_cast<uint32_t>(idx));
		openrand::float4 uniform = rng.draw_float4();

		// Box-Muller transform
		float r1 = sqrtf(-2.0f * logf(uniform.x));
		float theta1 = 2.0f * 3.1415926535f * uniform.y;
		float r2 = sqrtf(-2.0f * logf(uniform.z));
		float theta2 = 2.0f * 3.1415926535f * uniform.w;

		Vector3 random(r1 * cosf(theta1), r1 * sinf(theta1), r2 * cosf(theta2));

		pos.x += (diffusion.x / kT) * force.x * timestep +
				 sqrtf(2.0f * diffusion.x * timestep) * random.x;
		pos.y += (diffusion.y / kT) * force.y * timestep +
				 sqrtf(2.0f * diffusion.y * timestep) * random.y;
		pos.z += (diffusion.z / kT) * force.z * timestep +
				 sqrtf(2.0f * diffusion.z * timestep) * random.z;

		// Apply periodic boundary conditions
		pos.x = pos.x - box_size.x * roundf(pos.x / box_size.x);
		pos.y = pos.y - box_size.y * roundf(pos.y / box_size.y);
		pos.z = pos.z - box_size.z * roundf(pos.z / box_size.z);

		particle_view.pos[idx] = pos;
	}
};

} // namespace ARBD

// SYCL device copyable trait
#ifdef USE_SYCL
#include <sycl/sycl.hpp>
template<typename TemperatureType>
struct sycl::is_device_copyable<ARBD::BDIntegrate<TemperatureType>> : std::true_type {};
#endif
