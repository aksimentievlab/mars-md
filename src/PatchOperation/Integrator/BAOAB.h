#pragma once
#include "../Random/philox.h"
#include "Header.h"
#include "Objects/DeviceParticle.h"
#include "Types/BaseGrid.h"
#include "Types/IndexList.h"
#include "Types/Types.h"
#include "Types/Vector3.h"

namespace ARBD {
template<typename TemperatureType = float>
struct BAOABIntegrate {
	ParticleView particle_view;
	const ParticleTypeView* particle_types;
	Vector3 box_size;
	float timestep;
	TemperatureType kT;
	size_t num_particles;
	uint64_t base_seed;
	uint32_t base_ctr;

	// Constructor for proper initialization
	BAOABIntegrate(ParticleView pv,
				   const ParticleTypeView* pt,
				   const Vector3& box,
				   float dt,
				   TemperatureType temp,
				   size_t n,
				   uint64_t seed,
				   uint32_t ctr)
		: particle_view(pv), particle_types(pt), box_size(box), timestep(dt), kT(temp),
		  num_particles(n), base_seed(seed), base_ctr(ctr) {}

	DEVICE void operator()(idx_t idx) const {
		if (idx >= num_particles)
			return;

		Vector3 pos = particle_view.pos[idx];
		Vector3 mom = particle_view.mom[idx];
		Vector3 force = particle_view.ForceEnergy[idx];
		int type = particle_view.type_id[idx];

		const ParticleTypeView& pt = particle_types[type];
		float mass = pt.mass[type];
		Vector3 gamma3 = pt.transDamping[type];
		float gamma = gamma3.length();
		// BAOAB integration scheme
		// B: momentum update (half step)
		mom += 0.5f * timestep * force;

		// A: position update (half step)
		pos += 0.5f * timestep * mom / mass;

		// O: Ornstein-Uhlenbeck process
		float c = exp(-gamma * timestep);
		float noise_scale = sqrt(kT * mass * (1.0f - c * c));

		// Initialize RNG for this thread with unique counter
		openrand::Philox rng(base_seed, base_ctr + static_cast<uint32_t>(idx));

		// Generate Gaussian random vector using Box-Muller transform
		// Get 4 uniform random numbers
		openrand::float4 uniform = rng.draw_float4();

		// Box-Muller transform for generating pairs of Gaussian values
		float r1 = sqrtf(-2.0f * logf(uniform.x));
		float theta1 = 2.0f * 3.1415926535f * uniform.y;
		float r2 = sqrtf(-2.0f * logf(uniform.z));
		float theta2 = 2.0f * 3.1415926535f * uniform.w;

		// Generate three independent Gaussian values
		Vector3 random(noise_scale * r1 * cosf(theta1),
					   noise_scale * r1 * sinf(theta1),
					   noise_scale * r2 * cosf(theta2));
		Vector3 random_force = random;
		mom = c * mom + random_force;

		// A: position update (half step)
		pos += 0.5f * timestep * mom / mass;

		// Apply periodic boundary conditions
		pos.x = pos.x - box_size.x * floorf(pos.x / box_size.x);
		pos.y = pos.y - box_size.y * floorf(pos.y / box_size.y);
		pos.z = pos.z - box_size.z * floorf(pos.z / box_size.z);

		// B: momentum update (half step) - done in next step

		particle_view.pos[idx] = pos;
		particle_view.mom[idx] = mom;
	}
};
} // namespace ARBD

// SYCL device copyable trait
#ifdef USE_SYCL
#include <sycl/sycl.hpp>
template<typename TemperatureType>
struct sycl::is_device_copyable<ARBD::BAOABIntegrate<TemperatureType>> : std::true_type {};
#endif
