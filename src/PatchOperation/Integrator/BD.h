#pragma once

#include "../Random/philox.h"
#include "Header.h"
#include "Objects/DeviceParticle.h"
#include "Types/Types.h"

namespace ARBD {
template<typename TemperatureType = float>
struct BDIntegrate {
	ParticleView particle_view; // Pass by value (pointers are copied, data is shared)
	float timestep;
	TemperatureType kT;
	const ParticleTypeView* particle_types;
	Vector3 box_size; // Pass by value for device compatibility
	idx_t num_particles;
	uint64_t base_seed;
	uint32_t base_ctr;

	// Constructor for proper initialization
	BDIntegrate(ParticleView pv,
				const ParticleTypeView* pt,
				float dt,
				TemperatureType temp,
				idx_t n,
				const Vector3& box,
				uint64_t seed,
				uint32_t ctr)
		: particle_view(pv), timestep(dt), kT(temp), particle_types(pt), box_size(box),
		  num_particles(n), base_seed(seed), base_ctr(ctr) {}

	DEVICE void operator()(idx_t idx) const {
		if (idx >= num_particles)
			return;
		Vector3 pos = particle_view.pos[idx];
		Vector3 ForceEnergy = particle_view.ForceEnergy[idx];
		int type = particle_view.type_id[idx];
		const ParticleTypeView& pt = particle_types[type];
		Vector3 transDamping = pt.transDamping[type];
		float mass = pt.mass[type];

		float D = kT / transDamping.x; // Diffusion coefficient
		float sqrt_2Ddt = sqrt(2.0f * D * timestep);

		// Initialize RNG for thiss thread with unique counter
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
		Vector3 random(sqrt_2Ddt * r1 * cosf(theta1),
					   sqrt_2Ddt * r1 * sinf(theta1),
					   sqrt_2Ddt * r2 * cosf(theta2));

		// BD update: r(t+dt) = r(t) + D*F*dt/kT + sqrt(2D*dt) * R
		pos += (D / kT) * ForceEnergy * timestep + random;

		// Apply PBC - wrap each component to [0, box_size)
		pos.x = pos.x - box_size.x * floorf(pos.x / box_size.x);
		pos.y = pos.y - box_size.y * floorf(pos.y / box_size.y);
		pos.z = pos.z - box_size.z * floorf(pos.z / box_size.z);

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
