#pragma once
#include "../Random/philox.h"
#include "Header.h"
#include "Objects/ParticleProperties.h"
#include "System/SimSystem.h"
#include "Types/IndexList.h"
#include "Types/Types.h"
#include "Types/Vector3.h"

namespace ARBD {
struct BAOABIntegrate {

	template<typename T>
	DEVICE void operator()(idx_t idx,
						   Vector3* positions,
						   Vector3* momenta,
						   const Vector3* forces,
						   const int* types,
						   const ParticleTypeView* particle_types,
						   const Vector3& box_size,
						   float timestep,
						   float kT,
						   size_t num_particles,
						   uint64_t base_seed,
						   uint32_t base_ctr) {
		if (idx >= num_particles)
			return;

		Vector3 pos = positions[idx];
		Vector3 mom = momenta[idx];
		Vector3 force = forces[idx];
		int type = types[idx];

		const ParticleTypeView& pt = particle_types[idx]; // Do I really need this?
		float mass = *pt.mass;
		Vector3 gamma3 = pt.transDamping[idx];
		float gamma = gamma3.length(); //???
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

		positions[idx] = pos;
		momenta[idx] = mom;
	}
};
} // namespace ARBD
