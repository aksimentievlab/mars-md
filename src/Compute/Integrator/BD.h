#pragma once

#include "ARBDLogger.h"
#include "Header.h"
#include "Objects/ParticleProperties.h"
#include "../Random/philox.h"
#include "System/PeriodicBox.h"
#include "Types/Types.h"

namespace ARBD {
struct BDIntegrate {
	DEVICE void operator()(idx_t idx,
						   Vector3* positions,
							 Vector3* momenta,
						   const Vector3* forces,
						   const int* types,
						   const ParticleType* particle_types,
						   float timestep,
						   float kT,
						   const Vector3& box_size,
						   uint64_t base_seed,
						   uint32_t base_ctr) {
		Vector3 pos = positions[idx];
		Vector3 force = forces[idx];
		int type = types[idx];

		const ParticleType& pt = particle_types[type];
		float D = kT / pt.transDamping.x; // Diffusion coefficient
		float sqrt_2Ddt = sqrt(2.0f * D * timestep);

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
		Vector3 random(sqrt_2Ddt * r1 * cosf(theta1),
					   sqrt_2Ddt * r1 * sinf(theta1),
					   sqrt_2Ddt * r2 * cosf(theta2));

		// BD update: r(t+dt) = r(t) + D*F*dt/kT + sqrt(2D*dt) * R
		pos += (D / kT) * force * timestep + random;

		// Apply PBC - wrap each component to [0, box_size)
		pos.x = pos.x - box_size.x * floorf(pos.x / box_size.x);
		pos.y = pos.y - box_size.y * floorf(pos.y / box_size.y);
		pos.z = pos.z - box_size.z * floorf(pos.z / box_size.z);

		positions[idx] = pos;
	}
};
} // namespace ARBD
