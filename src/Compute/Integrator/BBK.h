#pragma once
#include "../Random/philox.h"
#include "Header.h"
#include "Objects/ParticleProperties.h"
#include "SimSystem.h"
#include "Types/IndexList.h"
#include "Types/Types.h"
#include "Types/Vector3.h"

namespace ARBD {
struct BBKIntegrate {
	DEVICE void operator()(idx_t idx,
						   Vector3* positions,
						   Vector3* momenta,
						   const Vector3* forces,
						   const int* types,
						   const ParticleType* particle_types,
						   const Vector3& box_size,
						   float timestep,
						   float kT,
						   size_t num_particles,
						   uint64_t base_seed,
						   uint32_t base_ctr) {
		Vector3 pos = positions[idx];
		Vector3 mom = momenta[idx];
		Vector3 force = forces[idx];
		int type = types[idx];

		const ParticleType& pt = particle_types[type];
		float mass = pt.mass;
		float gamma = pt.transDamping.x;
	}
};
} // namespace ARBD
