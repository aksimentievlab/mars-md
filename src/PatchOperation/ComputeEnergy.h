#pragma once

#include "Header.h"
#include "Types/Types.h"
#include "Types/Vector3.h"

namespace ARBD {

/**
 * @brief Forward declaration of BrownianParticleType structure
 * This structure should have at least a 'mass' member
 */
struct BrownianParticleType {
	float mass;
	// Add other members as needed
};

/**
 * @brief WorkGroup-based reduction kernel for computing Brownian particles kinetic energy
 *
 * This kernel computes the kinetic energy of Brownian particles using a reduction pattern.
 * It processes momentum vectors and computes p.length2() / mass for each particle,
 * handling multiple replicas with special indexing logic.
 * The block_size parameter should be a power of 2 (32, 64, 128, 256, or 512) for optimal
 * performance.
 */

struct BrownParticlesKineticEnergyKernel {

	template<typename WorkItem>
	DEVICE void operator()(size_t i,
						   WorkItem& item,
						   const Vector3* __restrict__ P_n,
						   const int* __restrict__ type,
						   const BrownianParticleType* __restrict__ part[],
						   float* __restrict__ vec_red,
						   unsigned int n,
						   unsigned int num,
						   unsigned int num_rb_attached_particles,
						   unsigned int num_replicas,
						   unsigned int block_size,
						   unsigned int grid_size) const {

		// Get shared memory for reduction
		auto* sdata = item.template get_shared_mem<float>();

		unsigned int tid = static_cast<unsigned int>(item.local_id());

		Vector3 p1, p2;
		float mass1, mass2;

		// Initialize shared memory
		sdata[tid] = 0.0f;

		unsigned int idx = static_cast<unsigned int>(item.group_id()) * (block_size << 1) + tid;

		// Load data into shared memory with multiple elements per thread
		// Process 2 elements per thread for better bandwidth utilization
		while (idx < n) {
			const unsigned int i1 = (idx % num) + (idx / num) * (num + num_rb_attached_particles);
			const int t1 = type[i1];
			const BrownianParticleType& pt1 = *part[t1];

			p1 = P_n[idx];
			mass1 = pt1.mass;

			if (idx + block_size < n) {
				const unsigned int i2 =
					((idx + block_size) % num) +
					((idx + block_size) / num) * (num + num_rb_attached_particles);
				const int t2 = type[i2];
				const BrownianParticleType& pt2 = *part[t2];

				p2 = P_n[idx + block_size];
				mass2 = pt2.mass;

				sdata[tid] += (p1.length2() / mass1 + p2.length2() / mass2);
			} else {
				sdata[tid] += p1.length2() / mass1;
			}

			idx += grid_size;
		}

		// Apply the 0.5 factor (kinetic energy = 0.5 * m * v^2, but we have p^2/m)
		sdata[tid] *= 0.5f;

		item.barrier();

		// Reduction using a generic loop (works for any power-of-2 block size)
		unsigned int s = block_size;
		while (s > 1) {
			s >>= 1;
			if (tid < s) {
				sdata[tid] += sdata[tid + s];
			}
			item.barrier();
		}

		// Write block sum to output
		if (tid == 0) {
			vec_red[item.group_id()] = sdata[0];
		}
	}
};

} // namespace ARBD
