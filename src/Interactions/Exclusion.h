#pragma once

#include "Backend/Kernels.h"
#include "Backend/Resource.h"
#include "BondedInteraction.h"
#include "Header.h"
#include "Types/Types.h"

namespace ARBD {
struct GenerateExclusionsFunctor {
	// A constant defining the max exclusion depth (e.g., 1-4 bonds)
	// This allows us to use a static array for the frontier, avoiding dynamic allocation.
	static constexpr int MAX_DEPTH = 4;

	KERNEL_FUNC void operator()(idx_t i,
								// Global thread ID, corresponds to the starting particle 'i'
								// --- Pointers to GPU Data ---
								DEVICE_PTR(const int2) adjacency_offsets,
								DEVICE_PTR(const int) adjacency_list,
								// Packed (ind1, ind2) pairs - the device-side SoA
								// representation of exclusions, matching
								// DeviceBondedInteractions::exclusion_pairs(). The
								// host-side AoS Exclude type deliberately stays off
								// the device.
								DEVICE_PTR(int2) exclusions_output, // Pre-allocated output buffer
								DEVICE_PTR(int)
									exclusion_count,	 // Atomic counter for the output buffer
								int max_exclusion_depth, // e.g., for "1-4", this would be 3
								int num_particles) const {
		// --- Per-thread data for the BFS ---
		int frontier[MAX_DEPTH * 64];
		// A simple array to hold nodes at each depth (64 is a safe upper bound for neighbors)
		bool visited[2048];
		// A simple hash set for visited nodes (assuming < 2048 particles)

		for (int k = 0; k < num_particles; ++k)
			visited[k] = false;

		int frontier_start = 0;
		int frontier_end = 1;
		frontier[0] = i; // The frontier starts with our assigned particle
		visited[i] = true;
		// --- Main BFS Loop ---
		for (int depth = 1; depth <= max_exclusion_depth; ++depth) {
			int current_frontier_size = frontier_end - frontier_start;
			if (current_frontier_size == 0)
				break; // No more nodes to explore

			int next_frontier_start = frontier_end;

			// For every node currently in our frontier...
			for (int j = frontier_start; j < frontier_end; ++j) {
				int current_particle = frontier[j];

				// ...find all of its neighbors.
				int neighbor_offset = adjacency_offsets[current_particle].x;
				int num_neighbors = adjacency_offsets[current_particle].y;

				for (int k = 0; k < num_neighbors; ++k) {
					int neighbor = adjacency_list[neighbor_offset + k];

					// If we haven't visited this neighbor yet...
					if (!visited[neighbor]) {
						visited[neighbor] = true;

						// Add it to the exclusion list
						int write_idx = ATOMIC_ADD(exclusion_count, 1);
						exclusions_output[write_idx] = int2{(int)i, neighbor};

						// And add it to the frontier for the next depth level
						frontier[frontier_end++] = neighbor;
					}
				}
			}
			frontier_start = next_frontier_start;
		}
	}
};

inline Event launch_exclusion_generation(
	const Resource& resource,
	int num_particles,
	int exclusion_depth, // e.g., 3 for a 1-4 exclusion
	const DeviceBuffer<int2>& adj_offsets,
	const DeviceBuffer<int>& adj_list,
	DeviceBuffer<int2>& out_exclusions,	   // Must be pre-sized large enough
	DeviceBuffer<int>& out_exclusion_count // A buffer with one integer, initialized to 0
) {
	KernelConfig config = KernelConfig::for_1d(num_particles, resource);

	return launch_kernel(resource,
						 config,
						 GenerateExclusionsFunctor{},
						 adj_offsets,
						 adj_list,
						 out_exclusions,
						 out_exclusion_count,
						 exclusion_depth,
						 num_particles);
}
} // namespace ARBD

// Explicit template instantiation declaration to prevent host instantiation.
// Launched with a trailing argument pack, so it is spelled out here exactly as
// launch_kernel forwards it through get_buffer_pointer(). Real definition lives
// in Bonded/BondedInstantiations.cu (exclusions are derived from bonded
// topology adjacency).
#ifdef USE_CUDA
#include "Backend/CUDA/KernelHelper.cuh"
namespace ARBD {
extern template Event launch_cuda_kernel(const Resource& resource,
										 const KernelConfig& config,
										 GenerateExclusionsFunctor kernel_func,
										 int2* adjacency_offsets,
										 int* adjacency_list,
										 int2* exclusions_output,
										 int* exclusion_count,
										 int max_exclusion_depth,
										 int num_particles);
} // namespace ARBD
#endif

#ifdef USE_SYCL
#include <sycl/sycl.hpp>
template<>
struct sycl::is_device_copyable<ARBD::GenerateExclusionsFunctor> : std::true_type {};
#endif
