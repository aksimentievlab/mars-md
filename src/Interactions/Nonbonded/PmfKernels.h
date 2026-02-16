#pragma once
#include "Objects/DeviceParticle.h"
#include "Types/BaseGridDevice.h"
#include "Types/Types.h"

namespace ARBD {
struct ComputePMFKernel {
	KERNEL_FUNC void operator()(size_t i,
								const Vector3* __restrict__ const positions,
								const int* __restrict__ const type_ids,
								Vector3* __restrict__ const forces,
								const ParticleTypeView* __restrict__ const types,
								const BaseGridView<float>* __restrict__ const grid_configs,
								const float** __restrict__ const grid_data_ptrs,
								idx_t num_particles) const {

		idx_t idx = static_cast<idx_t>(i);
		if (idx >= num_particles)
			return;

		const BaseGridView<float>& grid_cfg = grid_configs[idx];
		const float* grid_data = grid_data_ptrs[grid_cfg.grid_id];

		Vector3 pos = positions[idx];
		int type_id = type_ids[idx];

		const ParticleTypeView& ptype = types[type_id];

		Vector3 pmf_force(0.0f, 0.0f, 0.0f);

		// Loop over PMF grids for this type
		for (int g = 0; g < ptype.pmf_grid_id[0]; ++g) {
			int grid_idx = ptype.pmf_grid_id[g];

			// Get grid data and config
			const float* grid_data = grid_data_ptrs[grid_idx];

			// Use BaseGrid's device-safe interpolation!
			float pmf_value = interpolate_grid_point(grid_data,
													 pos,
													 grid_cfg.origin,
													 grid_cfg.basis.inverse(),
													 grid_cfg.dimensions,
													 grid_cfg.boundary_condition);

			// Apply scaling
			if (ptype.pmf_scale) {
				pmf_value *= ptype.pmf_scale[g];
			}

			// Compute gradient using BaseGrid's device function
			Vector3 grad = compute_gradient(grid_data,
											pos,
											grid_cfg.origin,
											grid_cfg.basis,
											grid_cfg.basis.inverse(),
											grid_cfg.dimensions,
											grid_cfg.boundary_condition);

			pmf_force -= grad; // Force = -grad(PMF)
		}

		forces[idx] += pmf_force;
	}
};
} // namespace ARBD
#ifdef USE_SYCL
#include <sycl/sycl.hpp>
template<>
struct sycl::is_device_copyable<ARBD::ComputePMFKernel> : std::true_type {};
#endif
