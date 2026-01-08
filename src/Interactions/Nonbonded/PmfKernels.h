// Kernel that uses PMF grids with your BaseGrid

#include "Backend/Kernels.h"
#include "Backend/Resource.h"
#include "Objects/DeviceParticle.h"
#include "Objects/ParticleProperties.h"
#include "SimParam.h"
#include "Types/BaseGrid.h"
#include "Types/Vector3.h"

namespace ARBD {
struct ComputePMFKernel {
	KERNEL_FUNC void
	operator()(size_t i,
			   const Vector3* __restrict__ const positions,
			   const int* __restrict__ const type_ids,
			   Vector3* __restrict__ const forces,
			   const ParticleTypeView* __restrict__ const types,
			   const float** __restrict__ const grid_data_ptrs, // Array of grid pointers
			   const BaseGrid<float>::Config* __restrict__ const grid_configs, // Grid metadata
			   idx_t num_particles) const {

		idx_t idx = static_cast<idx_t>(i);
		if (idx >= num_particles)
			return;

		Vector3 pos = positions[idx];
		int type_id = type_ids[idx];

		const ParticleTypeView& ptype = types[type_id];

		Vector3 pmf_force(0.0f, 0.0f, 0.0f);

		// Loop over PMF grids for this type
		for (int g = 0; g < ptype.pmf_grid_id[0]; ++g) {
			int grid_idx = ptype.pmf_grid_id[g];

			// Get grid data and config
			const float* grid_data = grid_data_ptrs[grid_idx];
			const auto& grid_cfg = grid_configs[grid_idx];

			// Use BaseGrid's device-safe interpolation!
			float pmf_value = interpolate_grid_point(grid_data,
													 pos,
													 grid_cfg.origin,
													 grid_cfg.basis.inverse(),
													 grid_cfg.dimensions,
													 static_cast<int>(grid_cfg.boundary));

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
											static_cast<int>(grid_cfg.boundary));

			pmf_force -= grad; // Force = -grad(PMF)
		}

		forces[idx] += pmf_force;
	}
};
} // namespace ARBD
