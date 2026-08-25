#pragma once
#include "Objects/DeviceParticle.h"
#include "Types/BaseGridDevice.h"
#include "Types/Types.h"

namespace MARS {

namespace pmf_detail {

HOST DEVICE inline GridSample<mars_real>
sample_pmf_grid(const BaseGridView<mars_real>& grid, const Vector3& pos, int scheme) {
	if (scheme == 0) {
		return sample_grid_linear(grid.data,
								  pos,
								  grid.origin,
								  grid.basis,
								  grid.basis_inv,
								  grid.dimensions,
								  grid.boundary_condition);
	}
	return sample_grid_cubic(grid.data,
							 pos,
							 grid.origin,
							 grid.basis,
							 grid.basis_inv,
							 grid.dimensions,
							 grid.boundary_condition);
}

HOST DEVICE inline float
sample_force_grid_value(const BaseGridView<mars_real>& grid, const Vector3& pos, int scheme) {
	if (scheme == 0) {
		return interpolate_grid_point(grid.data,
									  pos,
									  grid.origin,
									  grid.basis_inv,
									  grid.dimensions,
									  grid.boundary_condition);
	}
	return sample_grid_cubic(grid.data,
							 pos,
							 grid.origin,
							 grid.basis,
							 grid.basis_inv,
							 grid.dimensions,
							 grid.boundary_condition)
		.value;
}

} // namespace pmf_detail

/**
 * @brief Position-dependent forces from legacy GrandBrownTown.cuh
 * This function computes the position-dependent forces acting on a particle
 * due to the presence of PMF (Potential of Mean Force) grids.
 * A type can reference any number of PMF grids; its terms occupy
 * [offset, offset + count) of the flat table
 * @param pos: position of the particle
 * @param type_id: type id of the particle
 * @param types: particle type view
 * @param grid_configs: grid configurations
 * @param electric_field: electric field
 * @param scheme: 0 = linear interpolation, 1 = cubic (legacy: !scheme vs scheme)
 * @param get_energy: whether to get the energy
 */
HOST DEVICE inline Vector3
compute_position_dependent_force(const Vector3& pos,
								 int type_id,
								 const ParticleTypeView types,
								 const BaseGridView<mars_real>* grid_configs,
								 const Vector3& electric_field,
								 int scheme,
								 bool get_energy = false) {
	const float charge = types.charge[type_id];
	Vector3 force(charge * electric_field.x, charge * electric_field.y, charge * electric_field.z);

	if (grid_configs != nullptr) {
		const int pmf_begin = types.pmf_grid_offset[type_id];
		const int pmf_end = pmf_begin + types.pmf_grid_count[type_id];
		for (int k = pmf_begin; k < pmf_end; ++k) {
			const GridTerm term = types.pmf_grid_terms[k];
			if (!term.is_valid())
				continue;
			BaseGridView<mars_real> pmf_grid = grid_configs[term.grid_id];
			if (pmf_grid.data == nullptr)
				continue;
			if (term.boundary_condition >= 0)
				pmf_grid.boundary_condition = term.boundary_condition;
			const GridSample<mars_real> sample = pmf_detail::sample_pmf_grid(pmf_grid, pos, scheme);
			force += sample.gradient * (-term.scale);
			if (get_energy) {
				force.t += term.scale * sample.value;
			}
		}
	}

	const int3 force_grid_id = types.force_grid_id[type_id];
	if (grid_configs != nullptr) {
		const Vector3 force_grid_scale = types.force_grid_scale[type_id];
		if (force_grid_id.x >= 0) {
			const BaseGridView<mars_real>& grid = grid_configs[force_grid_id.x];
			if (grid.data != nullptr) {
				force.x +=
					force_grid_scale.x * pmf_detail::sample_force_grid_value(grid, pos, scheme);
			}
		}
		if (force_grid_id.y >= 0) {
			const BaseGridView<mars_real>& grid = grid_configs[force_grid_id.y];
			if (grid.data != nullptr) {
				force.y +=
					force_grid_scale.y * pmf_detail::sample_force_grid_value(grid, pos, scheme);
			}
		}
		if (force_grid_id.z >= 0) {
			const BaseGridView<mars_real>& grid = grid_configs[force_grid_id.z];
			if (grid.data != nullptr) {
				force.z +=
					force_grid_scale.z * pmf_detail::sample_force_grid_value(grid, pos, scheme);
			}
		}
	}

	return force;
}

struct ComputePMFKernel {
	Vector3 electric_field{0.0f, 0.0f, 0.0f};
	int scheme = 0;
	bool get_energy = false;

	KERNEL_FUNC void operator()(size_t i,
								const ParticleView particles,
								const ParticleTypeView types,
								const BaseGridView<mars_real>* __restrict__ grid_configs,
								idx_t num_particles) const {

		const idx_t idx = static_cast<idx_t>(i);
		if (idx >= num_particles)
			return;

		const int type_id = particles.type_id[idx];
		const Vector3 pos = particles.pos[idx];

		particles.ForceEnergy[idx] += compute_position_dependent_force(pos,
																	   type_id,
																	   types,
																	   grid_configs,
																	   electric_field,
																	   scheme,
																	   get_energy);
	}
};

} // namespace MARS
#ifdef USE_CUDA
#include "Backend/CUDA/KernelHelper.cuh"
namespace MARS {
extern template Event launch_cuda_kernel(const Resource& resource,
										 const KernelConfig& config,
										 ComputePMFKernel kernel_func,
										 ParticleView particles,
										 ParticleTypeView types,
										 const BaseGridView<mars_real>* grid_configs,
										 idx_t num_particles);
} // namespace MARS
#endif

#ifdef USE_SYCL
#include <sycl/sycl.hpp>
template<>
struct sycl::is_device_copyable<MARS::ComputePMFKernel> : std::true_type {};
#endif
