#pragma once
#include "Objects/DeviceParticle.h"
#include "Types/BaseGridDevice.h"
#include "Types/Types.h"

namespace ARBD {

namespace pmf_detail {

HOST DEVICE inline GridSample<float> sample_pmf_grid(const BaseGridView<float>& grid,
												   const Vector3& pos,
												   int scheme) {
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

HOST DEVICE inline float sample_force_grid_value(const BaseGridView<float>& grid,
												 const Vector3& pos,
												 int scheme) {
	if (scheme == 0) {
		return interpolate_grid_point(grid.data,
									  pos,
									  grid.origin,
									  grid.basis_inv,
									  grid.dimensions,
									  grid.boundary_condition);
	}
	return grid.interpolate(pos);
}

} // namespace pmf_detail

/**
 * @brief Position-dependent forces from legacy GrandBrownTown.cuh
 *
 * scheme: 0 = linear interpolation, 1 = cubic (legacy: !scheme vs scheme)
 */
HOST DEVICE inline Vector3 compute_position_dependent_force(const Vector3& pos,
															int type_id,
															const ParticleTypeView types,
															const BaseGridView<float>* grid_configs,
															float electric_field,
															int scheme,
															bool get_energy = false) {
	const float charge = types.charge[type_id];
	Vector3 force(0.0f, 0.0f, charge * electric_field);

	const int pmf_grid_id = types.pmf_grid_id[type_id];
	if (pmf_grid_id >= 0 && grid_configs != nullptr) {
		const BaseGridView<float>& pmf_grid = grid_configs[pmf_grid_id];
		if (pmf_grid.data != nullptr) {
			// sample_pmf_grid() already computes both .gradient (for force)
			// and .value (for energy) in a single grid interpolation, so
			// tracking energy here costs nothing extra - unlike the pairwise/
			// bonded kernels' get_energy, this isn't gating an extra atomic.
			const GridSample<float> sample = pmf_detail::sample_pmf_grid(pmf_grid, pos, scheme);
			const float scale = types.pmf_scale[type_id];
			force += sample.gradient * (-scale);
			if (get_energy) {
				force.t += scale * sample.value;
			}
		}
	}

	const int3 force_grid_id = types.force_grid_id[type_id];
	if (grid_configs != nullptr) {
		if (force_grid_id.x >= 0) {
			const BaseGridView<float>& grid = grid_configs[force_grid_id.x];
			if (grid.data != nullptr) {
				force.x += pmf_detail::sample_force_grid_value(grid, pos, scheme);
			}
		}
		if (force_grid_id.y >= 0) {
			const BaseGridView<float>& grid = grid_configs[force_grid_id.y];
			if (grid.data != nullptr) {
				force.y += pmf_detail::sample_force_grid_value(grid, pos, scheme);
			}
		}
		if (force_grid_id.z >= 0) {
			const BaseGridView<float>& grid = grid_configs[force_grid_id.z];
			if (grid.data != nullptr) {
				force.z += pmf_detail::sample_force_grid_value(grid, pos, scheme);
			}
		}
	}

	return force;
}

struct ComputePMFKernel {
	float electric_field = 0.0f;
	int scheme = 1;
	bool get_energy = false;

	KERNEL_FUNC void operator()(size_t i,
								const Vector3* __restrict__ positions,
								const int* __restrict__ type_ids,
								Vector3* __restrict__ forces,
								const ParticleTypeView types,
								const BaseGridView<float>* __restrict__ grid_configs,
								idx_t num_particles) const {

		const idx_t idx = static_cast<idx_t>(i);
		if (idx >= num_particles)
			return;

		const int type_id = type_ids[idx];
		const Vector3 pos = positions[idx];

		// One thread per particle - forces[idx] is only ever written by this
		// thread, so this is a plain read-modify-write, not an atomic like
		// the pairwise/bonded kernels' get_energy accumulation.
		forces[idx] += compute_position_dependent_force(
			pos, type_id, types, grid_configs, electric_field, scheme, get_energy);
	}
};

} // namespace ARBD

// Explicit template instantiation declaration to prevent host instantiation.
// Unlike the launch_* helpers elsewhere, this kernel is launched with a
// trailing argument pack (see Patch::calculate_nonbonded_forces), so the
// declaration must spell out those arguments exactly as launch_kernel
// forwards them through get_buffer_pointer(). Real definition lives in
// Nonbonded/NonbondedInstantiations.cu.
#ifdef USE_CUDA
#include "Backend/CUDA/KernelHelper.cuh"
namespace ARBD {
extern template Event launch_cuda_kernel(const Resource& resource,
										 const KernelConfig& config,
										 ComputePMFKernel kernel_func,
										 Vector3* positions,
										 int* type_ids,
										 Vector3* forces,
										 ParticleTypeView types,
										 const BaseGridView<float>* grid_configs,
										 idx_t num_particles);
} // namespace ARBD
#endif

#ifdef USE_SYCL
#include <sycl/sycl.hpp>
template<>
struct sycl::is_device_copyable<ARBD::ComputePMFKernel> : std::true_type {};
#endif
