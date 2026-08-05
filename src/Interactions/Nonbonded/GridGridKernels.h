#pragma once
#include "Types/BaseGridDevice.h"
#include "Types/Types.h"

namespace ARBD {

namespace gridgrid_detail {

/**
 * @brief Force+energy+torque contribution from one rho (density) voxel
 * sampled through u's (potential) field, ported from legacy
 * RigidBody's ComputeGridGrid.cu common_computeGridGridForce.
 *
 * rho/u grids are ordinary BaseGridView<float> (see Types/BaseGridDevice.h) -
 * unlike legacy's separate local-frame-only RigidBodyGrid, no extra grid type
 * is needed here. The lab-frame transform (rigid body orientation/position
 * combined with each grid's own static local origin/basis) is computed once
 * per pair per step by the caller (see architecture decision #1 in
 * arbd2v/plan.md) and passed in as basis_rho/basis_u_inv/origin_offset, so
 * this function only needs rho/u's raw data+dimensions - not their own
 * origin/basis fields, which are folded into those precomputed arguments.
 *
 * basis_rho: lab-frame basis of rho (rb1.orientation * rho_grid.basis)
 * basis_u_inv: inverse lab-frame basis of u ((rb2.orientation * u_grid.basis).inverse())
 * origin_offset: origin_rho_minus_origin_u, both origins already transformed
 *   to lab frame (rbN.orientation * gridN.origin + rbN.position)
 *
 * Interpolation is done in u's raw index space by calling sample_grid_linear/
 * sample_grid_cubic with an identity origin/basis - those functions normally
 * transform a world position into grid-index space via
 * basis_inv.transform(world_pos - origin), so passing identity/zero there
 * makes u_local (already computed in index space below) pass through
 * unchanged, exactly matching legacy's RigidBodyGrid semantics without
 * duplicating the interpolation math.
 *
 * No periodic wrapping is applied here (legacy's GridPositionTransformer vs.
 * PmfPositionTransformer only differ by a wrapDiff() call) - this matches
 * PmfPositionTransformer's plain `pos + o`. Periodic RB-RB pairs are future
 * work (see arbd2v/plan.md Phase 3+); this function only needs the offset
 * updated to include a wrapDiff() against a PeriodicBox when that lands.
 */
HOST DEVICE inline void grid_grid_voxel_force_torque(const BaseGridView<float>& rho,
													 const BaseGridView<float>& u,
													 const Matrix3& basis_rho,
													 const Matrix3& basis_u_inv,
													 const Vector3& origin_offset,
													 idx_t r_id,
													 int scheme,
													 Vector3& force_energy_out,
													 Vector3& torque_out) {
	const idx_t nz = rho.nz();
	const idx_t ny = rho.ny();
	const idx_t iz = r_id % nz;
	const idx_t iy = (r_id / nz) % ny;
	const idx_t ix = r_id / (nz * ny);

	Vector3 r_pos = basis_rho.transform(Vector3(float(ix), float(iy), float(iz)));
	const Vector3 u_local = basis_u_inv.transform(r_pos + origin_offset);

	const Matrix3 identity(1.0f);
	const GridSample<float> sample = (scheme == 0) ? sample_grid_linear(u.data,
																		u_local,
																		Vector3(0.0f),
																		identity,
																		identity,
																		u.dimensions,
																		u.boundary_condition)
												   : sample_grid_cubic(u.data,
																	   u_local,
																	   Vector3(0.0f),
																	   identity,
																	   identity,
																	   u.dimensions,
																	   u.boundary_condition);

	const float r_val = rho.data[r_id];
	// sample.gradient is raw (∂V/∂x); force = -gradient, then transform from
	// u's index space to lab frame via basis_u_inv^T, matching legacy's
	// basisInv.transpose().transform(r_val * fe.f).
	const Vector3 force_lab = basis_u_inv.transpose().transform(r_val * (-sample.gradient));

	force_energy_out = force_lab;
	// Fixed vs. legacy: ComputeGridGrid.cu's reduction sets force[tid].e =
	// r_val (dropping the sampled potential factor). The per-voxel energy
	// contribution to E = sum_voxels rho_i * u(x_i) is density * potential.
	force_energy_out.t = r_val * sample.value;

	// Torque about rho's own origin (r_pos is the offset from origin_rho, see
	// origin_offset's doc comment above), in lab frame.
	torque_out = r_pos.cross(force_lab);
}

} // namespace gridgrid_detail

/**
 * @brief Grid-grid force/torque kernel: one thread per rho voxel, block-wide
 * shared-memory reduction, atomicAdd into a single (force+energy, torque)
 * accumulator per grid pair.
 * Caller must set KernelConfig::shared_memory = 2 * block_size.x *
 * sizeof(Vector3) (force+energy array and torque array, one Vector3 each per
 * thread) and block_size.x to a power of two (legacy used 128).
 */
struct ComputeGridGridForceKernel {
	Matrix3 basis_rho;
	Matrix3 basis_u_inv;
	Vector3 origin_offset; // origin_rho_minus_origin_u, lab frame
	int scheme = 1;
	idx_t block_size = 128; // legacy used 128

	template<typename WorkItemT>
	KERNEL_FUNC void operator()(size_t i,
								WorkItemT& item,
								const BaseGridView<float> rho,
								const BaseGridView<float> u,
								Vector3* __restrict__ ret_force_energy,
								Vector3* __restrict__ ret_torque) const {
		Vector3* force = item.template get_shared_mem<Vector3>(0);
		Vector3* torque = item.template get_shared_mem<Vector3>(block_size * sizeof(Vector3));

		const idx_t tid = item.local_id();
		const idx_t r_id = i;

		force[tid] = Vector3(0.0f);
		torque[tid] = Vector3(0.0f);

		if (r_id < rho.size()) {
			gridgrid_detail::grid_grid_voxel_force_torque(rho,
														  u,
														  basis_rho,
														  basis_u_inv,
														  origin_offset,
														  r_id,
														  scheme,
														  force[tid],
														  torque[tid]);
		}

		item.barrier();
		for (idx_t offset = block_size / 2; offset > 0; offset >>= 1) {
			if (tid < offset) {
				// Vector3_t::operator+= only adds x/y/z (see Types/Vector3.h) -
				// it's not meant for accumulating the .t slot, which is exactly
				// where energy is packed here, so it needs an explicit add.
				// (torque[tid].t is always 0 and unused, but summed too for
				// consistency/robustness against future use.)
				force[tid] += force[tid + offset];
				force[tid].t += force[tid + offset].t; // energy
				torque[tid] += torque[tid + offset];
				torque[tid].t += torque[tid + offset].t; // torque energy
			}
			item.barrier();
		}

		if (tid == 0) {
			atomic_add(ret_force_energy, force[0]);
			atomic_add(ret_torque, torque[0]);
		}
	}
};

} // namespace ARBD

// Explicit template instantiation declaration to prevent host instantiation
// (see PmfKernels.h for why - real definition lives in
// Nonbonded/NonbondedInstantiations.cu).
#ifdef USE_CUDA
#include "Backend/CUDA/KernelHelper.cuh"
namespace ARBD {
extern template Event launch_cuda_kernel_with_workitem(const Resource& resource,
													   const KernelConfig& config,
													   ComputeGridGridForceKernel kernel_func,
													   const BaseGridView<float> rho,
													   const BaseGridView<float> u,
													   Vector3* ret_force_energy,
													   Vector3* ret_torque);
} // namespace ARBD
#endif

#ifdef USE_SYCL
#include <sycl/sycl.hpp>
template<>
struct sycl::is_device_copyable<ARBD::ComputeGridGridForceKernel> : std::true_type {};
#endif
