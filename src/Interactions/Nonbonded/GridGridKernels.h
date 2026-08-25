#pragma once
#include "Types/BaseGridDevice.h"
#include "Types/Types.h"

namespace MARS {

namespace gridgrid_detail {

/**
 * @brief Force+energy+torque contribution from one rho (density) voxel
 * sampled through u's (potential) field, ported from legacy
 * RigidBody's ComputeGridGrid.cu common_computeGridGridForce.
 *
 * rho/u grids are ordinary BaseGridView<mars_real> (see Types/BaseGridDevice.h) -
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
HOST DEVICE inline void grid_grid_voxel_force_torque(const BaseGridView<mars_real>& rho,
													 const BaseGridView<mars_real>& u,
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
	const GridSample<mars_real> sample = (scheme == 0) ? sample_grid_linear(u.data,
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
	// Legacy ComputeGridGrid.cu is wrong here; see GridGridKernels.md.
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
								const BaseGridView<mars_real> rho,
								const BaseGridView<mars_real> u,
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

/// @brief grid[i] = 0
template<typename T>
struct ZeroGridKernel {
	BaseGridMutableView<T> grid;

	KERNEL_FUNC void operator()(idx_t i) const {
		if (i >= grid.size())
			return;
		grid.set(i, T{0});
	}
};

/// @brief grid[i] *= factor
template<typename T>
struct ScaleGridKernel {
	BaseGridMutableView<T> grid;
	T factor;

	KERNEL_FUNC void operator()(idx_t i) const {
		if (i >= grid.size())
			return;
		grid.set(i, grid.get(i) * factor);
	}
};

/// @brief grid[i] += value
template<typename T>
struct ShiftGridKernel {
	BaseGridMutableView<T> grid;
	T value;

	KERNEL_FUNC void operator()(idx_t i) const {
		if (i >= grid.size())
			return;
		grid.set(i, grid.get(i) + value);
	}
};

/**
 * @brief dest[i] *= src[i], elementwise
 * @note Callers must check that dest and src agree in dimensions; the kernel
 *       only guards its own bounds, mirroring how BaseGrid::multiply leaves the
 *       size check to its own host-side precondition.
 */
template<typename T>
struct MultiplyGridKernel {
	BaseGridMutableView<T> dest;
	BaseGridView<T> src;

	KERNEL_FUNC void operator()(idx_t i) const {
		if (i >= dest.size())
			return;
		dest.set(i, dest.get(i) * src[i]);
	}
};

/**
 * @brief Real-space convolution out = density (*) kernel
 * @details One thread per output voxel, gathering over the kernel's stencil.
 *          Taps use the density's own boundary condition, so a periodic density
 *          reproduces the circular convolution an FFT would give.
 * @param scale Applied to each output voxel; 1 sums raw voxel products, the
 *        density's cell volume gives the volume integral.
 * @note The kernel is centered on voxel (n-1)/2 per axis. Even dimensions shift
 *       the result half a voxel - same caveat as arbdmodel's convolve helper.
 */
template<typename T>
struct ConvolveGridKernel {
	BaseGridView<T> density;
	BaseGridView<T> kernel;
	BaseGridMutableView<T> out;
	T scale;

	KERNEL_FUNC void operator()(idx_t i) const {
		if (i >= out.size())
			return;

		const idx_t onz = out.nz();
		const idx_t ony = out.ny();
		const int iz = static_cast<int>(i % onz);
		const int iy = static_cast<int>((i / onz) % ony);
		const int ix = static_cast<int>(i / (onz * ony));

		const int kx = static_cast<int>(kernel.nx());
		const int ky = static_cast<int>(kernel.ny());
		const int kz = static_cast<int>(kernel.nz());
		const int cx = (kx - 1) / 2;
		const int cy = (ky - 1) / 2;
		const int cz = (kz - 1) / 2;

		T sum = T{0};
		for (int a = 0; a < kx; ++a) {
			for (int b = 0; b < ky; ++b) {
				for (int c = 0; c < kz; ++c) {
					const T kv = kernel.data[c + b * kz + a * ky * kz];
					if (kv == T{0})
						continue;
					// K is indexed at (x - y + c), i.e. the density tap runs
					// backwards - convolution, not correlation. Identical for
					// symmetric kernels, mirrored for asymmetric ones.
					sum += kv * fetch_grid_value(density.data,
												 ix + cx - a,
												 iy + cy - b,
												 iz + cz - c,
												 density.dimensions,
												 density.boundary_condition);
				}
			}
		}
		out.set(i, sum * scale);
	}
};

} // namespace MARS

// Explicit template instantiation declaration to prevent host instantiation
// (see Pmf.h for why - real definition lives in
// Nonbonded/NonbondedInstantiations.cu).
#ifdef USE_CUDA
#include "Backend/CUDA/KernelHelper.cuh"
namespace MARS {
extern template Event launch_cuda_kernel_with_workitem(const Resource& resource,
													   const KernelConfig& config,
													   ComputeGridGridForceKernel kernel_func,
													   const BaseGridView<mars_real> rho,
													   const BaseGridView<mars_real> u,
													   Vector3* ret_force_energy,
													   Vector3* ret_torque);

extern template Event launch_cuda_kernel(const Resource& resource,
										 const KernelConfig& config,
										 ZeroGridKernel<mars_real> kernel_func);
extern template Event launch_cuda_kernel(const Resource& resource,
										 const KernelConfig& config,
										 ScaleGridKernel<mars_real> kernel_func);
extern template Event launch_cuda_kernel(const Resource& resource,
										 const KernelConfig& config,
										 ShiftGridKernel<mars_real> kernel_func);
extern template Event launch_cuda_kernel(const Resource& resource,
										 const KernelConfig& config,
										 MultiplyGridKernel<mars_real> kernel_func);
extern template Event launch_cuda_kernel(const Resource& resource,
										 const KernelConfig& config,
										 ConvolveGridKernel<mars_real> kernel_func);
} // namespace MARS
#endif

#ifdef USE_SYCL
#include <sycl/sycl.hpp>
template<>
struct sycl::is_device_copyable<MARS::ComputeGridGridForceKernel> : std::true_type {};
template<typename T>
struct sycl::is_device_copyable<MARS::ZeroGridKernel<T>> : std::true_type {};
template<typename T>
struct sycl::is_device_copyable<MARS::ScaleGridKernel<T>> : std::true_type {};
template<typename T>
struct sycl::is_device_copyable<MARS::ShiftGridKernel<T>> : std::true_type {};
template<typename T>
struct sycl::is_device_copyable<MARS::MultiplyGridKernel<T>> : std::true_type {};
template<typename T>
struct sycl::is_device_copyable<MARS::ConvolveGridKernel<T>> : std::true_type {};
#endif
