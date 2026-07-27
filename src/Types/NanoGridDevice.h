/*********************************************************************
 * @file  NanoGridDevice.h
 *
 * @brief Device-safe sparse (PNanoVDB) grid sampling, mirroring the shape
 *        of BaseGridDevice.h's dense sampling functions.
 *
 * @details Built only on Types/PNanoVDB.h (plain C99, no nanovdb C++
 *          template stack), so this header compiles cleanly under CUDA
 *          and SYCL device passes alike. Grid *building* (dense -> sparse)
 *          happens offline; this file only covers the runtime read path.
 *********************************************************************/
#pragma once

#include "BaseGridDevice.h" // reuses catmull_rom_value/catmull_rom_deriv
#include "Header.h"
#include "Types.h"
#include <type_traits>

#ifndef PNANOVDB_C
#define PNANOVDB_C
#endif
#include "PNanoVDB.h"

namespace ARBD {

/*===================*\
|  VEC3 CONVERSIONS   |
\*===================*/

template<typename T>
HOST DEVICE inline pnanovdb_vec3_t to_pnano_vec3(const Vector3_t<T>& v) {
	pnanovdb_vec3_t out;
	out.x = static_cast<float>(v.x);
	out.y = static_cast<float>(v.y);
	out.z = static_cast<float>(v.z);
	return out;
}

template<typename T>
HOST DEVICE inline Vector3_t<T> from_pnano_vec3(const pnanovdb_vec3_t& v) {
	return Vector3_t<T>(static_cast<T>(v.x), static_cast<T>(v.y), static_cast<T>(v.z));
}

/*=======================*\
|  GRID SAMPLING CONTEXT   |
\*=======================*/

/**
 * @brief Bundles the buffer, grid handle, grid type, and a (mutable, cached)
 *        read accessor needed to sample a PNanoVDB grid - the sparse
 *        counterpart of BaseGridView<T>.
 * @note acc is mutated by sampling calls (leaf/lower/upper cache); reusing
 *       the same context across nearby sample points is what makes the
 *       cache useful.
 */
struct NanoGridContext {
	pnanovdb_buf_t buf;
	pnanovdb_grid_handle_t grid;
	pnanovdb_grid_type_t grid_type;
	pnanovdb_readaccessor_t acc;
};

/**
 * @brief Initialize a NanoGridContext for a grid stored at the start of buf.
 */
HOST DEVICE inline NanoGridContext make_nano_grid_context(pnanovdb_buf_t buf) {
	NanoGridContext ctx;
	ctx.buf = buf;
	ctx.grid.address = pnanovdb_address_null();
	ctx.grid_type =
		static_cast<pnanovdb_grid_type_t>(pnanovdb_grid_get_grid_type(buf, ctx.grid));
	pnanovdb_tree_handle_t tree = pnanovdb_grid_get_tree(buf, ctx.grid);
	pnanovdb_root_handle_t root = pnanovdb_tree_get_root(buf, tree);
	pnanovdb_readaccessor_init(PNANOVDB_REF(ctx.acc), root);
	return ctx;
}

/*==================================*\
|  VALUE READ (type-dispatched)      |
\*==================================*/

template<typename T>
HOST DEVICE inline T read_grid_value(pnanovdb_buf_t buf, pnanovdb_address_t address) {
	if constexpr (std::is_same_v<T, double>) {
		return static_cast<T>(pnanovdb_read_double(buf, address));
	} else {
		return static_cast<T>(pnanovdb_read_float(buf, address));
	}
}

/**
 * @brief Fetch the grid value at an integer index-space coordinate.
 * @note Out-of-tree coordinates transparently return the grid's background
 *       value via the normal accessor lookup - no manual bounds check
 *       needed here. The offline builder must set background = 0 for this
 *       to match the dense path's zero-pad convention.
 */
template<typename T>
HOST DEVICE inline T fetch_nano_value(NanoGridContext& ctx, int ix, int iy, int iz) {
	pnanovdb_coord_t ijk;
	ijk.x = ix;
	ijk.y = iy;
	ijk.z = iz;
	pnanovdb_address_t address = pnanovdb_readaccessor_get_value_address(
		ctx.grid_type, ctx.buf, PNANOVDB_REF(ctx.acc), PNANOVDB_REF(ijk));
	return read_grid_value<T>(ctx.buf, address);
}

/*===============================*\
|  WORLD <-> INDEX SPACE TRANSFORM |
\*===============================*/

/**
 * @brief Transform a gradient from index space to world space via the
 *        inverse-Jacobian-transpose, matching BaseGridDevice.h's
 *        basis_inv.transpose().transform(...) convention. PNanoVDB has no
 *        direct helper for this combination (only forward/inverse Jacobi,
 *        not their transpose), so the 3x3 multiply is done here explicitly
 *        from the map's inverse-matrix components.
 */
template<typename T>
HOST DEVICE inline Vector3_t<T> apply_inverse_jacobi_transpose(pnanovdb_buf_t buf,
																pnanovdb_map_handle_t map,
																const Vector3_t<T>& grad_grid) {
	const float sx = static_cast<float>(grad_grid.x);
	const float sy = static_cast<float>(grad_grid.y);
	const float sz = static_cast<float>(grad_grid.z);

	Vector3_t<T> out;
	out.x = static_cast<T>(sx * pnanovdb_map_get_invmatf(buf, map, 0) +
						   sy * pnanovdb_map_get_invmatf(buf, map, 3) +
						   sz * pnanovdb_map_get_invmatf(buf, map, 6));
	out.y = static_cast<T>(sx * pnanovdb_map_get_invmatf(buf, map, 1) +
						   sy * pnanovdb_map_get_invmatf(buf, map, 4) +
						   sz * pnanovdb_map_get_invmatf(buf, map, 7));
	out.z = static_cast<T>(sx * pnanovdb_map_get_invmatf(buf, map, 2) +
						   sy * pnanovdb_map_get_invmatf(buf, map, 5) +
						   sz * pnanovdb_map_get_invmatf(buf, map, 8));
	return out;
}

/*=========================*\
|  TRILINEAR JOINT SAMPLE    |
\*=========================*/

/**
 * @brief Joint value+gradient sample via analytic trilinear interpolation.
 * @details Ported from nanovdb::math::BoxStencil::interpolation()/gradient()
 *          (extern/openvdb/nanovdb/nanovdb/math/Stencils.h), rewritten
 *          against PNanoVDB's accessor instead of GridT::AccessorType. This
 *          is the *analytic* derivative of the trilinear interpolant, unlike
 *          BaseGridDevice.h's compute_gradient (finite-difference) - the two
 *          are not bit-identical, though they agree in the smooth limit.
 */
template<typename T>
HOST DEVICE GridSample<T> sample_grid_linear(NanoGridContext& ctx, const Vector3_t<T>& world_pos) {
	const pnanovdb_vec3_t world_pnano = to_pnano_vec3(world_pos);
	const pnanovdb_vec3_t index_pos =
		pnanovdb_grid_world_to_indexf(ctx.buf, ctx.grid, PNANOVDB_REF(world_pnano));

	const int hx = static_cast<int>(math::floor(index_pos.x));
	const int hy = static_cast<int>(math::floor(index_pos.y));
	const int hz = static_cast<int>(math::floor(index_pos.z));
	const T u = static_cast<T>(index_pos.x) - static_cast<T>(hx);
	const T v = static_cast<T>(index_pos.y) - static_cast<T>(hy);
	const T w = static_cast<T>(index_pos.z) - static_cast<T>(hz);

	const T v000 = fetch_nano_value<T>(ctx, hx, hy, hz);
	const T v001 = fetch_nano_value<T>(ctx, hx, hy, hz + 1);
	const T v010 = fetch_nano_value<T>(ctx, hx, hy + 1, hz);
	const T v011 = fetch_nano_value<T>(ctx, hx, hy + 1, hz + 1);
	const T v100 = fetch_nano_value<T>(ctx, hx + 1, hy, hz);
	const T v101 = fetch_nano_value<T>(ctx, hx + 1, hy, hz + 1);
	const T v110 = fetch_nano_value<T>(ctx, hx + 1, hy + 1, hz);
	const T v111 = fetch_nano_value<T>(ctx, hx + 1, hy + 1, hz + 1);

	// Value: trilinear blend along z, then y, then x
	T A = v000 + (v001 - v000) * w;
	T B = v010 + (v011 - v010) * w;
	const T C = A + (B - A) * v;

	A = v100 + (v101 - v100) * w;
	B = v110 + (v111 - v110) * w;
	const T D = A + (B - A) * v;

	GridSample<T> result;
	result.value = C + (D - C) * u;

	// Gradient (index space), analytic derivative of the same trilinear blend
	const T d0 = v001 - v000;
	const T d1 = v011 - v010;
	const T d2 = v101 - v100;
	const T d3 = v111 - v110;

	A = d0 + (d1 - d0) * v;
	B = d2 + (d3 - d2) * v;
	Vector3_t<T> grad_grid(T{0}, T{0}, T{0});
	grad_grid.z = A + (B - A) * u;

	const T e0 = v000 + d0 * w;
	const T e1 = v010 + d1 * w;
	const T e2 = v100 + d2 * w;
	const T e3 = v110 + d3 * w;

	A = e0 + (e1 - e0) * v;
	B = e2 + (e3 - e2) * v;
	grad_grid.x = B - A;

	A = e1 - e0;
	B = e3 - e2;
	grad_grid.y = A + (B - A) * u;

	pnanovdb_map_handle_t map = pnanovdb_grid_get_map(ctx.buf, ctx.grid);
	result.gradient = apply_inverse_jacobi_transpose(ctx.buf, map, grad_grid);
	return result;
}

/*======================================*\
|  CUBIC (CATMULL-ROM) JOINT SAMPLE       |
\*======================================*/

/**
 * @brief Joint value+gradient sample via cubic (Catmull-Rom) interpolation.
 * @details Same coefficient math as BaseGridDevice.h::sample_grid_cubic
 *          (reused via catmull_rom_value/catmull_rom_deriv), with the 64
 *          neighbor taps fetched through the PNanoVDB accessor instead of
 *          dense-array indexing. No manual bounds checking: out-of-tree taps
 *          resolve to the grid's background value (must be 0, see
 *          fetch_nano_value's note).
 */
template<typename T>
HOST DEVICE GridSample<T> sample_grid_cubic(NanoGridContext& ctx, const Vector3_t<T>& world_pos) {
	const pnanovdb_vec3_t world_pnano = to_pnano_vec3(world_pos);
	const pnanovdb_vec3_t index_pos =
		pnanovdb_grid_world_to_indexf(ctx.buf, ctx.grid, PNANOVDB_REF(world_pnano));

	const int homeX = static_cast<int>(math::floor(index_pos.x));
	const int homeY = static_cast<int>(math::floor(index_pos.y));
	const int homeZ = static_cast<int>(math::floor(index_pos.z));
	const T wx = static_cast<T>(index_pos.x) - static_cast<T>(homeX);
	const T wy = static_cast<T>(index_pos.y) - static_cast<T>(homeY);
	const T wz = static_cast<T>(index_pos.z) - static_cast<T>(homeZ);

	// Stage 1: blend along x at each of the 4x4 (iy, iz) taps
	T dVdx_stage1[4][4];
	T V_stage1[4][4];
	for (int iz = 0; iz < 4; ++iz) {
		const int jz = homeZ - 1 + iz;
		for (int iy = 0; iy < 4; ++iy) {
			const int jy = homeY - 1 + iy;
			T v[4];
			for (int ix = 0; ix < 4; ++ix) {
				const int jx = homeX - 1 + ix;
				v[ix] = fetch_nano_value<T>(ctx, jx, jy, jz);
			}
			dVdx_stage1[iy][iz] = catmull_rom_deriv(v[0], v[1], v[2], v[3], wx);
			V_stage1[iy][iz] = catmull_rom_value(v[0], v[1], v[2], v[3], wx);
		}
	}

	// Stage 2: blend along y at each of the 4 iz taps
	T dVdx_stage2[4];
	T dVdy_stage2[4];
	T V_stage2[4];
	for (int iz = 0; iz < 4; ++iz) {
		dVdx_stage2[iz] = catmull_rom_value(
			dVdx_stage1[0][iz], dVdx_stage1[1][iz], dVdx_stage1[2][iz], dVdx_stage1[3][iz], wy);
		dVdy_stage2[iz] = catmull_rom_deriv(
			V_stage1[0][iz], V_stage1[1][iz], V_stage1[2][iz], V_stage1[3][iz], wy);
		V_stage2[iz] = catmull_rom_value(
			V_stage1[0][iz], V_stage1[1][iz], V_stage1[2][iz], V_stage1[3][iz], wy);
	}

	// Stage 3: blend along z
	Vector3_t<T> grad_grid(
		catmull_rom_value(dVdx_stage2[0], dVdx_stage2[1], dVdx_stage2[2], dVdx_stage2[3], wz),
		catmull_rom_value(dVdy_stage2[0], dVdy_stage2[1], dVdy_stage2[2], dVdy_stage2[3], wz),
		catmull_rom_deriv(V_stage2[0], V_stage2[1], V_stage2[2], V_stage2[3], wz));

	GridSample<T> result;
	result.value = catmull_rom_value(V_stage2[0], V_stage2[1], V_stage2[2], V_stage2[3], wz);

	pnanovdb_map_handle_t map = pnanovdb_grid_get_map(ctx.buf, ctx.grid);
	result.gradient = apply_inverse_jacobi_transpose(ctx.buf, map, grad_grid);
	return result;
}

} // namespace ARBD

#ifdef USE_SYCL
#include <sycl/sycl.hpp>
template<>
struct sycl::is_device_copyable<ARBD::NanoGridContext> : std::true_type {};
#endif
