#pragma once
#include "GridTerm.h"
#include "Header.h"
#include "IndexList.h"
#include "Math.h"
#include "Matrix3.h"
#include "Types.h"
#include "Vector3.h"

namespace ARBD {

/**
 * @brief Immutable device-side view of grid data
 * @details Lightweight POD struct safe for passing to CUDA/SYCL kernels by value.
 *          Contains only raw pointers and trivially copyable types.
 *          All operations are const and device-safe.
 *	@brief Boundary condition types for grid operations
 *
 * @tparam T Grid value type (float/double)
 */
template<typename T>
struct BaseGridView {
	// POD members only - safe for kernel parameters
	CONSTANT_PTR(T) __restrict__ data; ///< Raw pointer to device grid data
	Vector3_t<T> origin;			   ///< Grid origin in world space
	Matrix3_t<T> basis;				   ///< Basis vectors (grid spacing)
	Matrix3_t<T> basis_inv;			   ///< Cached inverse basis
	Vector3_t<idx_t> dimensions;	   ///< Grid dimensions (nx, ny, nz)
	int grid_id;
	int boundary_condition;
	// 1- Dirichlet, 2- Neumann, 3- Periodic
	/*===================*\
	|  INDEX OPERATIONS   |
	\*===================*/

	/**
	 * @brief Convert 3D indices to linear index
	 */
	HOST DEVICE constexpr idx_t index(idx_t ix, idx_t iy, idx_t iz) const noexcept {
		return iz + iy * dimensions.z + ix * dimensions.y * dimensions.z;
	}

	/**
	 * @brief Get grid dimensions
	 */
	HOST DEVICE constexpr idx_t nx() const noexcept {
		return dimensions.x;
	}
	HOST DEVICE constexpr idx_t ny() const noexcept {
		return dimensions.y;
	}
	HOST DEVICE constexpr idx_t nz() const noexcept {
		return dimensions.z;
	}
	HOST DEVICE constexpr idx_t size() const noexcept {
		return dimensions.x * dimensions.y * dimensions.z;
	}

	/*===================*\
	|  VALUE ACCESS       |
	\*===================*/

	/**
	 * @brief Direct indexed access (no bounds checking)
	 */
	HOST DEVICE constexpr T operator[](idx_t linear_idx) const noexcept {
		return data[linear_idx];
	}

	HOST DEVICE constexpr T operator()(idx_t ix, idx_t iy, idx_t iz) const noexcept {
		return data[index(ix, iy, iz)];
	}

	/*===================*\
	|  SPATIAL QUERIES    |
	\*===================*/

	/**
	 * @brief Transform world position to grid coordinates
	 */
	HOST DEVICE Vector3_t<T> world_to_grid(const Vector3_t<T>& world_pos) const noexcept {
		return basis_inv.transform(world_pos - origin);
	}

	/**
	 * @brief Transform grid coordinates to world position
	 */
	HOST DEVICE Vector3_t<T> grid_to_world(const Vector3_t<T>& grid_pos) const noexcept {
		return basis.transform(grid_pos) + origin;
	}

	/**
	 * @brief Trilinear interpolation at world position
	 */
	HOST DEVICE T interpolate(const Vector3_t<T>& world_pos) const noexcept {
		const Vector3_t<T> grid_pos = world_to_grid(world_pos);

		// Clamp to grid bounds
		const T gx = clamp(grid_pos.x, T{0}, static_cast<T>(nx() - 1) - T{0.001});
		const T gy = clamp(grid_pos.y, T{0}, static_cast<T>(ny() - 1) - T{0.001});
		const T gz = clamp(grid_pos.z, T{0}, static_cast<T>(nz() - 1) - T{0.001});

		const idx_t i0 = static_cast<idx_t>(gx);
		const idx_t j0 = static_cast<idx_t>(gy);
		const idx_t k0 = static_cast<idx_t>(gz);
		const idx_t i1 = i0 + 1;
		const idx_t j1 = j0 + 1;
		const idx_t k1 = k0 + 1;

		const T fx = gx - static_cast<T>(i0);
		const T fy = gy - static_cast<T>(j0);
		const T fz = gz - static_cast<T>(k0);

		// Trilinear interpolation
		const T c000 = (*this)(i0, j0, k0);
		const T c100 = (*this)(i1, j0, k0);
		const T c010 = (*this)(i0, j1, k0);
		const T c110 = (*this)(i1, j1, k0);
		const T c001 = (*this)(i0, j0, k1);
		const T c101 = (*this)(i1, j0, k1);
		const T c011 = (*this)(i0, j1, k1);
		const T c111 = (*this)(i1, j1, k1);

		const T c00 = c000 * (T{1} - fx) + c100 * fx;
		const T c10 = c010 * (T{1} - fx) + c110 * fx;
		const T c01 = c001 * (T{1} - fx) + c101 * fx;
		const T c11 = c011 * (T{1} - fx) + c111 * fx;

		const T c0 = c00 * (T{1} - fy) + c10 * fy;
		const T c1 = c01 * (T{1} - fy) + c11 * fy;

		return c0 * (T{1} - fz) + c1 * fz;
	}

	/**
	 * @brief Nearest neighbor lookup
	 */
	HOST DEVICE T nearest(const Vector3_t<T>& world_pos) const noexcept {
		const Vector3_t<T> grid_pos = world_to_grid(world_pos);
		const idx_t ix = static_cast<idx_t>(grid_pos.x + T{0.5});
		const idx_t iy = static_cast<idx_t>(grid_pos.y + T{0.5});
		const idx_t iz = static_cast<idx_t>(grid_pos.z + T{0.5});

		if (ix >= nx() || iy >= ny() || iz >= nz()) {
			return T{0}; // Out of bounds
		}

		return (*this)(ix, iy, iz);
	}

	/**
	 * @brief Compute gradient at world position (central differences)
	 */
	HOST DEVICE Vector3_t<T> gradient(const Vector3_t<T>& world_pos) const noexcept {
		const Vector3_t<T> grid_pos = world_to_grid(world_pos);

		// Check bounds for gradient calculation (need neighbors)
		if (grid_pos.x < T{1} || grid_pos.x >= static_cast<T>(nx() - 1) || grid_pos.y < T{1} ||
			grid_pos.y >= static_cast<T>(ny() - 1) || grid_pos.z < T{1} ||
			grid_pos.z >= static_cast<T>(nz() - 1)) {
			return Vector3_t<T>{T{0}, T{0}, T{0}};
		}

		const idx_t i = static_cast<idx_t>(grid_pos.x);
		const idx_t j = static_cast<idx_t>(grid_pos.y);
		const idx_t k = static_cast<idx_t>(grid_pos.z);

		// Central differences in grid space
		const T dx_grid = ((*this)(i + 1, j, k) - (*this)(i - 1, j, k)) / T{2};
		const T dy_grid = ((*this)(i, j + 1, k) - (*this)(i, j - 1, k)) / T{2};
		const T dz_grid = ((*this)(i, j, k + 1) - (*this)(i, j, k - 1)) / T{2};

		// Transform gradient from grid space to world space
		const Vector3_t<T> grad_grid(dx_grid, dy_grid, dz_grid);
		return basis_inv.transpose().transform(grad_grid);
	}

  private:
	HOST DEVICE static constexpr T clamp(T val, T min_val, T max_val) noexcept {
		return val < min_val ? min_val : (val > max_val ? max_val : val);
	}
};

template<typename T>
struct ScaleGrid {
	HOST DEVICE void operator()(T scale, T* grid_values) const {
		grid_values = grid_values * scale;
	}
};

/**
 * @brief Device-safe interpolation function (CUDA/SYCL compatible)
 */
template<typename T>
struct InterpolateGridPoint {
	T operator()(CONSTANT_PTR(T) __restrict__ grid_values,
				 const Vector3_t<T>& world_pos,
				 const Vector3_t<T>& origin,
				 const Matrix3_t<T>& basis_inv,
				 const Vector3_t<idx_t>& dimensions,
				 int boundary_condition) const {
		// Transform world position to grid coordinates
		const Vector3_t<T> grid_pos = basis_inv.transform(world_pos - origin);

		const idx_t nx = dimensions.x;
		const idx_t ny = dimensions.y;
		const idx_t nz = dimensions.z;

		// Check bounds
		if (grid_pos.x < 0 || grid_pos.x >= nx || grid_pos.y < 0 || grid_pos.y >= ny ||
			grid_pos.z < 0 || grid_pos.z >= nz) {
			return T{0};
		}

		// Linear interpolation
		const idx_t i0 = static_cast<idx_t>(grid_pos.x);
		const idx_t j0 = static_cast<idx_t>(grid_pos.y);
		const idx_t k0 = static_cast<idx_t>(grid_pos.z);

		const idx_t i1 = i0 + 1;
		const idx_t j1 = j0 + 1;
		const idx_t k1 = k0 + 1;

		const T fx = grid_pos.x - static_cast<T>(i0);
		const T fy = grid_pos.y - static_cast<T>(j0);
		const T fz = grid_pos.z - static_cast<T>(k0);

		// Get grid indices using consistent indexing
		const idx_t idx000 = k0 + j0 * nz + i0 * ny * nz;
		const idx_t idx001 = k1 + j0 * nz + i0 * ny * nz;
		const idx_t idx010 = k0 + j1 * nz + i0 * ny * nz;
		const idx_t idx011 = k1 + j1 * nz + i0 * ny * nz;
		const idx_t idx100 = k0 + j0 * nz + i1 * ny * nz;
		const idx_t idx101 = k1 + j0 * nz + i1 * ny * nz;
		const idx_t idx110 = k0 + j1 * nz + i1 * ny * nz;
		const idx_t idx111 = k1 + j1 * nz + i1 * ny * nz;

		// Trilinear interpolation
		const T v000 = grid_values[idx000];
		const T v001 = grid_values[idx001];
		const T v010 = grid_values[idx010];
		const T v011 = grid_values[idx011];
		const T v100 = grid_values[idx100];
		const T v101 = grid_values[idx101];
		const T v110 = grid_values[idx110];
		const T v111 = grid_values[idx111];

		const T v00 = v000 * (T{1} - fx) + v100 * fx;
		const T v01 = v001 * (T{1} - fx) + v101 * fx;
		const T v10 = v010 * (T{1} - fx) + v110 * fx;
		const T v11 = v011 * (T{1} - fx) + v111 * fx;

		const T v0 = v00 * (T{1} - fy) + v10 * fy;
		const T v1 = v01 * (T{1} - fy) + v11 * fy;

		return v0 * (T{1} - fz) + v1 * fz;
	}
};

/**
 * @brief Fetch a grid value at an integer index, or zero if out of bounds.
 *        Applied per-tap; matches the zero-pad convention used by cubic sampling.
 */
template<typename T>
HOST DEVICE inline T fetch_grid_value_or_zero(CONSTANT_PTR(T) __restrict__ grid_values,
											  int jx,
											  int jy,
											  int jz,
											  const Vector3_t<idx_t>& dimensions) {
	if (jx < 0 || jx >= static_cast<int>(dimensions.x) || jy < 0 ||
		jy >= static_cast<int>(dimensions.y) || jz < 0 ||
		jz >= static_cast<int>(dimensions.z)) {
		return T{0};
	}
	const idx_t idx = static_cast<idx_t>(jz) + static_cast<idx_t>(jy) * dimensions.z +
					  static_cast<idx_t>(jx) * dimensions.y * dimensions.z;
	return grid_values[idx];
}

/**
 * @brief Device-safe interpolation function (CUDA/SYCL compatible)
 */
template<typename T>
HOST DEVICE T interpolate_grid_point(CONSTANT_PTR(T) __restrict__ grid_values,
									 const Vector3_t<T>& world_pos,
									 const Vector3_t<T>& origin,
									 const Matrix3_t<T>& basis_inv,
									 const Vector3_t<idx_t>& dimensions,
									 int boundary_condition) {
	// Transform world position to grid coordinates
	const Vector3_t<T> grid_pos = basis_inv.transform(world_pos - origin);

	const idx_t nx = dimensions.x;
	const idx_t ny = dimensions.y;
	const idx_t nz = dimensions.z;

	// Check bounds
	if (grid_pos.x < 0 || grid_pos.x >= nx || grid_pos.y < 0 || grid_pos.y >= ny ||
		grid_pos.z < 0 || grid_pos.z >= nz) {
		return T{0};
	}

	// Linear interpolation - zero-pad missing neighbors at grid edges (legacy convention).
	const int i0 = static_cast<int>(grid_pos.x);
	const int j0 = static_cast<int>(grid_pos.y);
	const int k0 = static_cast<int>(grid_pos.z);

	const T fx = grid_pos.x - static_cast<T>(i0);
	const T fy = grid_pos.y - static_cast<T>(j0);
	const T fz = grid_pos.z - static_cast<T>(k0);

	const T v000 = fetch_grid_value_or_zero(grid_values, i0, j0, k0, dimensions);
	const T v001 = fetch_grid_value_or_zero(grid_values, i0, j0, k0 + 1, dimensions);
	const T v010 = fetch_grid_value_or_zero(grid_values, i0, j0 + 1, k0, dimensions);
	const T v011 = fetch_grid_value_or_zero(grid_values, i0, j0 + 1, k0 + 1, dimensions);
	const T v100 = fetch_grid_value_or_zero(grid_values, i0 + 1, j0, k0, dimensions);
	const T v101 = fetch_grid_value_or_zero(grid_values, i0 + 1, j0, k0 + 1, dimensions);
	const T v110 = fetch_grid_value_or_zero(grid_values, i0 + 1, j0 + 1, k0, dimensions);
	const T v111 = fetch_grid_value_or_zero(grid_values, i0 + 1, j0 + 1, k0 + 1, dimensions);

	const T v00 = v000 * (T{1} - fx) + v100 * fx;
	const T v01 = v001 * (T{1} - fx) + v101 * fx;
	const T v10 = v010 * (T{1} - fx) + v110 * fx;
	const T v11 = v011 * (T{1} - fx) + v111 * fx;

	const T v0 = v00 * (T{1} - fy) + v10 * fy;
	const T v1 = v01 * (T{1} - fy) + v11 * fy;

	return v0 * (T{1} - fz) + v1 * fz;
}

/**
 * @brief Get value at nearest grid point (device-safe)
 */
template<typename T>
HOST DEVICE T get_value_nearest(CONSTANT_PTR(T) __restrict__ grid_values,
								const Vector3_t<T>& world_pos,
								const Vector3_t<T>& origin,
								const Matrix3_t<T>& basis_inv,
								const Vector3_t<idx_t>& dimensions,
								int boundary_condition) {
	// Transform to grid coordinates
	const Vector3_t<T> grid_pos = basis_inv.transform(world_pos - origin);

	// Find nearest grid point
	const idx_t ix = static_cast<idx_t>(grid_pos.x + T{0.5});
	const idx_t iy = static_cast<idx_t>(grid_pos.y + T{0.5});
	const idx_t iz = static_cast<idx_t>(grid_pos.z + T{0.5});

	// Wrap for periodic boundaries (simple modulo)
	const idx_t wrapped_ix = ix % dimensions.x;
	const idx_t wrapped_iy = iy % dimensions.y;
	const idx_t wrapped_iz = iz % dimensions.z;

	const idx_t linear_idx =
		wrapped_iz + wrapped_iy * dimensions.z + wrapped_ix * dimensions.y * dimensions.z;
	return grid_values[linear_idx];
}

/**
 * @brief Compute gradient at a point using finite differences (device-safe)
 */
template<typename T>
HOST DEVICE Vector3_t<T> compute_gradient(CONSTANT_PTR(T) __restrict__ grid_values,
										  const Vector3_t<T>& world_pos,
										  const Vector3_t<T>& origin,
										  const Matrix3_t<T>& basis,
										  const Matrix3_t<T>& basis_inv,
										  const Vector3_t<idx_t>& dimensions,
										  int boundary_condition) {
	const Vector3_t<T> grid_pos = basis_inv.transform(world_pos - origin);

	const idx_t nx = dimensions.x;
	const idx_t ny = dimensions.y;
	const idx_t nz = dimensions.z;

	// Check if we're in bounds for gradient calculation
	if (grid_pos.x < 1 || grid_pos.x >= nx - 1 || grid_pos.y < 1 || grid_pos.y >= ny - 1 ||
		grid_pos.z < 1 || grid_pos.z >= nz - 1) {
		return Vector3_t<T>{T{0}, T{0}, T{0}};
	}

	const idx_t i = static_cast<idx_t>(grid_pos.x);
	const idx_t j = static_cast<idx_t>(grid_pos.y);
	const idx_t k = static_cast<idx_t>(grid_pos.z);

	// Central differences
	const idx_t idx_xp = k + j * nz + (i + 1) * ny * nz;
	const idx_t idx_xm = k + j * nz + (i - 1) * ny * nz;
	const idx_t idx_yp = k + (j + 1) * nz + i * ny * nz;
	const idx_t idx_ym = k + (j - 1) * nz + i * ny * nz;
	const idx_t idx_zp = (k + 1) + j * nz + i * ny * nz;
	const idx_t idx_zm = (k - 1) + j * nz + i * ny * nz;

	const T dx_grid = (grid_values[idx_xp] - grid_values[idx_xm]) / T{2};
	const T dy_grid = (grid_values[idx_yp] - grid_values[idx_ym]) / T{2};
	const T dz_grid = (grid_values[idx_zp] - grid_values[idx_zm]) / T{2};

	// Transform gradient from grid space to world space
	const Vector3_t<T> grad_grid(dx_grid, dy_grid, dz_grid);
	return basis_inv.transpose().transform(grad_grid);
}

/*=========================================*\
|  CUBIC (CATMULL-ROM) JOINT INTERPOLATION   |
\*=========================================*/

/**
 * @brief 1D Catmull-Rom cubic value at parameter t in [0,1] given taps v0..v3
 *        (v1,v2 are the interpolated segment's endpoints; v0,v3 supply tangents).
 */
template<typename T>
HOST DEVICE inline T catmull_rom_value(T v0, T v1, T v2, T v3, T t) {
	const T a3 = T{0.5} * (-v0 + T{3} * v1 - T{3} * v2 + v3);
	const T a2 = T{0.5} * (T{2} * v0 - T{5} * v1 + T{4} * v2 - v3);
	const T a1 = T{0.5} * (-v0 + v2);
	return ((a3 * t + a2) * t + a1) * t + v1;
}

/**
 * @brief Derivative of catmull_rom_value with respect to t.
 */
template<typename T>
HOST DEVICE inline T catmull_rom_deriv(T v0, T v1, T v2, T v3, T t) {
	const T a3 = T{0.5} * (-v0 + T{3} * v1 - T{3} * v2 + v3);
	const T a2 = T{0.5} * (T{2} * v0 - T{5} * v1 + T{4} * v2 - v3);
	const T a1 = T{0.5} * (-v0 + v2);
	return (T{3} * a3 * t + T{2} * a2) * t + a1;
}

/**
 * @brief Joint value+gradient sample via cubic (Catmull-Rom) interpolation.
 * @details Separable cubic spline over a 4x4x4 neighborhood: ported from legacy
 *          ARBD's RigidBodyGrid::interpolateForceD, restructured into two
 *          reusable 1D helpers (catmull_rom_value/catmull_rom_deriv) applied
 *          successively along x, then y, then z - mathematically identical to
 *          the original inline a1/a2/a3 form. Returns the raw (non-negated)
 *          gradient, matching compute_gradient's convention - the caller
 *          negates for force.
 */
template<typename T>
HOST DEVICE GridSample<T> sample_grid_cubic(CONSTANT_PTR(T) __restrict__ grid_values,
											const Vector3_t<T>& world_pos,
											const Vector3_t<T>& origin,
											const Matrix3_t<T>& basis,
											const Matrix3_t<T>& basis_inv,
											const Vector3_t<idx_t>& dimensions,
											int boundary_condition) {
	const Vector3_t<T> grid_pos = basis_inv.transform(world_pos - origin);

	const int homeX = static_cast<int>(math::floor(grid_pos.x));
	const int homeY = static_cast<int>(math::floor(grid_pos.y));
	const int homeZ = static_cast<int>(math::floor(grid_pos.z));
	const T wx = grid_pos.x - static_cast<T>(homeX);
	const T wy = grid_pos.y - static_cast<T>(homeY);
	const T wz = grid_pos.z - static_cast<T>(homeZ);

	// Stage 1: blend along x at each of the 4x4 (iy, iz) taps
	T dVdx_stage1[4][4]; // [iy][iz]
	T V_stage1[4][4];	 // [iy][iz]
	for (int iz = 0; iz < 4; ++iz) {
		const int jz = homeZ - 1 + iz;
		for (int iy = 0; iy < 4; ++iy) {
			const int jy = homeY - 1 + iy;
			T v[4];
			for (int ix = 0; ix < 4; ++ix) {
				const int jx = homeX - 1 + ix;
				v[ix] = fetch_grid_value_or_zero(grid_values, jx, jy, jz, dimensions);
			}
			dVdx_stage1[iy][iz] = catmull_rom_deriv(v[0], v[1], v[2], v[3], wx);
			V_stage1[iy][iz] = catmull_rom_value(v[0], v[1], v[2], v[3], wx);
		}
	}

	// Stage 2: blend along y at each of the 4 iz taps
	T dVdx_stage2[4]; // d/dx, blended along y
	T dVdy_stage2[4]; // d/dy, before z blend
	T V_stage2[4];	  // value, blended along x and y
	for (int iz = 0; iz < 4; ++iz) {
		dVdx_stage2[iz] = catmull_rom_value(
			dVdx_stage1[0][iz], dVdx_stage1[1][iz], dVdx_stage1[2][iz], dVdx_stage1[3][iz], wy);
		dVdy_stage2[iz] = catmull_rom_deriv(
			V_stage1[0][iz], V_stage1[1][iz], V_stage1[2][iz], V_stage1[3][iz], wy);
		V_stage2[iz] = catmull_rom_value(
			V_stage1[0][iz], V_stage1[1][iz], V_stage1[2][iz], V_stage1[3][iz], wy);
	}

	// Stage 3: blend along z
	const Vector3_t<T> grad_grid(
		catmull_rom_value(dVdx_stage2[0], dVdx_stage2[1], dVdx_stage2[2], dVdx_stage2[3], wz),
		catmull_rom_value(dVdy_stage2[0], dVdy_stage2[1], dVdy_stage2[2], dVdy_stage2[3], wz),
		catmull_rom_deriv(V_stage2[0], V_stage2[1], V_stage2[2], V_stage2[3], wz));

	GridSample<T> result;
	result.value = catmull_rom_value(V_stage2[0], V_stage2[1], V_stage2[2], V_stage2[3], wz);
	result.gradient = basis_inv.transpose().transform(grad_grid);
	return result;
}

/**
 * @brief Joint value+gradient sample via trilinear interpolation.
 *        Thin wrapper combining interpolate_grid_point + compute_gradient so
 *        callers have one entry point per InterpolationOrder regardless of
 *        scheme.
 */
template<typename T>
HOST DEVICE GridSample<T> sample_grid_linear(CONSTANT_PTR(T) __restrict__ grid_values,
											 const Vector3_t<T>& world_pos,
											 const Vector3_t<T>& origin,
											 const Matrix3_t<T>& basis,
											 const Matrix3_t<T>& basis_inv,
											 const Vector3_t<idx_t>& dimensions,
											 int boundary_condition) {
	GridSample<T> result;
	result.value = interpolate_grid_point(
		grid_values, world_pos, origin, basis_inv, dimensions, boundary_condition);
	result.gradient = compute_gradient(
		grid_values, world_pos, origin, basis, basis_inv, dimensions, boundary_condition);
	return result;
}

/**
 * @brief Device-safe index wrapping for periodic boundaries
 */
HOST DEVICE inline idx_t wrap_index(int index, idx_t size) {
	if (index < 0) {
		return static_cast<idx_t>(index + static_cast<int>(size) *
											  ((-index / static_cast<int>(size)) + 1)) %
			   size;
	}
	return static_cast<idx_t>(index) % size;
}
/**
 * @brief Device-safe neighbor list extraction (works with raw pointers)
 */
template<typename T>
HOST DEVICE void get_neighbor_list_from_grid(CONSTANT_PTR(T) __restrict__ grid_values,
											 idx_t ix,
											 idx_t iy,
											 idx_t iz,
											 const Vector3_t<idx_t>& dimensions,
											 IndexList<idx_t, MAX_NEIGHBORS>& neighbors) {

	const idx_t nx_val = dimensions.x;
	const idx_t ny_val = dimensions.y;
	const idx_t nz_val = dimensions.z;

#if (!defined(__CUDA_ARCH__) && !defined(__SYCL_DEVICE_ONLY__))
	LOGINFO("get_neighbor_list_from_grid: ix={}, iy={}, iz={}, nx={}, ny={}, nz={}",
			ix,
			iy,
			iz,
			nx_val,
			ny_val,
			nz_val);
#endif

	// Fill 3x3x3 neighborhood
	for (int di = -1; di <= 1; ++di) {
		for (int dj = -1; dj <= 1; ++dj) {
			for (int dk = -1; dk <= 1; ++dk) {
				// Calculate neighbor indices with wrapping
				const idx_t ni = wrap_index(static_cast<int>(ix) + di, nx_val);
				const idx_t nj = wrap_index(static_cast<int>(iy) + dj, ny_val);
				const idx_t nk = wrap_index(static_cast<int>(iz) + dk, nz_val);

				const idx_t neighbor_idx = nk + nj * nz_val + ni * ny_val * nz_val;
				const T value = grid_values[neighbor_idx];
				neighbors.add(neighbor_idx);

#if (!defined(__CUDA_ARCH__) && !defined(__SYCL_DEVICE_ONLY__))
				if (di == -1 && dj == 0 && dk == 0) {
					LOGINFO("left_neighbor: di={}, dj={}, dk={}, ni={}, nj={}, nk={}, "
							"idx={}, value={}",
							di,
							dj,
							dk,
							ni,
							nj,
							nk,
							neighbor_idx,
							value);
				}
#endif
			}
		}
	}

	return;
}

/**
 * @brief Device-safe single neighbor access (works with raw pointers)
 */
template<typename T>
HOST DEVICE T get_neighbor_from_grid(CONSTANT_PTR(T) __restrict__ grid_values,
									 idx_t ix,
									 idx_t iy,
									 idx_t iz,
									 int di,
									 int dj,
									 int dk,
									 const Vector3_t<idx_t>& dimensions,
									 int boundary_condition) {
	const idx_t ni = wrap_index(static_cast<int>(ix) + di, dimensions.x);
	const idx_t nj = wrap_index(static_cast<int>(iy) + dj, dimensions.y);
	const idx_t nk = wrap_index(static_cast<int>(iz) + dk, dimensions.z);

	const idx_t neighbor_idx = nk + nj * dimensions.z + ni * dimensions.y * dimensions.z;
	return grid_values[neighbor_idx];
}

/**
 * @brief Format-templated grid sampler seam for Phase 4 batched RB grid-grid kernels.
 * @details Dense is fully defined; Sparse is declared-only until PNanoVDB lands.
 */
template<GridFormat F, typename T>
HOST DEVICE GridSample<T> sample_grid(CONSTANT_PTR(T) __restrict__ grid_values,
									  const Vector3_t<T>& world_pos,
									  const Vector3_t<T>& origin,
									  const Matrix3_t<T>& basis,
									  const Matrix3_t<T>& basis_inv,
									  const Vector3_t<idx_t>& dimensions,
									  int boundary_condition,
									  int scheme) {
	if constexpr (F == GridFormat::Dense) {
		if (scheme == static_cast<int>(InterpolationOrder::Linear)) {
			return sample_grid_linear(
				grid_values, world_pos, origin, basis, basis_inv, dimensions, boundary_condition);
		}
		return sample_grid_cubic(
			grid_values, world_pos, origin, basis, basis_inv, dimensions, boundary_condition);
	} else {
		GridSample<T> zero{};
		return zero;
	}
}

} // namespace ARBD

#ifdef USE_SYCL
#include <sycl/sycl.hpp>
template<typename T>
struct sycl::is_device_copyable<ARBD::BaseGridView<T>> : std::true_type {};
#endif
