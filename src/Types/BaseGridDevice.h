#pragma once
#include "GridTerm.h"
#include "Header.h"
#include "IndexList.h"
#include "Math.h"
#include "Matrix3.h"
#include "Types.h"
#include "Vector3.h"
/**
 * @file BaseGridDevice.h
 * @brief Device-side grid views and shared sampling helpers.
 * @details GridGeometry<T> (geometry, no data), BaseGridView<T> (+ read-only
 *          pointer), BaseGridMutableView<T> (+ writable pointer).
 *          See BaseGridDevice.md.
 */
namespace ARBD {
/**
 * @brief Where a grid sits in space, plus the index math implied by that.
 * @details POD, no data pointer. Shared base of both views.
 * @tparam T Grid value type (float/double)
 */
template<typename T>
struct GridGeometry {
	Vector3_t<T> origin;		 ///< Grid origin in world space
	Matrix3_t<T> basis;			 ///< Basis vectors (grid spacing)
	Matrix3_t<T> basis_inv;		 ///< Cached inverse basis
	Vector3_t<idx_t> dimensions; ///< Grid dimensions (nx, ny, nz)
	int grid_id;
	int boundary_condition; ///< GridBoundaryCondition as int (see GridTerm.h)

	/*===================*\
	|  INDEX OPERATIONS   |
	\*===================*/

	/// @brief Convert 3D indices to linear index
	HOST DEVICE constexpr idx_t index(idx_t ix, idx_t iy, idx_t iz) const noexcept {
		return iz + iy * dimensions.z + ix * dimensions.y * dimensions.z;
	}

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
	|  SPATIAL QUERIES    |
	\*===================*/

	/// @brief Transform world position to grid coordinates
	HOST DEVICE Vector3_t<T> world_to_grid(const Vector3_t<T>& world_pos) const noexcept {
		return basis_inv.transform(world_pos - origin);
	}

	/// @brief Transform grid coordinates to world position
	HOST DEVICE Vector3_t<T> grid_to_world(const Vector3_t<T>& grid_pos) const noexcept {
		return basis.transform(grid_pos) + origin;
	}

	HOST DEVICE static constexpr T clamp(T val, T min_val, T max_val) noexcept {
		return val < min_val ? min_val : (val > max_val ? max_val : val);
	}
};

// Declared ahead of the views because the view methods below forward to them;
// the definitions follow the view definitions.
template<typename T>
HOST DEVICE T interpolate_grid_point(CONSTANT_PTR(T) __restrict__ grid_values,
									 const Vector3_t<T>& world_pos,
									 const Vector3_t<T>& origin,
									 const Matrix3_t<T>& basis_inv,
									 const Vector3_t<idx_t>& dimensions,
									 int boundary_condition);

template<typename T>
HOST DEVICE T get_value_nearest(CONSTANT_PTR(T) __restrict__ grid_values,
								const Vector3_t<T>& world_pos,
								const Vector3_t<T>& origin,
								const Matrix3_t<T>& basis_inv,
								const Vector3_t<idx_t>& dimensions,
								int boundary_condition);

template<typename T>
HOST DEVICE Vector3_t<T> compute_gradient(CONSTANT_PTR(T) __restrict__ grid_values,
										  const Vector3_t<T>& world_pos,
										  const Vector3_t<T>& origin,
										  const Matrix3_t<T>& basis,
										  const Matrix3_t<T>& basis_inv,
										  const Vector3_t<idx_t>& dimensions,
										  int boundary_condition);

/**
 * @brief Immutable device-side view of grid data
 * @details Lightweight POD safe for passing to CUDA/SYCL kernels by value.
 *          All operations are const and device-safe.
 */
template<typename T>
struct BaseGridView : GridGeometry<T> {
	CONSTANT_PTR(T) __restrict__ data; ///< Read-only pointer to device grid data

	/*===================*\
	|  VALUE ACCESS       |
	\*===================*/

	/// @brief Direct indexed access (no bounds checking)
	HOST DEVICE constexpr T operator[](idx_t linear_idx) const noexcept {
		return data[linear_idx];
	}

	HOST DEVICE constexpr T operator()(idx_t ix, idx_t iy, idx_t iz) const noexcept {
		return data[this->index(ix, iy, iz)];
	}

	/*===================*\
	|  SAMPLING           |
	\*===================*/
	// Thin forwards to the shared free functions - see the file header for why
	// these must not grow independent implementations.

	/// @brief Trilinear interpolation at world position
	HOST DEVICE T interpolate(const Vector3_t<T>& world_pos) const noexcept {
		return interpolate_grid_point<T>(data,
										 world_pos,
										 this->origin,
										 this->basis_inv,
										 this->dimensions,
										 this->boundary_condition);
	}

	/// @brief Nearest neighbor lookup
	HOST DEVICE T nearest(const Vector3_t<T>& world_pos) const noexcept {
		return get_value_nearest<T>(data,
									world_pos,
									this->origin,
									this->basis_inv,
									this->dimensions,
									this->boundary_condition);
	}

	/// @brief Gradient at world position (central differences)
	HOST DEVICE Vector3_t<T> gradient(const Vector3_t<T>& world_pos) const noexcept {
		return compute_gradient<T>(data,
								   world_pos,
								   this->origin,
								   this->basis,
								   this->basis_inv,
								   this->dimensions,
								   this->boundary_condition);
	}
};

/**
 * @brief Writable device-side view of grid data
 * @details Same geometry as BaseGridView but holds DEVICE_PTR(T), so grid
 *          mutation kernels (zero/scale/shift/multiply, convolution output) can
 *          write through it. Read access goes through const_view() so the
 *          sampling helpers are never duplicated for the mutable case.
 */
template<typename T>
struct BaseGridMutableView : GridGeometry<T> {
	DEVICE_PTR(T) __restrict__ data; ///< Writable pointer to device grid data

	HOST DEVICE T get(idx_t linear_idx) const noexcept {
		return data[linear_idx];
	}

	HOST DEVICE T get(idx_t ix, idx_t iy, idx_t iz) const noexcept {
		return data[this->index(ix, iy, iz)];
	}

	HOST DEVICE void set(idx_t linear_idx, T value) const noexcept {
		data[linear_idx] = value;
	}

	HOST DEVICE void set(idx_t ix, idx_t iy, idx_t iz, T value) const noexcept {
		data[this->index(ix, iy, iz)] = value;
	}

	/// @brief Read-only view over the same grid, for the sampling helpers.
	HOST DEVICE BaseGridView<T> const_view() const noexcept {
		BaseGridView<T> v;
		static_cast<GridGeometry<T>&>(v) = static_cast<const GridGeometry<T>&>(*this);
		v.data = data;
		return v;
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
		jy >= static_cast<int>(dimensions.y) || jz < 0 || jz >= static_cast<int>(dimensions.z)) {
		return T{0};
	}
	const idx_t idx = static_cast<idx_t>(jz) + static_cast<idx_t>(jy) * dimensions.z +
					  static_cast<idx_t>(jx) * dimensions.y * dimensions.z;
	return grid_values[idx];
}

/**
 * @brief Map one out-of-range index per boundary condition.
 * @return false if the tap lies outside the domain (Dirichlet)
 */
HOST DEVICE inline bool map_grid_index(int& j, idx_t n, int boundary_condition) {
	const int ni = static_cast<int>(n);
	if (j >= 0 && j < ni) {
		return true;
	}
	if (boundary_condition == static_cast<int>(GridBoundaryCondition::Periodic)) {
		j = ((j % ni) + ni) % ni;
		return true;
	}
	if (boundary_condition == static_cast<int>(GridBoundaryCondition::Neumann)) {
		j = j < 0 ? 0 : ni - 1; // zero derivative: replicate the edge value
		return true;
	}
	return false; // Dirichlet, and anything unrecognized
}

/// @brief Fetch one tap honoring the grid's boundary condition.
template<typename T>
HOST DEVICE inline T fetch_grid_value(CONSTANT_PTR(T) __restrict__ grid_values,
									  int jx,
									  int jy,
									  int jz,
									  const Vector3_t<idx_t>& dimensions,
									  int boundary_condition) {
	if (!map_grid_index(jx, dimensions.x, boundary_condition) ||
		!map_grid_index(jy, dimensions.y, boundary_condition) ||
		!map_grid_index(jz, dimensions.z, boundary_condition)) {
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

	const int i0 = static_cast<int>(math::floor(grid_pos.x));
	const int j0 = static_cast<int>(math::floor(grid_pos.y));
	const int k0 = static_cast<int>(math::floor(grid_pos.z));

	const T fx = grid_pos.x - static_cast<T>(i0);
	const T fy = grid_pos.y - static_cast<T>(j0);
	const T fz = grid_pos.z - static_cast<T>(k0);

	const T v000 = fetch_grid_value(grid_values, i0, j0, k0, dimensions, boundary_condition);
	const T v001 = fetch_grid_value(grid_values, i0, j0, k0 + 1, dimensions, boundary_condition);
	const T v010 = fetch_grid_value(grid_values, i0, j0 + 1, k0, dimensions, boundary_condition);
	const T v011 =
		fetch_grid_value(grid_values, i0, j0 + 1, k0 + 1, dimensions, boundary_condition);
	const T v100 = fetch_grid_value(grid_values, i0 + 1, j0, k0, dimensions, boundary_condition);
	const T v101 =
		fetch_grid_value(grid_values, i0 + 1, j0, k0 + 1, dimensions, boundary_condition);
	const T v110 =
		fetch_grid_value(grid_values, i0 + 1, j0 + 1, k0, dimensions, boundary_condition);
	const T v111 =
		fetch_grid_value(grid_values, i0 + 1, j0 + 1, k0 + 1, dimensions, boundary_condition);

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

	const int ix = static_cast<int>(math::floor(grid_pos.x + T{0.5}));
	const int iy = static_cast<int>(math::floor(grid_pos.y + T{0.5}));
	const int iz = static_cast<int>(math::floor(grid_pos.z + T{0.5}));

	return fetch_grid_value(grid_values, ix, iy, iz, dimensions, boundary_condition);
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

	// Dirichlet: no real neighbors at the edge, so keep the legacy zero.
	if (boundary_condition == static_cast<int>(GridBoundaryCondition::Dirichlet) &&
		(grid_pos.x < 1 || grid_pos.x >= nx - 1 || grid_pos.y < 1 || grid_pos.y >= ny - 1 ||
		 grid_pos.z < 1 || grid_pos.z >= nz - 1)) {
		return Vector3_t<T>{T{0}, T{0}, T{0}};
	}

	const int i = static_cast<int>(math::floor(grid_pos.x));
	const int j = static_cast<int>(math::floor(grid_pos.y));
	const int k = static_cast<int>(math::floor(grid_pos.z));

	// Central differences
	const T dx_grid = (fetch_grid_value(grid_values, i + 1, j, k, dimensions, boundary_condition) -
					   fetch_grid_value(grid_values, i - 1, j, k, dimensions, boundary_condition)) /
					  T{2};
	const T dy_grid = (fetch_grid_value(grid_values, i, j + 1, k, dimensions, boundary_condition) -
					   fetch_grid_value(grid_values, i, j - 1, k, dimensions, boundary_condition)) /
					  T{2};
	const T dz_grid = (fetch_grid_value(grid_values, i, j, k + 1, dimensions, boundary_condition) -
					   fetch_grid_value(grid_values, i, j, k - 1, dimensions, boundary_condition)) /
					  T{2};

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
				v[ix] = fetch_grid_value(grid_values, jx, jy, jz, dimensions, boundary_condition);
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
		dVdx_stage2[iz] = catmull_rom_value(dVdx_stage1[0][iz],
											dVdx_stage1[1][iz],
											dVdx_stage1[2][iz],
											dVdx_stage1[3][iz],
											wy);
		dVdy_stage2[iz] = catmull_rom_deriv(V_stage1[0][iz],
											V_stage1[1][iz],
											V_stage1[2][iz],
											V_stage1[3][iz],
											wy);
		V_stage2[iz] = catmull_rom_value(V_stage1[0][iz],
										 V_stage1[1][iz],
										 V_stage1[2][iz],
										 V_stage1[3][iz],
										 wy);
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
	result.value = interpolate_grid_point(grid_values,
										  world_pos,
										  origin,
										  basis_inv,
										  dimensions,
										  boundary_condition);
	result.gradient = compute_gradient(grid_values,
									   world_pos,
									   origin,
									   basis,
									   basis_inv,
									   dimensions,
									   boundary_condition);
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
			return sample_grid_linear(grid_values,
									  world_pos,
									  origin,
									  basis,
									  basis_inv,
									  dimensions,
									  boundary_condition);
		}
		return sample_grid_cubic(grid_values,
								 world_pos,
								 origin,
								 basis,
								 basis_inv,
								 dimensions,
								 boundary_condition);
	} else {
		GridSample<T> zero{};
		return zero;
	}
}

} // namespace ARBD

#ifdef USE_SYCL
#include <sycl/sycl.hpp>
template<typename T>
struct sycl::is_device_copyable<ARBD::GridGeometry<T>> : std::true_type {};
template<typename T>
struct sycl::is_device_copyable<ARBD::BaseGridView<T>> : std::true_type {};
template<typename T>
struct sycl::is_device_copyable<ARBD::BaseGridMutableView<T>> : std::true_type {};
#endif
