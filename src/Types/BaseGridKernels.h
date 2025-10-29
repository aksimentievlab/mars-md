#pragma once
#include "Header.h"
#include "IndexList.h"
#include "Matrix3.h"
#include "Vector3.h"

namespace ARBD {

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
	T operator()(const T* grid_values,
				 const Vector3_t<T>& world_pos,
				 const Vector3_t<T>& origin,
				 const Matrix3_t<T>& basis_inv,
				 const Vector3_t<idx_t>& dimensions) const {
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
 * @brief Device-safe interpolation function (CUDA/SYCL compatible)
 */
template<typename T>
HOST DEVICE T interpolate_grid_point(const T* grid_values,
									 const Vector3_t<T>& world_pos,
									 const Vector3_t<T>& origin,
									 const Matrix3_t<T>& basis_inv,
									 const Vector3_t<idx_t>& dimensions) {
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

/**
 * @brief Get value at nearest grid point (device-safe)
 */
template<typename T>
HOST DEVICE T get_value_nearest(const T* grid_values,
								const Vector3_t<T>& world_pos,
								const Vector3_t<T>& origin,
								const Matrix3_t<T>& basis_inv,
								const Vector3_t<idx_t>& dimensions) {
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
HOST DEVICE Vector3_t<T> compute_gradient(const T* grid_values,
										  const Vector3_t<T>& world_pos,
										  const Vector3_t<T>& origin,
										  const Matrix3_t<T>& basis,
										  const Matrix3_t<T>& basis_inv,
										  const Vector3_t<idx_t>& dimensions) {
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
HOST DEVICE void get_neighbor_list_from_grid(const T* grid_values,
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
HOST DEVICE T get_neighbor_from_grid(const T* grid_values,
									 idx_t ix,
									 idx_t iy,
									 idx_t iz,
									 int di,
									 int dj,
									 int dk,
									 const Vector3_t<idx_t>& dimensions) {
	const idx_t ni = wrap_index(static_cast<int>(ix) + di, dimensions.x);
	const idx_t nj = wrap_index(static_cast<int>(iy) + dj, dimensions.y);
	const idx_t nk = wrap_index(static_cast<int>(iz) + dk, dimensions.z);

	const idx_t neighbor_idx = nk + nj * dimensions.z + ni * dimensions.y * dimensions.z;
	return grid_values[neighbor_idx];
}
} // namespace ARBD
