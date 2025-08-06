/*********************************************************************
 * @file  BaseGridKernels.h
 *
 * @brief Kernel functors for BaseGrid operations - Multi-backend support
 *********************************************************************/
#pragma once

#include "Math/Types.h"

namespace ARBD {
enum class BoundaryCondition { dirichlet, neumann, periodic };
enum class InterpolationOrder { linear = 1, cubic = 3 };

namespace BaseGridKernels {
/*===============================*\
|      BASEGRID KERNEL FUNCTORS  |
\===============================*/

/**
 * @brief Kernel to zero out grid values
 */

struct ZeroKernel {
	HOST DEVICE void operator()(size_t i, float* data) const {
		data[i] = 0.0f;
	}
};

/**
 * @brief Kernel to copy data between grids
 */
struct CopyKernel {
	HOST DEVICE void operator()(size_t i, const float* src, float* dst) const {
		dst[i] = src[i];
	}
};

/**
 * @brief Kernel to add scalar value to grid
 */
struct AddScalarKernel {
	float value;

	HOST DEVICE void operator()(size_t i, float* data) const {
		data[i] += value;
	}
};

/**
 * @brief Kernel to multiply grid by scalar value
 */
struct MultiplyScalarKernel {
	float value;

	HOST DEVICE void operator()(size_t i, float* data) const {
		data[i] *= value;
	}
};

/**
 * @brief Kernel to add two grids element-wise
 */
struct AddGridKernel {
	HOST DEVICE void operator()(size_t i, float* dst, const float* src) const {
		dst[i] += src[i];
	}
};

/**
 * @brief Kernel to multiply two grids element-wise
 */
struct MultiplyGridKernel {
	HOST DEVICE void operator()(size_t i, float* dst, const float* src) const {
		dst[i] *= src[i];
	}
};

/**
 * @brief Kernel to subtract two grids element-wise
 */
struct SubtractGridKernel {
	HOST DEVICE void operator()(size_t i, float* dst, const float* src) const {
		dst[i] -= src[i];
	}
};

/**
 * @brief Kernel to divide two grids element-wise (with safety check)
 */
struct DivideGridKernel {
	float epsilon; // Small value to avoid division by zero

	DivideGridKernel(float eps = 1e-10f) : epsilon(eps) {}

	HOST DEVICE void operator()(size_t i, float* dst, const float* src) const {
		if (fabsf(src[i]) > epsilon) {
			dst[i] /= src[i];
		} else {
			dst[i] = 0.0f; // Or some other safe value
		}
	}
};

/**
 * @brief Kernel to set grid to constant value
 */
struct SetConstantKernel {
	float value;

	HOST DEVICE void operator()(size_t i, float* data) const {
		data[i] = value;
	}
};

/**
 * @brief Kernel to apply absolute value to grid
 */
struct AbsKernel {
	HOST DEVICE void operator()(size_t i, float* data) const {
		data[i] = fabsf(data[i]);
	}
};

/**
 * @brief Kernel to clamp grid values to range [min_val, max_val]
 */
struct ClampKernel {
	float min_val, max_val;

	HOST DEVICE void operator()(size_t i, float* data) const {
		if (data[i] < min_val)
			data[i] = min_val;
		else if (data[i] > max_val)
			data[i] = max_val;
	}
};

/**
 * @brief Kernel for linear interpolation between two grids
 * result = (1-t) * grid1 + t * grid2
 */
struct LerpKernel {
	float t; // Interpolation parameter [0,1]

	HOST DEVICE void operator()(size_t i, const float* src1, const float* src2, float* dst) const {
		dst[i] = (1.0f - t) * src1[i] + t * src2[i];
	}
};

/**
 * @brief Kernel to apply power function to grid values
 */
struct PowerKernel {
	float exponent;

	HOST DEVICE void operator()(size_t i, float* data) const {
		data[i] = powf(data[i], exponent);
	}
};

/**
 * @brief Kernel to apply exponential function to grid values
 */
struct ExpKernel {
	HOST DEVICE void operator()(size_t i, float* data) const {
		data[i] = expf(data[i]);
	}
};

/**
 * @brief Kernel to apply natural logarithm to grid values (with safety)
 */
struct LogKernel {
	float epsilon;

	LogKernel(float eps = 1e-10f) : epsilon(eps) {}

	HOST DEVICE void operator()(size_t i, float* data) const {
		if (data[i] > epsilon) {
			data[i] = logf(data[i]);
		} else {
			data[i] = logf(epsilon); // Clamp to avoid -inf
		}
	}
};

/**
 * @brief Kernel to compute squared magnitude of grid values
 */
struct SquareKernel {
	HOST DEVICE void operator()(size_t i, float* data) const {
		data[i] = data[i] * data[i];
	}
};

/**
 * @brief Kernel to compute square root of grid values (with safety)
 */
struct SqrtKernel {
	HOST DEVICE void operator()(size_t i, float* data) const {
		data[i] = sqrtf(fmaxf(0.0f, data[i])); // Ensure non-negative
	}
};

/*===============================*\
|    INTERPOLATION KERNELS       |
\===============================*/

/**
 * @brief Kernel for batch trilinear interpolation
 * Used when interpolating many points at once
 */
template<BoundaryCondition BC>
struct TrilinearInterpolationKernel {
	int nx, ny, nz;
	Vector3 origin;
	Matrix3 basisInv;

	HOST DEVICE void
	operator()(size_t i, const Vector3* positions, const float* grid_data, float* results) const {
		Vector3 grid_pos = basisInv.transform(positions[i] - origin);

		// Get integer and fractional parts
		int i0 = int(floorf(grid_pos.x));
		int j0 = int(floorf(grid_pos.y));
		int k0 = int(floorf(grid_pos.z));

		float fx = grid_pos.x - i0;
		float fy = grid_pos.y - j0;
		float fz = grid_pos.z - k0;

		// Apply boundary conditions and sample 8 corner points
		auto getIdx = [&](int ii, int jj, int kk) -> size_t {
			if constexpr (BC == BoundaryCondition::periodic) {
				ii = ((ii % nx) + nx) % nx;
				jj = ((jj % ny) + ny) % ny;
				kk = ((kk % nz) + nz) % nz;
			} else if constexpr (BC == BoundaryCondition::dirichlet) {
				ii = fmaxf(0, fminf(nx - 1, ii));
				jj = fmaxf(0, fminf(ny - 1, jj));
				kk = fmaxf(0, fminf(nz - 1, kk));
			} else { // neumann
				if (ii < 0)
					ii = -ii;
				else if (ii >= nx)
					ii = 2 * nx - 1 - ii;
				if (jj < 0)
					jj = -jj;
				else if (jj >= ny)
					jj = 2 * ny - 1 - jj;
				if (kk < 0)
					kk = -kk;
				else if (kk >= nz)
					kk = 2 * nz - 1 - kk;
			}
			return size_t(ii) * ny * nz + size_t(jj) * nz + size_t(kk);
		};

		float c000 = grid_data[getIdx(i0, j0, k0)];
		float c001 = grid_data[getIdx(i0, j0, k0 + 1)];
		float c010 = grid_data[getIdx(i0, j0 + 1, k0)];
		float c011 = grid_data[getIdx(i0, j0 + 1, k0 + 1)];
		float c100 = grid_data[getIdx(i0 + 1, j0, k0)];
		float c101 = grid_data[getIdx(i0 + 1, j0, k0 + 1)];
		float c110 = grid_data[getIdx(i0 + 1, j0 + 1, k0)];
		float c111 = grid_data[getIdx(i0 + 1, j0 + 1, k0 + 1)];

		// Trilinear interpolation
		float c00 = c000 * (1.0f - fx) + c100 * fx;
		float c01 = c001 * (1.0f - fx) + c101 * fx;
		float c10 = c010 * (1.0f - fx) + c110 * fx;
		float c11 = c011 * (1.0f - fx) + c111 * fx;

		float c0 = c00 * (1.0f - fy) + c10 * fy;
		float c1 = c01 * (1.0f - fy) + c11 * fy;

		results[i] = c0 * (1.0f - fz) + c1 * fz;
	}
};

} // namespace BaseGridKernels
} // namespace ARBD