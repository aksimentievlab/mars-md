#pragma once
#include "Header.h"
#include "Matrix3.h"
#include "Vector3.h"

namespace ARBD {

/**
 * @brief Grid configuration structure for initialization
 */
/**
 * @brief Interpolation orders supported by the grid
 */
enum class InterpolationOrder : int {
	Linear = 1, ///< Linear interpolation
	Cubic = 3	///< Cubic interpolation
};

template<typename T = float>
struct GridConfig {
	/**
	 * @brief Boundary condition types for grid operations
	 */
	enum class BoundaryCondition : int {
		Dirichlet = 0, ///< Fixed value at boundary
		Neumann = 1,   ///< Fixed derivative at boundary
		Periodic = 2   ///< Periodic boundary conditions
	};
	Vector3_t<T> origin{0, 0, 0};		  ///< Origin point of the grid
	Matrix3_t<T> basis{T(1)};			  ///< Basis vectors defining grid spacing
	Vector3_t<idx_t> dimensions{1, 1, 1}; ///< Grid dimensions (nx, ny, nz)
	BoundaryCondition boundary = BoundaryCondition::Periodic;

	HOST DEVICE constexpr idx_t total_size() const noexcept {
		return dimensions.x * dimensions.y * dimensions.z;
	}

	HOST DEVICE constexpr bool is_valid() const noexcept {
		return dimensions.x > 0 && dimensions.y > 0 && dimensions.z > 0;
	}
};
} // namespace ARBD
