#pragma once
#include "Header.h"

#include <cstdint>

namespace MARS {

/**
 * @file GridTerm.h
 * @brief Grid vocabulary shared by every grid backend and by the kernels that
 *        sample grids - deliberately free of any storage-format dependency.
 *
 * Lives in Types/ (not Objects/Grid.h) because device code needs it: the dense
 * path (BaseGridDevice.h) and the sparse PNanoVDB path both consume these, and
 * neither can include Objects/Grid.h, which is a host-side registry pulling in
 * DeviceBuffer, std::unordered_map and file IO. Objects/Grid.h keeps GridType,
 * GridKey and GridManager - the things that only make sense on the host.
 */

/**
 * @brief How a grid behaves outside its own bounds
 * @details Values are the wire format: BaseGridView stores this as a plain int
 *          (see BaseGrid<T>::Config::boundary_as_int) so the struct stays POD
 *          across the CUDA/SYCL/Metal backends.
 */
enum class GridBoundaryCondition : uint8_t {
	Dirichlet = 0, ///< Fixed value at boundary (legacy zero-pads out-of-bounds taps)
	Neumann = 1,   ///< Fixed derivative at boundary
	Periodic = 2   ///< Periodic boundary conditions
};

/// Storage backend a grid is held in. grid_ids are unified across both.
enum class GridFormat : uint8_t { Dense = 0, Sparse = 1 };

/// Interpolation order used when sampling a grid (legacy's `scheme` flag).
enum class InterpolationOrder : uint8_t { Linear = 1, Cubic = 3 };

/**
 * @brief One (entity type -> grid) reference: which grid, and how it is weighted
 * @details A type can reference several grids - legacy MARS's `gridFile` takes a
 *          whitespace-separated list with one scale and one boundary condition
 *          per entry - so per-type grid data is stored as an offset+count range
 *          into a flat array of these terms (see ParticleTypeView::pmf_grid_terms
 *          and RigidBodyTypeView's grid ranges).
 *
 *          `grid_id` indexes GridManager's per-resource grid view array rather
 *          than inlining a view here, for two reasons. It keeps the term
 *          format-agnostic - GridManager hands out one unified id space for
 *          dense and sparse grids, so the same term works once the PNanoVDB
 *          path lands, whereas an inlined BaseGridView<mars_real> would have made
 *          this dense-only. And it decouples initialization order: views exist
 *          only after grids are uploaded, while terms are built when types are
 *          uploaded.
 */
struct GridTerm {
	int grid_id = -1;		  ///< Index into GridManager's grid views; <0 = unused slot
	float scale = 1.0f;		  ///< legacy: gridFileScale
	float scale_slope = 0.0f; ///< legacy: gridFileScaleSlope
	/// legacy: gridFileBoundaryConditions; <0 = use the grid's own.
	/// Recorded per (type, grid) because that's how the config keys it, but not
	/// yet observable: BaseGridDevice.h's samplers take a boundary_condition
	/// argument and ignore it (out-of-bounds always reads as zero, neighbor
	/// taps always wrap). Honoring it is a change to those samplers, not here.
	int boundary_condition = -1;

	HOST DEVICE constexpr bool is_valid() const noexcept {
		return grid_id >= 0;
	}
};

} // namespace MARS
