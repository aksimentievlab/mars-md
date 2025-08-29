#pragma once
#include "Header.h"
#include "Vector3.h"
#include "nanovdb/GridHandle.h"
#include "nanovdb/NanoVDB.h"
#include "nanovdb/math/Stencils.h"

namespace ARBD {

// --- Functor for GridHandle Meta Copy ---
template<typename GridDataT, typename GridHandleMetaDataT>
struct CpyGridHandleMetaFunctor {
	const GridDataT* d_data;
	GridHandleMetaDataT* d_meta;

	HOST DEVICE void
	operator()(size_t i, const GridDataT* d_data, GridHandleMetaDataT* d_meta) const {
		if (i == 0) { // Only one thread needs to do this
			nanovdb::cpyGridHandleMeta(d_data, d_meta);
		}
	}
};

// --- Functor for Grid Count Update ---
template<typename GridDataT>
struct UpdateGridCountFunctor {
	GridDataT* d_data;
	uint32_t gridIndex;
	uint32_t gridCount;
	bool* d_dirty;

	HOST DEVICE void operator()(size_t i,
								GridDataT* d_data,
								uint32_t gridIndex,
								uint32_t gridCount,
								bool* d_dirty) const {
		if (i == 0) { // Only one thread needs to do this
			*d_dirty = (d_data->mGridIndex != gridIndex) || (d_data->mGridCount != gridCount);
			if (*d_dirty) {
				d_data->mGridIndex = gridIndex;
				d_data->mGridCount = gridCount;
				if (d_data->mChecksum.isEmpty()) {
					*d_dirty = false; // no need to update checksum if it didn't already exist
				}
			}
		}
	}
};

// --- Stencil Gradient Computation Functor ---
template<typename ValueT, typename CoordT>
struct GradientStencilFunctor {
	const nanovdb::NanoGrid<ValueT>* grid;
	const CoordT* coordinates;
	nanovdb::math::Vec3<ValueT>* results;
	size_t num_points;

	HOST DEVICE void operator()(size_t idx) const {
		if (idx < num_points) {
			nanovdb::math::GradStencil<nanovdb::NanoGrid<ValueT>> stencil(*grid);
			stencil.moveTo(coordinates[idx]);
			results[idx] = stencil.gradient();
		}
	}
};

// --- Stencil Laplacian Computation Functor ---
template<typename ValueT, typename CoordT>
struct LaplacianStencilFunctor {
	const nanovdb::NanoGrid<ValueT>* grid;
	const CoordT* coordinates;
	ValueT* results;
	size_t num_points;

	HOST DEVICE void operator()(size_t idx) const {
		if (idx < num_points) {
			nanovdb::math::GradStencil<nanovdb::NanoGrid<ValueT>> stencil(*grid);
			stencil.moveTo(coordinates[idx]);
			results[idx] = stencil.laplacian();
		}
	}
};

// --- Stencil Mean Curvature Computation Functor ---
template<typename ValueT, typename CoordT>
struct MeanCurvatureStencilFunctor {
	const nanovdb::NanoGrid<ValueT>* grid;
	const CoordT* coordinates;
	ValueT* results;
	size_t num_points;

	HOST DEVICE void operator()(size_t idx) const {
		if (idx < num_points) {
			nanovdb::math::CurvatureStencil<nanovdb::NanoGrid<ValueT>> stencil(*grid);
			stencil.moveTo(coordinates[idx]);
			results[idx] = stencil.meanCurvature();
		}
	}
};

// --- Trilinear Interpolation Functor ---
template<typename ValueT, typename PosT>
struct TrilinearInterpolationFunctor {
	const nanovdb::NanoGrid<ValueT>* grid;
	const PosT* world_positions;
	ValueT* results;
	size_t num_points;

	HOST DEVICE void operator()(size_t idx) const {
		if (idx < num_points) {
			nanovdb::math::BoxStencil<nanovdb::NanoGrid<ValueT>> stencil(*grid);
			auto coord = grid->worldToIndex(world_positions[idx]);
			stencil.moveTo(coord);
			results[idx] = stencil.interpolation(world_positions[idx]);
		}
	}
};

} // namespace ARBD
