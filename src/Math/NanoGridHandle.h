/**
 * @file NanoGrid.h
 * @brief Sparse grid for NanoVDB
 *
 * This file contains the NanoGrid class, which is a sparse grid for NanoVDB.
 * It is a wrapper around the NanoVDB library, and provides a simple interface
 * for creating and manipulating NanoVDB grids.
 *
 * @copyright Copyright (c) 2025
 * Modified from NanoVDB library
 */
#pragma once
#include "Backend/Buffer.h"
#include "Backend/Events.h"
#include "Backend/Header.h"
#include "Backend/Kernels.h"
#include "Backend/Resource.h"
#include "nanovdb/GridHandle.h"
#include "nanovdb/HostBuffer.h"
#include "nanovdb/NanoVDB.h"
#include "nanovdb/math/Stencils.h"

namespace nanovdb {
template<typename T, typename Policy>
struct BufferTraits<ARBD::Buffer<T, Policy>> {
	static constexpr bool hasDeviceDual = true;
};
} // namespace nanovdb

namespace ARBD {
using GridData = nanovdb::GridData;
using GridHandleMetaData = nanovdb::GridHandleMetaData;
using NodeManagerData = nanovdb::NodeManagerData;

template<typename BufferT>
using GridHandle = nanovdb::GridHandle<BufferT>;

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

} // namespace ARBD

namespace ARBD {
using GridData = nanovdb::GridData;
using GridHandleMetaData = nanovdb::GridHandleMetaData;
using NodeManagerData = nanovdb::NodeManagerData;

template<typename BufferT>
using GridHandle = nanovdb::GridHandle<BufferT>;

template<typename Resource>
class NanoGrid {
  private:
	const Resource& resource_;

  public:
	explicit NanoGrid(const Resource& resource) : resource_(resource) {
		if (!resource.is_device()) {
			throw std::invalid_argument("NanoGrid requires a valid device resource");
		}
	}

	// --- GridHandle Meta Copy ---
	Event cpyGridHandleMeta(const GridData* d_data, GridHandleMetaData* d_meta) {
		ARBD::KernelConfig config;
		CpyGridHandleMetaFunctor<GridData, GridHandleMetaData> func{d_data, d_meta};

		auto inputs = std::make_tuple(d_data, d_meta);
		auto outputs = std::make_tuple(); // No outputs

		return launch_kernel(resource_, 1, config, inputs, outputs, func);
	}

	// --- Grid Count Update ---
	Event updateGridCount(GridData* d_data, idx_t gridIndex, idx_t gridCount, bool* d_dirty) {
		ARBD::KernelConfig config;
		UpdateGridCountFunctor<GridData> func{d_data,
											  static_cast<uint32_t>(gridIndex),
											  static_cast<uint32_t>(gridCount),
											  d_dirty};

		auto inputs = std::make_tuple(d_data,
									  static_cast<uint32_t>(gridIndex),
									  static_cast<uint32_t>(gridCount),
									  d_dirty);
		auto outputs = std::make_tuple(); // No outputs

		return launch_kernel(resource_, 1, config, inputs, outputs, func);
	}
};

} // namespace ARBD

#ifdef USE_SYCL
#include <sycl/sycl.hpp>

template<typename GridDataT, typename GridHandleMetaDataT>
struct sycl::is_device_copyable<ARBD::CpyGridHandleMetaFunctor<GridDataT, GridHandleMetaDataT>>
	: std::true_type {};

template<typename GridDataT>
struct sycl::is_device_copyable<ARBD::UpdateGridCountFunctor<GridDataT>> : std::true_type {};

#endif
