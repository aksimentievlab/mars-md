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
#include "Backend/Kernels.h"
#include "Backend/Resource.h"
#include "Header.h"
#include "NanoGridKernels.h"
#include "nanovdb/GridHandle.h"
#include "nanovdb/HostBuffer.h"
#include "nanovdb/NanoVDB.h"
#include "nanovdb/io/IO.h"
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

template<typename BufferT = DeviceBuffer<uint8_t>>
class NanoGridAdapter {
  private:
	nanovdb::GridHandle<nanovdb::HostBuffer> nvdb_handle_;
	BufferT arbd_buffer_;
	Resource resource_;

  public:
	/**
	 * @brief Create adapter from existing NanoVDB GridHandle
	 */
	explicit NanoGridAdapter(nanovdb::GridHandle<nanovdb::HostBuffer>&& handle,
							 const Resource& resource)
		: nvdb_handle_(std::move(handle)), resource_(resource) {

		// Copy grid data to ARBD buffer for cross-backend access
		size_t total_size = nvdb_handle_.bufferSize();
		arbd_buffer_.resize(total_size, resource_);
		arbd_buffer_.copy_from_host(reinterpret_cast<const uint8_t*>(nvdb_handle_.data()),
									total_size);
	}

	/**
	 * @brief Load grid from file using NanoVDB I/O
	 */
	static NanoGridAdapter from_file(const std::string& filename,
									 const Resource& resource,
									 const std::string& grid_name = "") {
		auto handle = grid_name.empty() ? nanovdb::io::readGrid(filename)
										: nanovdb::io::readGrid(filename, grid_name);

		return NanoGridAdapter(std::move(handle), resource);
	}

	/**
	 * @brief Get NanoVDB grid pointer for stencil operations
	 */
	template<typename ValueT>
	const nanovdb::NanoGrid<ValueT>* grid() const {
		return nvdb_handle_.grid<ValueT>();
	}

	/**
	 * @brief Get ARBD buffer for kernel operations
	 */
	BufferT& buffer() {
		return arbd_buffer_;
	}
	const BufferT& buffer() const {
		return arbd_buffer_;
	}

	/**
	 * @brief Get device pointer to grid data for kernels
	 */
	template<typename ValueT>
	DEVICE_PTR(nanovdb::NanoGrid<ValueT>)
	device_grid() const {
		// Use const_cast to remove const qualifier for device pointer
		// The buffer data is logically non-const from device perspective
		return reinterpret_cast<DEVICE_PTR(nanovdb::NanoGrid<ValueT>)>(
			const_cast<uint8_t*>(arbd_buffer_.data()));
	}

	/**
	 * @brief Get resource this adapter is allocated on
	 */
	const Resource& resource() const {
		return resource_;
	}

	/**
	 * @brief Sync data back to host if needed
	 */
	void sync_to_host() {
		std::vector<uint8_t> temp_data(arbd_buffer_.size());
		arbd_buffer_.copy_to_host(temp_data.data(), arbd_buffer_.size());

		// Copy back to NanoVDB handle
		std::memcpy(const_cast<void*>(nvdb_handle_.data()), temp_data.data(), temp_data.size());
	}

	size_t size() const {
		return arbd_buffer_.size();
	}
};

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

	// --- Stencil Operations ---
	template<typename ValueT, typename CoordT>
	Event compute_gradients(const DeviceBuffer<CoordT>& coords,
							DeviceBuffer<nanovdb::math::Vec3<ValueT>>& results,
							const nanovdb::NanoGrid<ValueT>* grid,
							const KernelConfig& config = KernelConfig{}) {

		size_t num_points = coords.size();
		results.resize(num_points, resource_);

		GradientStencilFunctor<ValueT, CoordT> func{grid,
													coords.data(),
													results.data(),
													num_points};

		auto inputs = std::make_tuple(grid, coords.data(), results.data(), num_points);
		auto outputs = std::make_tuple(results.data());

		return launch_kernel(resource_, num_points, config, inputs, outputs, func);
	}

	template<typename ValueT, typename CoordT>
	Event compute_laplacians(const DeviceBuffer<CoordT>& coords,
							 DeviceBuffer<ValueT>& results,
							 const nanovdb::NanoGrid<ValueT>* grid,
							 const KernelConfig& config = KernelConfig{}) {

		size_t num_points = coords.size();
		results.resize(num_points, resource_);

		LaplacianStencilFunctor<ValueT, CoordT> func{grid,
													 coords.data(),
													 results.data(),
													 num_points};

		auto inputs = std::make_tuple(grid, coords.data(), results.data(), num_points);
		auto outputs = std::make_tuple(results.data());

		return launch_kernel(resource_, num_points, config, inputs, outputs, func);
	}

	template<typename ValueT, typename PosT>
	Event interpolate_values(const DeviceBuffer<PosT>& positions,
							 DeviceBuffer<ValueT>& results,
							 const nanovdb::NanoGrid<ValueT>* grid,
							 const KernelConfig& config = KernelConfig{}) {

		size_t num_points = positions.size();
		results.resize(num_points, resource_);

		TrilinearInterpolationFunctor<ValueT, PosT> func{grid,
														 positions.data(),
														 results.data(),
														 num_points};

		auto inputs = std::make_tuple(grid, positions.data(), results.data(), num_points);
		auto outputs = std::make_tuple(results.data());

		return launch_kernel(resource_, num_points, config, inputs, outputs, func);
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
