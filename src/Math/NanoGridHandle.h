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
#include "nanovdb/NanoVDB.h"
#include "nanovdb/math/Stencils.h"

namespace ARBD {
class NanoGrid {
	using GridData = nanovdb::GridData;
	using GridHandleMetaData = nanovdb::GridHandleMetaData;
	using NodeManagerData = nanovdb::NodeManagerData;

	template<typename BufferT>
	using GridHandle = nanovdb::GridHandle<BufferT>;

	void cpyGridHandleMeta(const GridData* d_data, GridHandleMetaData* d_meta) {
		nanovdb::cpyGridHandleMeta(d_data, d_meta);
	}

	void updateGridCount(nanovdb::GridData* d_data,
						 uint32_t gridIndex,
						 uint32_t gridCount,
						 bool* d_dirty) {
		*d_dirty = (d_data->mGridIndex != gridIndex) || (d_data->mGridCount != gridCount);
		if (*d_dirty) {
			d_data->mGridIndex = gridIndex;
			d_data->mGridCount = gridCount;
			if (d_data->mChecksum.isEmpty())
				*d_dirty = false; // no need to update checksum if it didn't already exist
		}
	}

	template<typename BufferT, template<class, class...> class VectorT = std::vector>
	VectorT<GridHandle<BufferT>> splitGridHandles(const GridHandle<BufferT>& handle,
												  const BufferT* other = nullptr,
												  auto& stream = nullptr) {
		const void* ptr = handle.deviceData();
		if (ptr == nullptr)
			return VectorT<GridHandle<BufferT>>();
		VectorT<GridHandle<BufferT>> handles(handle.gridCount());
		bool dirty, *d_dirty; // use this to check if the checksum needs to be recomputed
		CUDA_CHECK(cudaMallocAsync((void**)&d_dirty, sizeof(bool), stream));
		int device = 0;
		cudaGetDevice(&device);
		for (uint32_t n = 0; n < handle.gridCount(); ++n) {
			auto buffer = BufferT::create(handle.gridSize(n), other, device, stream);
			GridData* dst = reinterpret_cast<GridData*>(buffer.deviceData());
			const GridData* src = reinterpret_cast<const GridData*>(ptr);
			CUDA_CHECK(
				cudaMemcpyAsync(dst, src, handle.gridSize(n), cudaMemcpyDeviceToDevice, stream));
			updateGridCount<<<1, 1, 0, stream>>>(dst, 0u, 1u, d_dirty);
			CUDA_CHECK(cudaStreamSynchronize(stream));
			CUDA_CHECK(
				cudaMemcpyAsync(&dirty, d_dirty, sizeof(bool), cudaMemcpyDeviceToHost, stream));
			CUDA_CHECK(cudaStreamSynchronize(stream));
			if (dirty)
				nanovdb::tools::cuda::updateChecksum(dst, nanovdb::CheckMode::Partial, stream);
			handles[n] = nanovdb::GridHandle<BufferT>(std::move(buffer));
			ptr = nanovdb::util::PtrAdd(ptr, handle.gridSize(n));
		}
		CUDA_CHECK(cudaFreeAsync(d_dirty, stream));
		return std::move(handles);
	} // cuda::splitGridHandles

	template<typename BufferT, template<class, class...> class VectorT>
	inline typename nanovdb::util::enable_if<nanovdb::BufferTraits<BufferT>::hasDeviceDual,
											 GridHandle<BufferT>>::type
	mergeGridHandles(const VectorT<GridHandle<BufferT>>& handles,
					 const BufferT* other = nullptr,
					 cudaStream_t stream = 0) {
		uint64_t size = 0u;
		uint32_t counter = 0u, gridCount = 0u;
		for (auto& h : handles) {
			gridCount += h.gridCount();
			for (uint32_t n = 0; n < h.gridCount(); ++n)
				size += h.gridSize(n);
		}
		int device = 0;
		(cudaGetDevice(&device));
		auto buffer = BufferT::create(size, other, device, stream);
		void* dst = buffer.deviceData();
		bool dirty, *d_dirty; // use this to check if the checksum needs to be recomputed
		(util::cuda::mallocAsync((void**)&d_dirty, sizeof(bool), stream));
		for (auto& h : handles) {
			const void* src = h.deviceData();
			for (uint32_t n = 0; n < h.gridCount(); ++n) {
				(cudaMemcpyAsync(dst, src, h.gridSize(n), cudaMemcpyDeviceToDevice, stream));
				GridData* data = reinterpret_cast<GridData*>(dst);
				updateGridCount<<<1, 1, 0, stream>>>(data, counter++, gridCount, d_dirty);
				Error();
				(cudaMemcpyAsync(&dirty, d_dirty, sizeof(bool), cudaMemcpyDeviceToHost, stream));
				(cudaStreamSynchronize(stream));
				if (dirty)
					tools::cuda::updateChecksum(data, CheckMode::Partial, stream);
				dst = util::PtrAdd(dst, h.gridSize(n));
				src = util::PtrAdd(src, h.gridSize(n));
			}
		}
		(util::cuda::freeAsync(d_dirty, stream));
		return GridHandle<BufferT>(std::move(buffer));
	} // cuda::mergeGridHandles

	template<typename T, typename util::enable_if<BufferTraits<T>::hasDeviceDual, int>::type>
	GridHandle<BufferT>::GridHandle(T&& buffer) {
		static_assert(util::is_same<T, BufferT>::value, "Expected U==BufferT");
		mBuffer = std::move(buffer);
		if (auto* data = reinterpret_cast<const GridData*>(mBuffer.data())) {
			if (!data->isValid())
				throw std::runtime_error("GridHandle was constructed with an invalid host buffer");
			mMetaData.resize(data->mGridCount);
			cpyGridHandleMeta(data, mMetaData.data());
		} else {
			if (auto* d_data = reinterpret_cast<const GridData*>(mBuffer.deviceData())) {
				GridData tmp;
				(cudaMemcpy(&tmp, d_data, sizeof(GridData), cudaMemcpyDeviceToHost));
				if (!tmp.isValid())
					throw std::runtime_error(
						"GridHandle was constructed with an invalid device buffer");
				GridHandleMetaData* d_metaData;
				cudaMalloc((void**)&d_metaData, tmp.mGridCount * sizeof(GridHandleMetaData));
				cuda::cpyGridHandleMeta<<<1, 1>>>(d_data, d_metaData);
				mMetaData.resize(tmp.mGridCount);
				(cudaMemcpy(mMetaData.data(),
							d_metaData,
							tmp.mGridCount * sizeof(GridHandleMetaData),
							cudaMemcpyDeviceToHost));
				(cudaFree(d_metaData));
			}
		}
	} // GridHandle(T&& buffer)

	// Dummy function that ensures instantiation of the move-constructor above when
	// BufferT=cuda::DeviceBuffer
	namespace {
	auto __dummy() {
		return nanovdb::GridHandle<nanovdb::cuda::DeviceBuffer>(
			std::move(nanovdb::cuda::DeviceBuffer()));
	}
	}
	template<typename BuildT, typename BufferT = DeviceBuffer>
	inline typename util::enable_if<BufferTraits<BufferT>::hasDeviceDual,
									NodeManagerHandle<BufferT>>::type
	createNodeManager(const NanoGrid<BuildT>* d_grid,
					  const BufferT& pool = BufferT(),
					  cudaStream_t stream = 0) {
		int device = 0;
		(cudaGetDevice(&device));
		auto buffer = BufferT::create(sizeof(NodeManagerData), &pool, device, stream);
		auto* d_data = (NodeManagerData*)buffer.deviceData();
		size_t size = 0u, *d_size;
		(util::cuda::mallocAsync((void**)&d_size, sizeof(size_t), stream));
		util::cuda::lambdaKernel<<<1, 1, 0, stream>>>(1, [=] __device__(size_t) {
			*d_data = NodeManagerData((void*)d_grid);
			*d_size = sizeof(NodeManagerData);
			auto& tree = d_grid->tree();
			if (NodeManager<BuildT>::FIXED_SIZE && d_grid->isBreadthFirst()) {
				d_data->mLinear = uint8_t(1u);
				d_data->mOff[0] = util::PtrDiff(tree.template getFirstNode<0>(), d_grid);
				d_data->mOff[1] = util::PtrDiff(tree.template getFirstNode<1>(), d_grid);
				d_data->mOff[2] = util::PtrDiff(tree.template getFirstNode<2>(), d_grid);
			} else {
				*d_size += sizeof(uint64_t) * tree.totalNodeCount();
			}
		});
		Error();
		(cudaMemcpyAsync(&size, d_size, sizeof(size_t), cudaMemcpyDeviceToHost, stream));
		(util::cuda::freeAsync(d_size, stream));
		if (size > sizeof(NodeManagerData)) {
			auto tmp =
				BufferT::create(size, &pool, device, stream); // only allocate buffer on the device
			(cudaMemcpyAsync(tmp.deviceData(),
							 buffer.deviceData(),
							 sizeof(NodeManagerData),
							 cudaMemcpyDeviceToDevice,
							 stream));
			buffer = std::move(tmp);
			d_data = reinterpret_cast<NodeManagerData*>(buffer.deviceData());
			util::cuda::lambdaKernel<<<1, 1, 0, stream>>>(1, [=] __device__(size_t) {
				auto& tree = d_grid->tree();
				int64_t* ptr0 = d_data->mPtr[0] = reinterpret_cast<int64_t*>(d_data + 1);
				int64_t* ptr1 = d_data->mPtr[1] = d_data->mPtr[0] + tree.nodeCount(0);
				int64_t* ptr2 = d_data->mPtr[2] = d_data->mPtr[1] + tree.nodeCount(1);
				// Performs depth first traversal but breadth first insertion
				for (auto it2 = tree.root().cbeginChild(); it2; ++it2) {
					*ptr2++ = util::PtrDiff(&*it2, d_grid);
					for (auto it1 = it2->cbeginChild(); it1; ++it1) {
						*ptr1++ = util::PtrDiff(&*it1, d_grid);
						for (auto it0 = it1->cbeginChild(); it0; ++it0) {
							*ptr0++ = util::PtrDiff(&*it0, d_grid);
						} // loop over child nodes of the lower internal node
					}	  // loop over child nodes of the upper internal node
				}		  // loop over child nodes of the root node
			});
		}

		return NodeManagerHandle<BufferT>(toGridType<BuildT>(), std::move(buffer));
	} // cuda::createNodeManager

} // namespace cuda

template<typename BuildT, typename BufferT = cuda::DeviceBuffer>
[[deprecated("Use cuda::createNodeManager instead")]] inline
	typename util::enable_if<BufferTraits<BufferT>::hasDeviceDual, NodeManagerHandle<BufferT>>::type
	cudaCreateNodeManager(const NanoGrid<BuildT>* d_grid,
						  const BufferT& pool = BufferT(),
						  cudaStream_t stream = 0) {
	return cuda::createNodeManager<BuildT, BufferT>(d_grid, pool, stream);
}

}
} // namespace ARBD
