// src/Backend/CUDA/CUDABuffer.h
#pragma once
#include "Backend/CUDA/CUDAManager.h"
#ifdef USE_CUDA
#include "ARBDException.h"
#include "Backend/Resource.h"
// Forward declarations for CUDA memory operations
namespace ARBD {
namespace CUDA {
template<typename T>
void fill_impl(void* dst, T value, size_t num_elements, void* queue, bool sync);
} // namespace CUDA
} // namespace ARBD
#include <cstddef>
#include <cstring>
#include <cuda_runtime.h>

namespace ARBD {
namespace CUDA {

/// Maps a device ordinal to a cudaMemLocation. cudaCpuDeviceId (-1) means host.
inline cudaMemLocation mem_location(int device_id) {
	cudaMemLocation loc{};
	if (device_id == cudaCpuDeviceId) {
		loc.type = cudaMemLocationTypeHost;
		loc.id = 0;
	} else {
		loc.type = cudaMemLocationTypeDevice;
		loc.id = device_id;
	}
	return loc;
}

// The location-based managed-memory API is spelled `_v2` on CUDA 12.2-12.x and
// unsuffixed on CUDA 13, which dropped the old device-ordinal overloads.
// Requires CUDA >= 12.2.
inline cudaError_t
prefetch_async(const void* ptr, size_t bytes, cudaMemLocation loc, cudaStream_t stream) {
#if CUDART_VERSION >= 13000
	return cudaMemPrefetchAsync(ptr, bytes, loc, 0, stream);
#else
	return cudaMemPrefetchAsync_v2(ptr, bytes, loc, 0, stream);
#endif
}

inline cudaError_t
advise(const void* ptr, size_t bytes, cudaMemoryAdvise advice, cudaMemLocation loc) {
#if CUDART_VERSION >= 13000
	return cudaMemAdvise(ptr, bytes, advice, loc);
#else
	return cudaMemAdvise_v2(ptr, bytes, advice, loc);
#endif
}

struct Policy {
	static void*
	allocate(const Resource& resource, size_t bytes, void* queue = nullptr, bool sync = true) {
		if (resource.type() != ResourceType::CUDA) {
			ARBD_Exception(ExceptionType::ValueError,
						   "CUDA Policy requires CUDA resource, got {}",
						   resource.getTypeString());
		}

		// Resource handles device context via ensure_context()
		resource.activate();

		void* ptr = nullptr;
		CUDA_CHECK(cudaMalloc(&ptr, bytes));

		return ptr;
	}

	static void deallocate(void* ptr, void* queue = nullptr, bool sync = true) {
		if (ptr) {
			CUDA_CHECK(cudaFree(ptr));
		}
	}

	static void copy_to_host(void* host_dst,
							 const void* device_src,
							 size_t bytes,
							 void* queue = nullptr,
							 bool sync = false) {
		if (!host_dst || !device_src || bytes == 0)
			return;

		if (sync) {
			CUDA_CHECK(cudaMemcpy(host_dst, device_src, bytes, cudaMemcpyDefault));
		} else {
			cudaStream_t stream = static_cast<cudaStream_t>(queue);
			if (!stream) {
				ARBD_Exception(ExceptionType::RuntimeError, "Async CUDA copy requires a stream");
			}
			CUDA_CHECK(cudaMemcpyAsync(host_dst, device_src, bytes, cudaMemcpyDefault, stream));
		}
	}

	static void copy_from_host(void* device_dst,
							   const void* host_src,
							   size_t bytes,
							   void* queue = nullptr,
							   bool sync = false) {
		if (!device_dst || !host_src || bytes == 0)
			return;

		if (sync) {
			CUDA_CHECK(cudaMemcpy(device_dst, host_src, bytes, cudaMemcpyDefault));
		} else {
			cudaStream_t stream = static_cast<cudaStream_t>(queue);
			if (!stream) {
				ARBD_Exception(ExceptionType::RuntimeError, "Async CUDA copy requires a stream");
			}
			CUDA_CHECK(cudaMemcpyAsync(device_dst, host_src, bytes, cudaMemcpyDefault, stream));
		}
	}

	static void copy_device_to_device(void* dst,
									  const void* src,
									  size_t bytes,
									  void* queue = nullptr,
									  bool sync = false) {
		if (!dst || !src || bytes == 0)
			return;

		if (sync) {
			CUDA_CHECK(cudaMemcpy(dst, src, bytes, cudaMemcpyDefault));
		} else {
			cudaStream_t stream = static_cast<cudaStream_t>(queue);
			if (!stream) {
				ARBD_Exception(ExceptionType::RuntimeError, "Async CUDA copy requires a stream");
			}
			CUDA_CHECK(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDefault, stream));
		}
	}
	template<typename T>
	static void
	fill(void* dst, T value, size_t num_elements, void* queue = nullptr, bool sync = false) {
		fill_impl<T>(dst, value, num_elements, queue, sync);
	}
};
struct PinnedPolicy {
	static void*
	allocate(const Resource& resource, size_t bytes, void* queue = nullptr, bool sync = true) {
		if (resource.type() != ResourceType::CUDA) {
			ARBD_Exception(ExceptionType::ValueError,
						   "CUDA PinnedPolicy requires CUDA resource, got {}",
						   resource.getTypeString());
		}

		void* ptr = nullptr;
		CUDA_CHECK(cudaHostAlloc(&ptr, bytes, cudaHostAllocPortable | cudaHostAllocMapped));
		return ptr;
	}

	static void deallocate(void* ptr, void* queue = nullptr, bool sync = true) {
		if (ptr) {
			CUDA_CHECK(cudaFreeHost(ptr));
		}
	}

	static void upload_to_device(void* device_dst,
								 const void* pinned_src,
								 size_t bytes,
								 const Resource& resource,
								 void* queue = nullptr) {
		cudaStream_t stream = queue ? static_cast<cudaStream_t>(queue)
									: static_cast<cudaStream_t>(resource.get_stream());
		if (!stream) {
			ARBD_Exception(ExceptionType::RuntimeError, "Upload requires a stream");
		}
		CUDA_CHECK(cudaMemcpyAsync(device_dst, pinned_src, bytes, cudaMemcpyHostToDevice, stream));
	}

	static void download_from_device(void* pinned_dst,
									 const void* device_src,
									 size_t bytes,
									 const Resource& resource,
									 void* queue = nullptr) {
		cudaStream_t stream = queue ? static_cast<cudaStream_t>(queue)
									: static_cast<cudaStream_t>(resource.get_stream());
		if (!stream) {
			ARBD_Exception(ExceptionType::RuntimeError, "Download requires a stream");
		}
		CUDA_CHECK(cudaMemcpyAsync(pinned_dst, device_src, bytes, cudaMemcpyDeviceToHost, stream));
	}
};

struct UnifiedPolicy {
	static void*
	allocate(const Resource& resource, size_t bytes, void* queue = nullptr, bool sync = true) {
		if (resource.type() != ResourceType::CUDA) {
			ARBD_Exception(ExceptionType::ValueError,
						   "CUDA UnifiedPolicy requires CUDA resource, got {}",
						   resource.getTypeString());
		}

		void* ptr = nullptr;
		CUDA_CHECK(cudaMallocManaged(&ptr, bytes, cudaMemAttachGlobal));
		return ptr;
	}

	static void deallocate(void* ptr, void* queue = nullptr, bool sync = true) {
		if (ptr) {
			CUDA_CHECK(cudaFree(ptr));
		}
	}

	static void prefetch(void* ptr, size_t bytes, int device_id, void* queue = nullptr) {
		cudaStream_t stream = queue ? static_cast<cudaStream_t>(queue) : 0;
		CUDA_CHECK(prefetch_async(ptr, bytes, mem_location(device_id), stream));
	}

	static void mem_advise(void* ptr, size_t bytes, int advice, int device_id) {
		if (!ptr || bytes == 0)
			return;

		cudaMemoryAdvise cuda_advice =
			(advice == 0) ? cudaMemAdviseSetReadMostly : static_cast<cudaMemoryAdvise>(advice);

		CUDA_CHECK(advise(ptr, bytes, cuda_advice, mem_location(device_id)));
	}

	static void copy_from_host(void* unified_dst,
							   const void* host_src,
							   size_t bytes,
							   void* queue = nullptr,
							   bool sync = false) {
		std::memcpy(unified_dst, host_src, bytes);

		// Optionally prefetch to current device
		int device;
		cudaGetDevice(&device);
		cudaStream_t stream = queue ? static_cast<cudaStream_t>(queue) : 0;
		CUDA_CHECK(prefetch_async(unified_dst, bytes, mem_location(device), stream));
	}

	static void copy_to_host(void* host_dst,
							 const void* unified_src,
							 size_t bytes,
							 void* queue = nullptr,
							 bool sync = false) {
		cudaStream_t stream = queue ? static_cast<cudaStream_t>(queue) : 0;

		// Prefetch to host
		CUDA_CHECK(prefetch_async(unified_src, bytes, mem_location(cudaCpuDeviceId), stream));
		if (stream) {
			CUDA_CHECK(cudaStreamSynchronize(stream));
		} else {
			CUDA_CHECK(cudaDeviceSynchronize());
		}

		std::memcpy(host_dst, unified_src, bytes);
	}

	static void copy_device_to_device(void* dst,
									  const void* src,
									  size_t bytes,
									  void* queue = nullptr,
									  bool sync = false) {
		if (sync) {
			CUDA_CHECK(cudaMemcpy(dst, src, bytes, cudaMemcpyDefault));
		} else {
			cudaStream_t stream = static_cast<cudaStream_t>(queue);
			if (!stream) {
				ARBD_Exception(ExceptionType::RuntimeError, "Async copy requires a stream");
			}
			CUDA_CHECK(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDefault, stream));
		}
	}
};

struct TexturePolicy {
	struct Texture {
		cudaArray_t array{nullptr};
		cudaTextureObject_t textureObject{0};
	};

	static void* allocate(const Resource& resource,
						  size_t width,
						  size_t height,
						  size_t depth,
						  cudaChannelFormatDesc channelDesc,
						  const void* src_data = nullptr,
						  bool is_host_data = false) {
		if (resource.type() != ResourceType::CUDA) {
			ARBD_Exception(ExceptionType::ValueError,
						   "CUDA TexturePolicy requires CUDA resource, got {}",
						   resource.getTypeString());
		}

		Texture* texture = new Texture();

		const unsigned int bits_x = channelDesc.x < 0 ? static_cast<unsigned int>(-channelDesc.x)
													  : static_cast<unsigned int>(channelDesc.x);
		const unsigned int bits_y = channelDesc.y < 0 ? static_cast<unsigned int>(-channelDesc.y)
													  : static_cast<unsigned int>(channelDesc.y);
		const unsigned int bits_z = channelDesc.z < 0 ? static_cast<unsigned int>(-channelDesc.z)
													  : static_cast<unsigned int>(channelDesc.z);
		const unsigned int bits_w = channelDesc.w < 0 ? static_cast<unsigned int>(-channelDesc.w)
													  : static_cast<unsigned int>(channelDesc.w);

		const size_t element_size_bytes =
			(static_cast<size_t>(bits_x + bits_y + bits_z + bits_w)) / 8u;
		const size_t effective_height = (height > 0 ? height : 1);
		const size_t effective_depth = (depth > 0 ? depth : 1);

		cudaResourceDesc resDesc;
		std::memset(&resDesc, 0, sizeof(resDesc));

		if (is_host_data && src_data) {
			resDesc.resType = cudaResourceTypeArray;
			cudaExtent extent = make_cudaExtent(width, height, depth);

			if (depth > 0) {
				CUDA_CHECK(cudaMalloc3DArray(&texture->array, &channelDesc, extent, 0));
			} else if (height > 0) {
				CUDA_CHECK(cudaMallocArray(&texture->array, &channelDesc, width, height, 0));
			} else {
				CUDA_CHECK(cudaMallocArray(&texture->array, &channelDesc, width, 0, 0));
			}
			resDesc.res.array.array = texture->array;

			cudaMemcpy3DParms copyParams;
			std::memset(&copyParams, 0, sizeof(copyParams));
			copyParams.srcPtr = make_cudaPitchedPtr(const_cast<void*>(src_data),
													width * element_size_bytes,
													width,
													effective_height);
			copyParams.extent = make_cudaExtent(width, effective_height, effective_depth);
			copyParams.dstArray = texture->array;
			copyParams.kind = cudaMemcpyHostToDevice;
			CUDA_CHECK(cudaMemcpy3D(&copyParams));
		} else if (src_data) {
			resDesc.resType = cudaResourceTypeLinear;
			resDesc.res.linear.devPtr = const_cast<void*>(src_data);
			resDesc.res.linear.desc = channelDesc;
			resDesc.res.linear.sizeInBytes =
				width * effective_height * effective_depth * element_size_bytes;
		} else {
			delete texture;
			ARBD_Exception(ExceptionType::ValueError, "CUDA TexturePolicy requires source data");
			return nullptr;
		}

		cudaTextureDesc texDesc;
		std::memset(&texDesc, 0, sizeof(texDesc));
		texDesc.addressMode[0] = cudaAddressModeClamp;
		texDesc.addressMode[1] = cudaAddressModeClamp;
		texDesc.addressMode[2] = cudaAddressModeClamp;
		texDesc.filterMode = cudaFilterModePoint;
		texDesc.readMode = cudaReadModeElementType;
		texDesc.normalizedCoords = 0;

		CUDA_CHECK(cudaCreateTextureObject(&texture->textureObject, &resDesc, &texDesc, nullptr));
		return static_cast<void*>(texture);
	}

	static void deallocate(void* ptr, void* queue = nullptr, bool sync = true) {
		if (!ptr)
			return;

		Texture* texture = static_cast<Texture*>(ptr);
		if (texture->textureObject) {
			CUDA_CHECK(cudaDestroyTextureObject(texture->textureObject));
		}
		if (texture->array) {
			CUDA_CHECK(cudaFreeArray(texture->array));
		}
		delete texture;
	}

	static void copy_from_buffer(void* texture_ptr,
								 const void* buffer_ptr,
								 size_t width,
								 size_t height,
								 size_t depth,
								 size_t element_size_bytes) {
		if (!texture_ptr || !buffer_ptr)
			return;

		Texture* texture = static_cast<Texture*>(texture_ptr);
		if (!texture->array) {
			ARBD_Exception(ExceptionType::RuntimeError,
						   "Cannot copy to texture without cudaArray backing");
			return;
		}

		CUDA_CHECK(cudaMemcpy2DToArray(texture->array,
									   0,
									   0,
									   buffer_ptr,
									   width * element_size_bytes,
									   width * element_size_bytes,
									   (height > 0 ? height : 1),
									   cudaMemcpyDeviceToDevice));
	}
};

} // namespace CUDA
} // namespace ARBD
#endif
