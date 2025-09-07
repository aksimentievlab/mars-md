#pragma once
#ifdef USE_CUDA
#include "ARBDException.h"
#include "Backend/CUDA/CUDAManager.h"
#include "Backend/Resource.h"
#include <cstddef>
#include <cstring>
#include <cuda_runtime.h>

namespace ARBD {
namespace CUDA {
struct Policy {
	// queue is a cuda stream
	static void*
	allocate(const Resource& resource, size_t bytes, void* queue = nullptr, bool sync = true) {
		if (resource.type != ResourceType::CUDA) {
			ARBD_Exception(ExceptionType::ValueError,
						   "CUDA Policy requires CUDA resource, got {}",
						   resource.type);
		}

		// Thread-safe device context management
		int old_device;
		CUDA_CHECK(cudaGetDevice(&old_device));
		CUDA_CHECK(cudaSetDevice(static_cast<int>(resource.id)));

		void* ptr = nullptr;

		CUDA_CHECK(cudaMalloc(&ptr, bytes));

		// Restore previous device context
		CUDA_CHECK(cudaSetDevice(old_device));

		return ptr;
	}

	static void deallocate(void* ptr, void* queue = nullptr, bool sync = true) {
		if (ptr) {
			// Use synchronous deallocation for reliable cleanup
			// This ensures proper cleanup without stream dependencies
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
			// Use the provided queue if available, otherwise get a new stream
			cudaStream_t stream = queue ? static_cast<cudaStream_t>(queue)
										: Manager::get_current_device().get_next_stream();
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
			cudaStream_t stream = queue ? static_cast<cudaStream_t>(queue)
										: Manager::get_current_device().get_next_stream();
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
			// Use the provided queue if available, otherwise get a new stream
			cudaStream_t stream = queue ? static_cast<cudaStream_t>(queue)
										: Manager::get_current_device().get_next_stream();
			CUDA_CHECK(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDefault, stream));
		}
	}
};
struct PinnedPolicy {
	static void*
	allocate(const Resource& resource, size_t bytes, void* queue = nullptr, bool sync = true) {
		if (resource.type != ResourceType::CUDA) {
			ARBD_Exception(ExceptionType::ValueError,
						   "CUDA Policy requires CUDA resource, got {}",
						   resource.type);
		}
		void* ptr = nullptr;
		CUDA_CHECK(cudaHostAlloc(&ptr, bytes, cudaHostAllocPortable | cudaHostAllocMapped));

		return ptr;
	}

	static void deallocate(void* ptr, void* queue = nullptr, bool sync = true) {
		// Pinned memory deallocation doesn't need queue or sync parameters
		(void)queue;
		(void)sync;
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
									: Manager::get_device(resource.id).get_next_stream();
		CUDA_CHECK(cudaMemcpyAsync(device_dst, pinned_src, bytes, cudaMemcpyHostToDevice, stream));
	}

	static void download_from_device(void* pinned_dst,
									 const void* device_src,
									 size_t bytes,
									 const Resource& resource,
									 void* queue = nullptr) {
		cudaStream_t stream = queue ? static_cast<cudaStream_t>(queue)
									: Manager::get_device(resource.id).get_next_stream();
		CUDA_CHECK(cudaMemcpyAsync(pinned_dst, device_src, bytes, cudaMemcpyDeviceToHost, stream));
	}
};
struct UnifiedPolicy {
	static void*
	allocate(const Resource& resource, size_t bytes, void* queue = nullptr, bool sync = true) {
		if (resource.type != ResourceType::CUDA) {
			ARBD_Exception(ExceptionType::ValueError,
						   "CUDA Policy requires CUDA resource, got {}",
						   resource.type);
		}
		void* ptr = nullptr;
		CUDA_CHECK(cudaMallocManaged(&ptr, bytes, cudaMemAttachGlobal));
		return ptr;
	}

	static void deallocate(void* ptr, void* queue = nullptr, bool sync = true) {
		// Unified memory deallocation doesn't need queue or sync parameters
		(void)queue;
		(void)sync;
		if (ptr) {
			CUDA_CHECK(cudaFree(ptr));
		}
	}
	static void prefetch(void* ptr, size_t bytes, int device_id, void* queue = nullptr) {
		cudaStream_t stream =
			queue ? static_cast<cudaStream_t>(queue)
				  : (device_id >= 0 ? Manager::get_device(device_id).get_next_stream() : 0);
		CUDA_CHECK(cudaMemPrefetchAsync(ptr, bytes, device_id, stream));
	}

	static void mem_advise(void* ptr, size_t bytes, int advice, int device_id) {
		// Validate parameters before calling cudaMemAdvise
		if (!ptr || bytes == 0) {
			return; // No-op for invalid pointers or zero bytes
		}

		cudaMemoryAdvise cuda_advice =
			(advice == 0) ? cudaMemAdviseSetReadMostly : static_cast<cudaMemoryAdvise>(advice);

		CUDA_CHECK(cudaMemAdvise(ptr, bytes, cuda_advice, device_id));
	}

	static void copy_from_host(void* unified_dst,
							   const void* host_src,
							   size_t bytes,
							   void* queue = nullptr,
							   bool sync = false) {
		std::memcpy(unified_dst, host_src, bytes);
		// Optionally prefetch to the current device to warm it up
		int device;
		cudaGetDevice(&device);
		cudaStream_t stream = queue ? static_cast<cudaStream_t>(queue) : 0;
		CUDA_CHECK(cudaMemPrefetchAsync(unified_dst, bytes, device, stream));
	}

	static void copy_to_host(void* host_dst,
							 const void* unified_src,
							 size_t bytes,
							 void* queue = nullptr,
							 bool sync = false) {
		// Prefetch to the host to ensure data is resident, then copy
		cudaStream_t stream = queue ? static_cast<cudaStream_t>(queue) : 0;
		CUDA_CHECK(
			cudaMemPrefetchAsync(const_cast<void*>(unified_src), bytes, cudaCpuDeviceId, stream));
		if (stream) {
			CUDA_CHECK(cudaStreamSynchronize(stream));
		} else {
			CUDA_CHECK(cudaDeviceSynchronize()); // Sync if default stream
		}
		std::memcpy(host_dst, unified_src, bytes);
	}

	static void copy_device_to_device(void* dst,
									  const void* src,
									  size_t bytes,
									  void* queue = nullptr,
									  bool sync = false) {
		if (sync) {
			// cudaMemcpyDefault handles peer-to-peer automatically
			CUDA_CHECK(cudaMemcpy(dst, src, bytes, cudaMemcpyDefault));
		} else {
			cudaStream_t stream = queue ? static_cast<cudaStream_t>(queue)
										: Manager::get_current_device().get_next_stream();
			CUDA_CHECK(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDefault, stream));
		}
	}
};
} // namespace CUDA
} // namespace ARBD
#endif
