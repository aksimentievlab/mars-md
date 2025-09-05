#pragma once
#ifdef USE_SYCL
#include "ARBDException.h"
#include "Backend/Resource.h"
#include "Backend/SYCL/SYCLManager.h"
#include <sycl/sycl.hpp>
#include <cstddef>
#include <cstring>


namespace ARBD {
namespace SYCL {
struct Policy {
	static void*
	allocate(const Resource& resource, size_t bytes, void* queue = nullptr, bool sync = true) {
		if (resource.type != ResourceType::SYCL) {
			ARBD_Exception(ExceptionType::ValueError,
						   "SYCL Policy requires SYCL resource, got {}",
						   resource.toString());
		}

		// Get the specific device queue for this resource
		auto& device_manager = Manager::get_device(resource.id);

		void* ptr = nullptr; // Initialize the pointer
		auto& q = queue ? *static_cast<sycl::queue*>(queue) : device_manager.get_next_queue();
		SYCL_CHECK(ptr = sycl::malloc_device(bytes, q));
		if (!ptr) {
			ARBD_Exception(ExceptionType::RuntimeError,
						   "Failed to allocate {} bytes on SYCL device {}",
						   bytes,
						   resource.id);
		}

		return ptr;
	}

	static void deallocate(void* ptr, void* queue = nullptr, bool sync = true) {
		if (ptr) {
			// For deallocation, we need to use the current queue context
			// since we don't have the original resource information
			auto& current_queue = Manager::get_current_queue();
			SYCL_CHECK(sycl::free(ptr, current_queue.get()));
		}
	}

	static void copy_to_host(void* host_dst,
							 const void* device_src,
							 size_t bytes,
							 void* queue = nullptr,
							 bool sync = false) {
		auto& q = queue ? *static_cast<sycl::queue*>(queue) : Manager::get_current_queue().get();
		if (sync) {
			SYCL_CHECK(q.memcpy(host_dst, device_src, bytes).wait());
		} else {
			SYCL_CHECK(q.memcpy(host_dst, device_src, bytes));
		}
	}

	static void copy_from_host(void* device_dst,
							   const void* host_src,
							   size_t bytes,
							   void* queue = nullptr,
							   bool sync = false) {
		auto& q = queue ? *static_cast<sycl::queue*>(queue) : Manager::get_current_queue().get();
		if (sync) {
			SYCL_CHECK(q.memcpy(device_dst, host_src, bytes).wait());
		} else {
			SYCL_CHECK(q.memcpy(device_dst, host_src, bytes));
		}
	}

	static void copy_device_to_device(void* dst,
									  const void* src,
									  size_t bytes,
									  void* queue = nullptr,
									  bool sync = false) {
		auto& q = queue ? *static_cast<sycl::queue*>(queue) : Manager::get_current_queue().get();
		if (sync) {
			SYCL_CHECK(q.memcpy(dst, src, bytes).wait());
		} else {
			SYCL_CHECK(q.memcpy(dst, src, bytes));
		}
	}
};
struct PinnedPolicy {
	static void*
	allocate(const Resource& resource, size_t bytes, void* queue = nullptr, bool sync = true) {
		if (resource.type != ResourceType::SYCL) {
			ARBD_Exception(ExceptionType::ValueError,
						   "SYCL Policy requires SYCL resource, got {}",
						   resource.toString());
		}
		auto& device = SYCL::Manager::get_device(resource.id);
		auto& q = queue ? *static_cast<sycl::queue*>(queue) : device.get_next_queue();

		void* ptr = nullptr; // Initialize the pointer
		SYCL_CHECK(ptr = sycl::malloc_host(bytes, q));
		if (!ptr) {
			ARBD_Exception(ExceptionType::SYCLRuntimeError,
						   "Failed to allocate {} bytes of SYCL host memory",
						   bytes);
		}
		return ptr;
	}

	static void deallocate(void* ptr, void* queue = nullptr, bool sync = true) {
		if (ptr) {
			auto& queue = SYCL::Manager::get_current_queue();
			SYCL_CHECK(sycl::free(ptr, queue));
		}
	}
	static void upload_to_device(void* device_dst,
								 const void* pinned_src,
								 size_t bytes,
								 const Resource& resource,
								 void* queue = nullptr) {
		auto& q = queue ? *static_cast<sycl::queue*>(queue) : Manager::get_current_queue().get();
		SYCL_CHECK(q.memcpy(device_dst, pinned_src, bytes));
		SYCL_CHECK(q.wait_and_throw());  // Ensure operation completes
	}

	static void download_from_device(void* pinned_dst,
									 const void* device_src,
									 size_t bytes,
									 const Resource& resource,
									 void* queue = nullptr) {
		auto& q = queue ? *static_cast<sycl::queue*>(queue) : Manager::get_current_queue().get();
		SYCL_CHECK(q.memcpy(pinned_dst, device_src, bytes));
		SYCL_CHECK(q.wait_and_throw());  // Ensure operation completes
	}

	static void copy_from_host(void* pinned_dst,
							   const void* host_src,
							   size_t bytes,
							   void* queue = nullptr,
							   bool sync = false) {
		std::memcpy(pinned_dst, host_src, bytes);
	}

	// Copies from this pinned buffer to a standard host buffer.
	static void copy_to_host(void* host_dst,
							 const void* pinned_src,
							 size_t bytes,
							 void* queue = nullptr,
							 bool sync = false) {
		std::memcpy(host_dst, pinned_src, bytes);
	}
	static void copy_device_to_device(void* dst,
									  const void* src,
									  size_t bytes,
									  void* queue = nullptr,
									  bool sync = false) {
		std::memcpy(dst, src, bytes);
	}
};
struct UnifiedPolicy {
	static void*
	allocate(const Resource& resource, size_t bytes, void* queue = nullptr, bool sync = true) {
		if (resource.type != ResourceType::SYCL) {
			ARBD_Exception(ExceptionType::ValueError,
						   "SYCLUnifiedMemoryPolicy requires a SYCL resource.");
		}
		// Get the queue associated with the target SYCL device
		auto& device = SYCL::Manager::get_device(resource.id);
		auto& q = queue ? *static_cast<sycl::queue*>(queue) : device.get_next_queue();

		void* ptr = nullptr; // Initialize the pointer
		SYCL_CHECK(ptr = sycl::malloc_shared(bytes, q));
		if (!ptr) {
			ARBD_Exception(ExceptionType::SYCLRuntimeError,
						   "Failed to allocate {} bytes of SYCL shared memory",
						   bytes);
		}
		return ptr;
	}

	static void deallocate(void* ptr, void* queue = nullptr, bool sync = true) {
		if (ptr) {
			auto& queue = SYCL::Manager::get_current_queue();
			SYCL_CHECK(sycl::free(ptr, queue));
		}
	}
	// Should be default async
	static void prefetch(void* ptr, size_t bytes, int device_id, void* queue = nullptr) {
		// SYCL prefetch is not standard and causes deadlocks in HipSYCL
		// This is a performance hint, so it's safe to make it a no-op
		(void)ptr; (void)bytes; (void)device_id; (void)queue; // Suppress unused parameter warnings
	}

	static void
	mem_advise(void* ptr, size_t bytes, int advice, int device_id, void* queue = nullptr) {
		// SYCL mem_advise is not standard and causes deadlocks in HipSYCL  
		// This is a performance hint, so it's safe to make it a no-op
		(void)ptr; (void)bytes; (void)advice; (void)device_id; (void)queue; // Suppress unused parameter warnings
	}
	static void copy_from_host(void* unified_dst,
							   const void* host_src,
							   size_t bytes,
							   void* queue = nullptr,
							   bool sync = false) {
		std::memcpy(unified_dst, host_src, bytes);
		// Skip prefetch - causes deadlocks in HipSYCL
		(void)queue; (void)sync; // Suppress unused parameter warnings
	}

	static void copy_to_host(void* host_dst,
							 const void* unified_src,
							 size_t bytes,
							 void* queue = nullptr,
							 bool sync = false) {
		// Skip prefetch - causes deadlocks in HipSYCL
		std::memcpy(host_dst, unified_src, bytes);
		(void)queue; (void)sync; // Suppress unused parameter warnings
	}

	static void copy_device_to_device(void* dst,
									  const void* src,
									  size_t bytes,
									  void* queue = nullptr,
									  bool sync = false) {
		auto& q = queue ? *static_cast<sycl::queue*>(queue) : Manager::get_current_queue().get();
		if (sync) {
			q.memcpy(dst, src, bytes).wait();
		} else {
			q.memcpy(dst, src, bytes);
		}
	}
};
} // namespace SYCL
} // namespace ARBD
#endif
