#pragma once

// Address space qualifiers for different backends
#ifdef __METAL_VERSION__
#ifndef DEVICE_PTR
#define DEVICE_PTR(type) device type*
#endif
#ifndef CONSTANT_PTR
#define CONSTANT_PTR(type) constant type*
#endif
#else
#ifndef DEVICE_PTR
#define DEVICE_PTR(type) type*
#endif
#ifndef CONSTANT_PTR
#define CONSTANT_PTR(type) const type*
#endif
#endif

#ifndef __METAL_VERSION__
#include <cstring>
#include <memory>
#include <type_traits>
#include <vector>

#ifdef USE_CUDA
#include "CUDA/CUDAManager.h"
#include <thrust/tuple.h>
#endif
#ifdef USE_SYCL
#include "SYCL/SYCLManager.h"
#endif
#ifdef USE_METAL
#include "METAL/METALManager.h"
#endif

#include "ARBDLogger.h"
#include "Events.h"
#include "Resource.h"

namespace ARBD {

// ============================================================================
// Resource-Aware Memory Policies
// ============================================================================

#ifdef USE_CUDA
namespace CUDA {
struct Policy {
	static void* allocate(const Resource& resource, size_t bytes) {
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

	static void deallocate(void* ptr) {
		if (ptr) {
			CUDA_CHECK(cudaFree(ptr));
		}
	}

	static void copy_to_host(void* host_dst, const void* device_src, size_t bytes) {
		CUDA_CHECK(cudaMemcpy(host_dst, device_src, bytes, cudaMemcpyDeviceToHost));
	}

	static void copy_from_host(void* device_dst, const void* host_src, size_t bytes) {
		CUDA_CHECK(cudaMemcpy(device_dst, host_src, bytes, cudaMemcpyHostToDevice));
	}

	static void copy_device_to_device(void* dst, const void* src, size_t bytes) {
		CUDA_CHECK(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice));
	}
};
struct PinnedPolicy {
	static void* allocate(const Resource& resource, size_t bytes) {
		if (resource.type != ResourceType::CUDA) {
			ARBD_Exception(ExceptionType::ValueError,
						   "CUDA Policy requires CUDA resource, got {}",
						   resource.type);
		}
		void* ptr = nullptr;
		CUDA_CHECK(cudaMallocHost(&ptr, bytes));
		return ptr;
	}

	static void deallocate(void* ptr) {
		if (ptr) {
			CUDA_CHECK(cudaFreeHost(ptr));
		}
	}

	static void copy_to_host(void* host_dst, const void* device_src, size_t bytes) {
		CUDA_CHECK(cudaMemcpy(host_dst, device_src, bytes, cudaMemcpyDeviceToHost));
	}

	static void copy_from_host(void* device_dst, const void* host_src, size_t bytes) {
		CUDA_CHECK(cudaMemcpy(device_dst, host_src, bytes, cudaMemcpyHostToDevice));
	}

	static void copy_device_to_device(void* dst, const void* src, size_t bytes) {
		CUDA_CHECK(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice));
	}
};
struct UnifiedPolicy {
	static void* allocate(const Resource& resource, size_t bytes) {
		if (resource.type != ResourceType::CUDA) {
			ARBD_Exception(ExceptionType::ValueError,
						   "CUDA Policy requires CUDA resource, got {}",
						   resource.type);
		}
		void* ptr = nullptr;
		CUDA_CHECK(cudaMallocManaged(&ptr, bytes));
		return ptr;
	}

	static void deallocate(void* ptr) {
		if (ptr) {
			CUDA_CHECK(cudaFree(ptr));
		}
	}

	static void copy_to_host(void* host_dst, const void* device_src, size_t bytes) {
		CUDA_CHECK(cudaMemcpy(host_dst, device_src, bytes, cudaMemcpyDeviceToHost));
	}

	static void copy_from_host(void* device_dst, const void* host_src, size_t bytes) {
		CUDA_CHECK(cudaMemcpy(device_dst, host_src, bytes, cudaMemcpyHostToDevice));
	}

	static void copy_device_to_device(void* dst, const void* src, size_t bytes) {
		CUDA_CHECK(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice));
	}
};
} // namespace CUDA
#endif

#ifdef USE_SYCL
namespace SYCL {
struct Policy {
	static void* allocate(const Resource& resource, size_t bytes) {
		if (resource.type != ResourceType::SYCL) {
			ARBD_Exception(ExceptionType::ValueError,
						   "SYCL Policy requires SYCL resource, got {}",
						   resource.toString());
		}

		// Get the specific device queue for this resource
		auto& device_manager = Manager::get_device(resource.id);
		auto& queue = device_manager.get_next_queue();

		void* ptr = nullptr; // Initialize the pointer
		SYCL_CHECK(ptr = sycl::malloc_device(bytes, queue));
		if (!ptr) {
			ARBD_Exception(ExceptionType::RuntimeError,
						   "Failed to allocate {} bytes on SYCL device {}",
						   bytes,
						   resource.id);
		}

		return ptr;
	}

	static void deallocate(void* ptr) {
		if (ptr) {
			// For deallocation, we need to use the current queue context
			// since we don't have the original resource information
			auto& queue = Manager::get_current_queue();
			SYCL_CHECK(sycl::free(ptr, queue.get()));
		}
	}

	static void copy_to_host(void* host_dst, const void* device_src, size_t bytes) {
		auto& queue = Manager::get_current_queue();
		SYCL_CHECK(queue.get().memcpy(host_dst, device_src, bytes).wait());
	}

	static void copy_from_host(void* device_dst, const void* host_src, size_t bytes) {
		auto& queue = Manager::get_current_queue();
		SYCL_CHECK(queue.get().memcpy(device_dst, host_src, bytes).wait());
	}

	static void copy_device_to_device(void* dst, const void* src, size_t bytes) {
		auto& queue = Manager::get_current_queue();
		SYCL_CHECK(queue.get().memcpy(dst, src, bytes).wait());
	}

	static sycl::event copy_to_host_async(void* host_dst, const void* device_src, size_t bytes) {
		auto& queue = Manager::get_current_queue();
		// Return the event instead of waiting on it
		return queue.get().memcpy(host_dst, device_src, bytes);
	}

	static sycl::event copy_from_host_async(void* device_dst, const void* host_src, size_t bytes) {
		auto& queue = Manager::get_current_queue();
		// Return the event
		return queue.get().memcpy(device_dst, host_src, bytes);
	}

	static sycl::event copy_device_to_device_async(void* dst, const void* src, size_t bytes) {
		auto& queue = Manager::get_current_queue();
		// Return the event
		return queue.get().memcpy(dst, src, bytes);
	}
};
struct PinnedPolicy {
	static void* allocate(const Resource& resource, size_t bytes) {
		if (resource.type != ResourceType::SYCL) {
			ARBD_Exception(ExceptionType::ValueError,
						   "SYCL Policy requires SYCL resource, got {}",
						   resource.toString());
		}
		auto& device = SYCL::Manager::get_device(resource.id);
		auto& queue = device.get_next_queue();

		void* ptr = nullptr; // Initialize the pointer
		SYCL_CHECK(ptr = sycl::malloc_host(bytes, queue));
		if (!ptr) {
			ARBD_Exception(ExceptionType::SYCLRuntimeError,
						   "Failed to allocate {} bytes of SYCL host memory",
						   bytes);
		}
		return ptr;
	}

	static void deallocate(void* ptr) {
		if (ptr) {
			auto& queue = SYCL::Manager::get_current_queue();
			SYCL_CHECK(sycl::free(ptr, queue));
		}
	}
};
struct UnifiedPolicy {
	static void* allocate(const Resource& resource, size_t bytes) {
		if (resource.type != ResourceType::SYCL) {
			ARBD_Exception(ExceptionType::ValueError,
						   "SYCLUnifiedMemoryPolicy requires a SYCL resource.");
		}
		// Get the queue associated with the target SYCL device
		auto& device = SYCL::Manager::get_device(resource.id);
		auto& queue = device.get_next_queue();

		void* ptr = nullptr; // Initialize the pointer
		SYCL_CHECK(ptr = sycl::malloc_shared(bytes, queue));
		if (!ptr) {
			ARBD_Exception(ExceptionType::SYCLRuntimeError,
						   "Failed to allocate {} bytes of SYCL shared memory",
						   bytes);
		}
		return ptr;
	}

	static void deallocate(void* ptr) {
		if (ptr) {
			auto& queue = SYCL::Manager::get_current_queue();
			SYCL_CHECK(sycl::free(ptr, queue));
		}
	}
};
} // namespace SYCL
#endif

#ifdef USE_METAL
namespace METAL {
struct Policy {
	static void* allocate(const Resource& resource,
						  size_t bytes,
						  MTL::ResourceOptions storage_mode = MTL::ResourceStorageModeShared) {
		if (resource.type != ResourceType::METAL) {
			ARBD_Exception(ExceptionType::ValueError,
						   "Metal Policy requires Metal resource, got {}",
						   resource.toString());
		}

		// Get the specific device for this resource
		auto& device_manager = Manager::get_device(resource.id);
		void* ptr = device_manager.allocate_raw(bytes, storage_mode);

		if (!ptr) {
			ARBD_Exception(ExceptionType::RuntimeError,
						   "Failed to allocate {} bytes on Metal device {}",
						   bytes,
						   resource.id);
		}

		return ptr;
	}

	static void deallocate(void* ptr) {
		if (ptr) {
			Manager::deallocate_raw(ptr);
		}
	}

	static void copy_to_host(void* host_dst, const void* device_src, size_t bytes) {
		MTL::Buffer* mtl_buffer = Manager::get_metal_buffer_from_ptr(const_cast<void*>(device_src));
		if (!mtl_buffer) {
			ARBD_Exception(ExceptionType::MetalRuntimeError,
						   "copy_to_host: Invalid buffer pointer");
		}

		auto& device_manager = Manager::get_current_device();
		auto& queue = device_manager.get_next_queue();
		MTL::CommandBuffer* cmd_buffer =
			static_cast<MTL::CommandBuffer*>(queue.create_command_buffer());
		MTL::BlitCommandEncoder* blit_encoder = cmd_buffer->blitCommandEncoder();

		if (mtl_buffer->storageMode() == MTL::StorageModeShared) {
			blit_encoder->synchronizeResource(mtl_buffer);
			blit_encoder->endEncoding();
			cmd_buffer->commit();
			cmd_buffer->waitUntilCompleted();
			std::memcpy(host_dst, mtl_buffer->contents(), bytes);
		} else {
			MTL::Buffer* staging_buffer =
				device_manager.metal_device()->newBuffer(bytes, MTL::ResourceStorageModeShared);
			blit_encoder->copyFromBuffer(mtl_buffer, 0, staging_buffer, 0, bytes);
			blit_encoder->endEncoding();
			cmd_buffer->commit();
			cmd_buffer->waitUntilCompleted();
			std::memcpy(host_dst, staging_buffer->contents(), bytes);
			staging_buffer->release();
		}
	}

	static void copy_from_host(void* device_dst, const void* host_src, size_t bytes) {
		MTL::Buffer* mtl_buffer = Manager::get_metal_buffer_from_ptr(device_dst);
		if (!mtl_buffer) {
			ARBD_Exception(ExceptionType::MetalRuntimeError,
						   "copy_from_host: Invalid buffer pointer");
		}

		if (mtl_buffer->storageMode() == MTL::StorageModeShared) {
			std::memcpy(mtl_buffer->contents(), host_src, bytes);

			auto& device_manager = Manager::get_current_device();
			auto& queue = device_manager.get_next_queue();
			MTL::CommandBuffer* cmd_buffer =
				static_cast<MTL::CommandBuffer*>(queue.create_command_buffer());
			MTL::BlitCommandEncoder* blit_encoder = cmd_buffer->blitCommandEncoder();
			blit_encoder->synchronizeResource(mtl_buffer);
			blit_encoder->endEncoding();
			cmd_buffer->commit();
			cmd_buffer->waitUntilCompleted();
		} else {
			auto& device_manager = Manager::get_current_device();
			MTL::Buffer* staging_buffer =
				device_manager.metal_device()->newBuffer(bytes, MTL::ResourceStorageModeShared);
			std::memcpy(staging_buffer->contents(), host_src, bytes);

			auto& queue = device_manager.get_next_queue();
			MTL::CommandBuffer* cmd_buffer =
				static_cast<MTL::CommandBuffer*>(queue.create_command_buffer());
			MTL::BlitCommandEncoder* blit_encoder = cmd_buffer->blitCommandEncoder();
			blit_encoder->copyFromBuffer(staging_buffer, 0, mtl_buffer, 0, bytes);
			blit_encoder->endEncoding();
			cmd_buffer->commit();
			cmd_buffer->waitUntilCompleted();
			staging_buffer->release();
		}
	}

	static void copy_device_to_device(void* dst, const void* src, size_t bytes) {
		auto& device_manager = Manager::get_current_device();
		auto& queue = device_manager.get_next_queue();
		MTL::CommandBuffer* cmd_buffer =
			static_cast<MTL::CommandBuffer*>(queue.create_command_buffer());
		MTL::BlitCommandEncoder* blit_encoder = cmd_buffer->blitCommandEncoder();

		MTL::Buffer* src_buffer = Manager::get_metal_buffer_from_ptr(const_cast<void*>(src));
		MTL::Buffer* dst_buffer = Manager::get_metal_buffer_from_ptr(dst);

		if (!src_buffer || !dst_buffer) {
			ARBD_Exception(ExceptionType::MetalRuntimeError,
						   "copy_device_to_device: Invalid buffer pointer(s)");
		}

		blit_encoder->copyFromBuffer(src_buffer, 0, dst_buffer, 0, bytes);
		blit_encoder->endEncoding();
		cmd_buffer->commit();
		cmd_buffer->waitUntilCompleted();
	}
};
} // namespace METAL
#endif

// ============================================================================
// Compile-time backend selection for policies
// ============================================================================

#if defined(USE_CUDA)
using BackendPolicy = CUDA::Policy;
using PinnedPolicy = CUDA::PinnedPolicy;
using UnifiedPolicy = CUDA::UnifiedPolicy;
#elif defined(USE_SYCL)
using BackendPolicy = SYCL::Policy;
using PinnedPolicy = SYCL::PinnedPolicy;
using UnifiedPolicy = SYCL::UnifiedPolicy;
#elif defined(USE_METAL)
using BackendPolicy = METAL::Policy;
using PinnedPolicy = METAL::Policy;
using UnifiedPolicy = METAL::Policy;
#else
#error "No backend selected. Please define USE_CUDA, USE_SYCL, or USE_METAL."
#endif

// ============================================================================
// Production-Ready Buffer Class
// ============================================================================

/**
 * @brief A production-ready buffer class with explicit resource management.
 *
 * This buffer implementation eliminates the global state dependency that causes
 * race conditions in multi-threaded, multi-GPU environments. All memory operations
 * are explicitly tied to a specific Resource, making the code thread-safe and
 * suitable for production deployment.
 *
 * @tparam T The element type.
 * @tparam Policy The memory management policy (CUDA, SYCL, or Metal).
 */
template<typename T, typename Policy>
class Buffer {
  public:
	/**
	 * @brief Default constructor creates an empty buffer with no resource.
	 *
	 * Note: Buffers created this way cannot allocate memory until a resource
	 * is explicitly assigned via resize() or assignment.
	 */
	Buffer() = default;

	/**
	 * @brief BACKWARD COMPATIBLE: Creates a buffer with size only.
	 * Uses the best available device (prioritizes GPU over CPU).
	 *
	 * @param count The number of elements to allocate
	 */
	explicit Buffer(size_t count) : count_(count), resource_(get_best_available_resource()) {
		if (count_ > 0) {
			allocate_on_resource(resource_, count_);
		}
	}

	/**
	 * @brief PRODUCTION CONSTRUCTOR: Creates a buffer tied to a specific resource.
	 *
	 * @param count The number of elements to allocate
	 * @param resource The compute resource (device) to allocate memory on
	 */
	explicit Buffer(size_t count, const Resource& resource) : count_(count), resource_(resource) {
		if (count_ > 0) {
			allocate_on_resource(resource_, count_);
		}
	}

	/**
	 * @brief Copy constructor with explicit resource binding.
	 *
	 * @param other The source buffer to copy from
	 * @param resource The target resource for the new buffer
	 */
	Buffer(const Buffer& other, const Resource& resource)
		: count_(other.count_), resource_(resource) {
		if (count_ > 0) {
			allocate_on_resource(resource_, count_);
			copy_device_to_device(other, count_);
		}
	}

	/**
	 * @brief Copy constructor (preserves source resource).
	 */
	Buffer(const Buffer& other) : resource_(other.resource_), count_(other.count_) {
		if (count_ > 0) {
			allocate_on_resource(resource_, count_);
			copy_device_to_device(other, count_);
		}
	}

	/**
	 * @brief Move constructor.
	 */
	Buffer(Buffer&& other) noexcept
		: resource_(other.resource_), count_(other.count_), device_ptr_(other.device_ptr_) {
		other.count_ = 0;
		other.device_ptr_ = nullptr;
		other.resource_ = Resource{}; // Reset to CPU default
	}

	/**
	 * @brief Copy assignment operator.
	 */
	Buffer& operator=(const Buffer& other) {
		if (this != &other) {
			deallocate();
			resource_ = other.resource_;
			count_ = other.count_;
			if (count_ > 0) {
				allocate_on_resource(resource_, count_);
				copy_device_to_device(other, count_);
			}
		}
		return *this;
	}

	/**
	 * @brief Move assignment operator.
	 */
	Buffer& operator=(Buffer&& other) noexcept {
		if (this != &other) {
			deallocate();
			resource_ = other.resource_;
			count_ = other.count_;
			device_ptr_ = other.device_ptr_;
			other.count_ = 0;
			other.device_ptr_ = nullptr;
			other.resource_ = Resource{}; // Reset to CPU default
		}
		return *this;
	}

	/**
	 * @brief Destructor deallocates the buffer.
	 */
	~Buffer() {
		deallocate();
	}

	/**
	 * @brief Resizes the buffer and potentially changes the resource.
	 *
	 * @param count New number of elements
	 * @param resource The target resource (optional, uses current resource if not specified)
	 */
	void resize(size_t count, const Resource& resource = Resource{}) {
		Resource target_resource = (resource == Resource{}) ? resource_ : resource;
		if (count == count_ && target_resource == resource_) {
			return; // No change needed
		}

		// First, try to allocate the new buffer.
		T* new_ptr = nullptr;
		if (count > 0) {
			new_ptr = static_cast<T*>(Policy::allocate(target_resource, count * sizeof(T)));
			if (!new_ptr) {
				// Allocation failed. The original buffer is untouched.
				// You could throw an exception here to signal the failure.
				// For now, we'll just return, preserving the original buffer.
				return;
			}
		}

		// The new allocation was successful, so it's now safe to deallocate the old buffer.
		if (device_ptr_) {
			Policy::deallocate(device_ptr_);
		}

		// Finally, update the buffer's state to point to the new memory.
		device_ptr_ = new_ptr;
		count_ = count;
		resource_ = target_resource;
	}

	/**
	 * @brief Returns the resource this buffer is allocated on.
	 */
	const Resource& resource() const {
		return resource_;
	}

	/**
	 * @brief Checks if the buffer has a valid resource (always true since CPU is always available).
	 */
	bool has_valid_resource() const {
		return true;
	}

	/**
	 * @brief Returns the raw device pointer.
	 */
	T* data() {
		return device_ptr_;
	}
	T* data() const {
		return device_ptr_;
	}

	/**
	 * @brief Returns device-qualified pointers for kernel use.
	 */
	DEVICE_PTR(T) device_data() {
		return static_cast<DEVICE_PTR(T)>(device_ptr_);
	}

	CONSTANT_PTR(T) constant_data() const {
		return static_cast<CONSTANT_PTR(T)>(device_ptr_);
	}

	/**
	 * @brief Returns the number of elements.
	 */
	size_t size() const {
		return count_;
	}

	/**
	 * @brief Returns the total size in bytes.
	 */
	size_t bytes() const {
		return count_ * sizeof(T);
	}

	/**
	 * @brief Checks if the buffer is empty.
	 */
	bool empty() const {
		return count_ == 0;
	}

	/**
	 * @brief Copy data to host.
	 */
	void copy_to_host(std::vector<T>& host_dst) const {
		host_dst.resize(count_);
		copy_to_host(host_dst.data(), count_);
	}

	void copy_to_host(T* host_dst, size_t num_elements) const {
		if (num_elements > count_) {
			ARBD_Exception(ExceptionType::ValueError, "Copy size exceeds buffer size");
		}
		if (!device_ptr_) {
			ARBD_Exception(ExceptionType::ValueError, "Cannot copy from null buffer");
		}
		Policy::copy_to_host(host_dst, device_ptr_, num_elements * sizeof(T));
#if !defined(__CUDA_ARCH__) && !defined(__SYCL_DEVICE_ONLY__) && !defined(__METAL_VERSION__)
		LOGTRACE("Copied {} bytes to host from {}", num_elements * sizeof(T), resource_.toString());
#endif
	}

	/**
	 * @brief Copy data from host with automatic resize if needed.
	 */
	void copy_from_host(const std::vector<T>& host_src) {
		if (host_src.size() != count_) {
			resize(host_src.size());
		}
		copy_from_host(host_src.data(), host_src.size());
	}

	void copy_from_host(const T* host_src, size_t num_elements) {
		if (num_elements > count_) {
#if !defined(__CUDA_ARCH__) && !defined(__SYCL_DEVICE_ONLY__) && !defined(__METAL_VERSION__)
			ARBD_Exception(ExceptionType::ValueError, "Copy size exceeds buffer size");
#else
			// Device code: clamp to safe size
			num_elements = (count_ < num_elements) ? count_ : num_elements;
			if (num_elements == 0)
				return;
#endif
		}
		if (!device_ptr_) {
			ARBD_Exception(ExceptionType::ValueError, "Cannot copy to null buffer");
		}
		Policy::copy_from_host(device_ptr_, host_src, num_elements * sizeof(T));
#if !defined(__CUDA_ARCH__) && !defined(__SYCL_DEVICE_ONLY__) && !defined(__METAL_VERSION__)
		LOGTRACE("Copied {} bytes from host to {}", num_elements * sizeof(T), resource_.toString());
#endif
	}

	/**
	 * @brief Copy between device buffers.
	 */
	void copy_device_to_device(const Buffer& src, size_t num_elements) {
		if (num_elements > count_ || num_elements > src.count_) {
#if !defined(__CUDA_ARCH__) && !defined(__SYCL_DEVICE_ONLY__) && !defined(__METAL_VERSION__)
			ARBD_Exception(ExceptionType::ValueError, "Copy size exceeds buffer size");
#else
			// Device code: clamp to safe size
			num_elements = (count_ < num_elements) ? count_ : num_elements;
			num_elements = (src.count_ < num_elements) ? src.count_ : num_elements;
			if (num_elements == 0)
				return;
#endif
		}
		if (!device_ptr_ || !src.device_ptr_) {
			ARBD_Exception(ExceptionType::ValueError, "Cannot copy with null buffer(s)");
		}
		Policy::copy_device_to_device(device_ptr_, src.device_ptr_, num_elements * sizeof(T));
#if !defined(__CUDA_ARCH__) && !defined(__SYCL_DEVICE_ONLY__) && !defined(__METAL_VERSION__)
		LOGTRACE("Copied {} bytes device-to-device from {} to {}",
				 num_elements * sizeof(T),
				 src.resource_.toString(),
				 resource_.toString());
#endif
	}

#ifdef USE_METAL
	/**
	 * @brief Bind buffer to Metal compute encoder.
	 */
	void bind_to_encoder(MTL::ComputeCommandEncoder* encoder, uint32_t index) const {
		auto* metal_buffer = METAL::Manager::get_metal_buffer_from_ptr(device_ptr_);
		if (!metal_buffer) {
			ARBD_Exception(ExceptionType::MetalRuntimeError,
						   "Failed to get Metal buffer for binding at index {}",
						   index);
		}
		LOGINFO("Binding Metal buffer {} to encoder at index {}", (void*)metal_buffer, index);
		encoder->setBuffer(metal_buffer, 0, index);
	}
#endif

  private:
	/**
	 * @brief Get the best available resource (prioritizes GPU devices over CPU)
	 * @todo Not implemented.
	 */
	static Resource get_best_available_resource() {
#ifdef USE_SYCL
		try {
			// Try SYCL first if available
			auto& current_device = SYCL::Manager::get_current_device();
			return Resource{ResourceType::SYCL, static_cast<size_t>(current_device.id())};
		} catch (...) {
			// Continue to next option
		}
#endif

#ifdef USE_CUDA
		try {
			// Try CUDA next
			int device;
			if (cudaGetDevice(&device) == cudaSuccess) {
				return Resource{ResourceType::CUDA, static_cast<size_t>(device)};
			}
		} catch (...) {
			// Continue to next option
		}
#endif

#ifdef USE_METAL
		try {
			// Try Metal next
			auto& current_device = METAL::Manager::get_current_device();
			return Resource{ResourceType::METAL, static_cast<size_t>(current_device.id())};
		} catch (...) {
			// Continue to next option
		}
#endif

		// Fallback to CPU only if no GPU devices available
		return Resource{ResourceType::CPU, 0};
	}

	void allocate_on_resource(const Resource& resource, size_t count) {
		count_ = count;
		if (count_ > 0) {
			// Use the resource-aware allocation method
			device_ptr_ = static_cast<T*>(Policy::allocate(resource, count_ * sizeof(T)));
			if (!device_ptr_) {
				ARBD_Exception(ExceptionType::RuntimeError,
							   "Failed to allocate {} bytes on {}",
							   count_ * sizeof(T),
							   resource.toString());
			}
#if !defined(__CUDA_ARCH__) && !defined(__SYCL_DEVICE_ONLY__) && !defined(__METAL_VERSION__)
			LOGTRACE("Allocated {} bytes on {}", count_ * sizeof(T), resource.toString());
#endif
		}
	}

	void deallocate() {
		if (device_ptr_) {
			Policy::deallocate(device_ptr_);
			device_ptr_ = nullptr;
#if !defined(__CUDA_ARCH__) && !defined(__SYCL_DEVICE_ONLY__) && !defined(__METAL_VERSION__)
			LOGTRACE("Deallocated buffer on {}", resource_.toString());
#endif
		}
		count_ = 0;
	}

	Resource resource_{};	 // The compute resource this buffer is allocated on
	size_t count_{0};		 // Number of elements
	T* device_ptr_{nullptr}; // Device memory pointer
};

// ============================================================================
// Convenient Aliases
// ============================================================================

/**
 * @brief A convenient alias for device buffers using the active backend.
 *
 * This hides the policy template parameter since it's determined at compile time
 * by the backend selection (USE_CUDA, USE_SYCL, or USE_METAL).
 */
template<typename T>
using DeviceBuffer = Buffer<T, BackendPolicy>;
template<typename T>
using PinnedBuffer = Buffer<T, PinnedPolicy>;
template<typename T>
using UnifiedBuffer = Buffer<T, UnifiedPolicy>;

// Backend-specific aliases for explicit use cases
#ifdef USE_CUDA
template<typename T>
using CudaBuffer = Buffer<T, CUDA::Policy>;
#endif

#ifdef USE_SYCL
template<typename T>
using SyclBuffer = Buffer<T, SYCL::Policy>;
#endif

#ifdef USE_METAL
template<typename T>
using MetalBuffer = Buffer<T, METAL::Policy>;
#endif

// Helper function to get buffer pointers for kernel launches
template<typename... Buffers, std::size_t... Is>
auto get_buffer_pointers_impl(const std::tuple<Buffers...>& buffer_tuple,
							  std::index_sequence<Is...>) {
	return std::make_tuple(std::get<Is>(buffer_tuple).data()...);
}

template<typename... Buffers>
auto get_buffer_pointers(const std::tuple<Buffers...>& buffer_tuple) {
	return get_buffer_pointers_impl(buffer_tuple, std::make_index_sequence<sizeof...(Buffers)>{});
}

} // namespace ARBD

#endif // __METAL_VERSION__
