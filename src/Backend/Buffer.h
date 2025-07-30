#pragma once

#include <cstring>
#include <memory>
#include <type_traits>
#include <vector>

// Required Manager Headers for Policies
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
// Backend Policies
// ============================================================================

#ifdef USE_CUDA
/**
 * @brief Policy for CUDA memory operations.
 */
struct CUDAPolicy {
	static void* allocate(size_t bytes) {
		void* ptr = nullptr;
		CUDA_CHECK(cudaMalloc(&ptr, bytes));
#ifndef __CUDA_ARCH__ // Host-only logging
		LOGTRACE("CUDAPolicy: Allocated {} bytes.", bytes);
#endif
		return ptr;
	}

	static void deallocate(void* ptr) {
		if (ptr) {
#ifndef __CUDA_ARCH__ // Host-only logging
			LOGTRACE("CUDAPolicy: Deallocating pointer.");
#endif
			CUDA_CHECK(cudaFree(ptr));
		}
	}

	static void copy_to_host(void* host_dst, const void* device_src, size_t bytes) {
#ifndef __CUDA_ARCH__ // Host-only logging
		LOGTRACE("CUDAPolicy: Copying {} bytes from device to host.", bytes);
#endif
		CUDA_CHECK(cudaMemcpy(host_dst, device_src, bytes, cudaMemcpyDeviceToHost));
	}

	static void copy_from_host(void* device_dst, const void* host_src, size_t bytes) {
#ifndef __CUDA_ARCH__ // Host-only logging
		LOGTRACE("CUDAPolicy: Copying {} bytes from host to device.", bytes);
#endif
		CUDA_CHECK(cudaMemcpy(device_dst, host_src, bytes, cudaMemcpyHostToDevice));
	}

	static void copy_device_to_device(void* device_dst, const void* device_src, size_t bytes) {
#ifndef __CUDA_ARCH__ // Host-only logging
		LOGTRACE("CUDAPolicy: Copying {} bytes from device to device.", bytes);
#endif
		CUDA_CHECK(cudaMemcpy(device_dst, device_src, bytes, cudaMemcpyDeviceToDevice));
	}
};
#endif // USE_CUDA

#ifdef USE_SYCL
/**
 * @brief Policy for SYCL memory operations.
 */
struct SYCLPolicy {
	static void* allocate(size_t bytes) {
		auto& queue = SYCL::SYCLManager::get_current_queue();
		void* ptr = sycl::malloc_device(bytes, queue.get());
		LOGTRACE("SYCLPolicy: Allocated {} bytes.", bytes);
		return ptr;
	}

	static void deallocate(void* ptr) {
		if (ptr) {
			auto& queue = SYCL::SYCLManager::get_current_queue();
			LOGTRACE("SYCLPolicy: Deallocating pointer.");
			sycl::free(ptr, queue.get());
		}
	}

	static void copy_to_host(void* host_dst, const void* device_src, size_t bytes) {
		auto& queue = SYCL::SYCLManager::get_current_queue();
		LOGTRACE("SYCLPolicy: Copying {} bytes from device to host.", bytes);
		queue.get().memcpy(host_dst, device_src, bytes).wait();
	}

	static void copy_from_host(void* device_dst, const void* host_src, size_t bytes) {
		auto& queue = SYCL::SYCLManager::get_current_queue();
		LOGTRACE("SYCLPolicy: Copying {} bytes from host to device.", bytes);
		queue.get().memcpy(device_dst, host_src, bytes).wait();
	}

	static void copy_device_to_device(void* device_dst, const void* device_src, size_t bytes) {
		auto& queue = SYCL::SYCLManager::get_current_queue();
		LOGTRACE("SYCLPolicy: Copying {} bytes from device to device.", bytes);
		queue.get().memcpy(device_dst, device_src, bytes).wait();
	}
};
#endif // USE_SYCL

#ifdef USE_METAL
/**
 * @brief Policy for Metal memory operations.
 */
struct METALPolicy {
	static void* allocate(size_t bytes) {
		auto& device = METAL::METALManager::get_current_device();
		auto& queue = device.get_queue(0);
		auto buffer = METAL::METALManager::allocate_raw(bytes);
		LOGTRACE("METALPolicy: Allocated {} bytes.", bytes);
		return buffer;
	}

	static void deallocate(void* ptr) {
		if (ptr) {
			LOGTRACE("METALPolicy: Deallocating pointer.");
			METAL::METALManager::deallocate_raw(ptr);
		}
	}

	static void copy_to_host(void* host_dst, const void* device_src, size_t bytes) {
		LOGTRACE("METALPolicy: Copying {} bytes from device to host.", bytes);
		std::memcpy(host_dst, device_src, bytes);
	}

	static void copy_from_host(void* device_dst, const void* host_src, size_t bytes) {
		LOGTRACE("METALPolicy: Copying {} bytes from host to device.", bytes);
		std::memcpy(device_dst, host_src, bytes);
	}

	static void copy_device_to_device(void* device_dst, const void* device_src, size_t bytes) {
		LOGTRACE("METALPolicy: Copying {} bytes from device to device.", bytes);
		std::memcpy(device_dst, device_src, bytes);
	}
};
#endif // USE_METAL

// ============================================================================
// Generic Buffer Class
// ============================================================================

/**
 * @brief A generic buffer class that manages device memory using the specified policy.
 *
 * @tparam T The element type.
 * @tparam Policy The memory management policy (e.g., CUDAPolicy, SYCLPolicy).
 */
template<typename T, typename Policy>
class Buffer {
  public:
	/**
	 * @brief Default constructor creates an empty buffer.
	 */
	Buffer() = default;

	/**
	 * @brief Constructor that allocates a buffer for count elements.
	 *
	 * @param count The number of elements to allocate.
	 */
	explicit Buffer(size_t count) : count_(count) {
		allocate(count);
	}

	/**
	 * @brief Copy constructor.
	 */
	Buffer(const Buffer& other) : count_(other.count_) {
		if (count_ > 0) {
			allocate(count_);
			copy_device_to_device(other, count_);
		}
	}

	/**
	 * @brief Move constructor.
	 */
	Buffer(Buffer&& other) noexcept : count_(other.count_), device_ptr_(other.device_ptr_) {
		other.count_ = 0;
		other.device_ptr_ = nullptr;
	}

	/**
	 * @brief Copy assignment operator.
	 */
	Buffer& operator=(const Buffer& other) {
		if (this != &other) {
			deallocate();
			count_ = other.count_;
			if (count_ > 0) {
				allocate(count_);
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
			count_ = other.count_;
			device_ptr_ = other.device_ptr_;
			other.count_ = 0;
			other.device_ptr_ = nullptr;
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
	 * @brief Resizes the buffer to hold count elements.
	 */
	void resize(size_t count) {
		if (count != count_) {
			deallocate();
			allocate(count);
		}
	}

	/**
	 * @brief Returns the raw device pointer.
	 */
	T* data() {
		return device_ptr_;
	}

	/**
	 * @brief Returns the device pointer for kernel access.
	 */
	T* data() const {
		return device_ptr_;
	}

	/**
	 * @brief Returns the number of elements the buffer can hold.
	 */
	size_t size() const {
		return count_;
	}

	/**
	 * @brief Returns the total size of the buffer in bytes.
	 */
	size_t bytes() const {
		return count_ * sizeof(T);
	}

	/**
	 * @brief Checks if the buffer is empty (has zero size).
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
			throw std::runtime_error("Copy size exceeds buffer size.");
		}
		Policy::copy_to_host(host_dst, device_ptr_, num_elements * sizeof(T));
#ifndef __CUDA_ARCH__ // Host-only logging
		LOGTRACE("Copied {} bytes to host", num_elements * sizeof(T));
#endif
	}

	/**
	 * @brief Copy data from host.
	 */
	void copy_from_host(const std::vector<T>& host_src) {
		if (host_src.size() != count_) {
			resize(host_src.size());
		}
		copy_from_host(host_src.data(), host_src.size());
	}

	void copy_from_host(const T* host_src, size_t num_elements) {
		if (num_elements > count_) {
			throw std::runtime_error("Copy size exceeds buffer size.");
		}
		Policy::copy_from_host(device_ptr_, host_src, num_elements * sizeof(T));
#ifndef __CUDA_ARCH__ // Host-only logging
		LOGTRACE("Copied {} bytes from host", num_elements * sizeof(T));
#endif
	}

	void copy_device_to_device(const Buffer& src, size_t num_elements) {
		if (num_elements > count_ || num_elements > src.count_) {
			throw std::runtime_error("Copy size exceeds buffer size.");
		}
		Policy::copy_device_to_device(device_ptr_, src.device_ptr_, num_elements * sizeof(T));
#ifndef __CUDA_ARCH__ // Host-only logging
		LOGTRACE("Copied {} bytes device-to-device", num_elements * sizeof(T));
#endif
	}

  private:
	void allocate(size_t count) {
		count_ = count;
		if (count_ > 0) {
			device_ptr_ = static_cast<T*>(Policy::allocate(count_ * sizeof(T)));
#ifndef __CUDA_ARCH__ // Host-only logging
			LOGTRACE("Allocated {} bytes", count_ * sizeof(T));
#endif
		}
	}

	void deallocate() {
		if (device_ptr_) {
			Policy::deallocate(device_ptr_);
			device_ptr_ = nullptr;
#ifndef __CUDA_ARCH__ // Host-only logging
			LOGTRACE("Deallocated buffer");
#endif
		}
		count_ = 0;
	}

	size_t count_{0};
	T* device_ptr_{nullptr};
};

// ============================================================================
// Compile-Time Backend Selection
// ============================================================================

#if defined(USE_CUDA)
using BackendPolicy = CUDAPolicy;
#elif defined(USE_SYCL)
using BackendPolicy = SYCLPolicy;
#elif defined(USE_METAL)
using BackendPolicy = METALPolicy;
#else
#error "No backend selected. Please define USE_CUDA, USE_SYCL, or USE_METAL."
#endif

/**
 * @brief A convenient alias for the Buffer class using the active backend policy.
 */
template<typename T>
using DeviceBuffer = Buffer<T, BackendPolicy>;
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
