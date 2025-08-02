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

#if defined(USE_CUDA)
using BackendPolicy = CUDA::Policy;
#elif defined(USE_SYCL)
using BackendPolicy = SYCL::Policy;
#elif defined(USE_METAL)
using BackendPolicy = METAL::Policy;
#else
#error "No backend selected. Please define USE_CUDA, USE_SYCL, or USE_METAL."
#endif
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
	DEVICE_PTR(T) device_data() {
		return static_cast<DEVICE_PTR(T)>(device_ptr_);
	}

	CONSTANT_PTR(T) constant_data() const {
		return static_cast<CONSTANT_PTR(T)>(device_ptr_);
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
#if !defined(__CUDA_ARCH__) && !defined(__SYCL_DEVICE_ONLY__) && \
	!defined(__METAL_VERSION__) // Host-only logging
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
#if !defined(__CUDA_ARCH__) && !defined(__SYCL_DEVICE_ONLY__) && !defined(__METAL_VERSION__)
			throw std::runtime_error("Copy size exceeds buffer size.");
#else
			// Device code: clamp to safe size
			num_elements = (count_ < num_elements) ? count_ : num_elements;
			if (num_elements == 0)
				return;
#endif
		}
		Policy::copy_from_host(device_ptr_, host_src, num_elements * sizeof(T));
#if !defined(__CUDA_ARCH__) && !defined(__SYCL_DEVICE_ONLY__) && \
	!defined(__METAL_VERSION__) // Host-only logging
		LOGTRACE("Copied {} bytes from host", num_elements * sizeof(T));
#endif
	}

	void copy_device_to_device(const Buffer& src, size_t num_elements) {
		if (num_elements > count_ || num_elements > src.count_) {
#if !defined(__CUDA_ARCH__) && !defined(__SYCL_DEVICE_ONLY__) && !defined(__METAL_VERSION__)
			throw std::runtime_error("Copy size exceeds buffer size.");
#else
			// Device code: clamp to safe size
			num_elements = min(num_elements, min(count_, src.count_));
			if (num_elements == 0)
				return;
#endif
		}
		Policy::copy_device_to_device(device_ptr_, src.device_ptr_, num_elements * sizeof(T));
#if !defined(__CUDA_ARCH__) && !defined(__SYCL_DEVICE_ONLY__) && !defined(__METAL_VERSION__)
		LOGTRACE("Copied {} bytes device-to-device", num_elements * sizeof(T));
#endif
	}
#ifdef USE_METAL
	void bind_to_encoder(MTL::ComputeCommandEncoder* encoder, uint32_t index) const {
		auto* metal_buffer = METAL::Manager::get_metal_buffer_from_ptr(device_ptr_);
		if (!metal_buffer) {
			ARBD::throw_value_error("Failed to get Metal buffer for binding at index {}", index);
		}
		LOGINFO("Binding Metal buffer {} to encoder at index {}", (void*)metal_buffer, index);
		encoder->setBuffer(metal_buffer, 0, index);
	}
#endif
  private:
	void allocate(size_t count) {
		count_ = count;
		if (count_ > 0) {
			device_ptr_ = static_cast<T*>(Policy::allocate(count_ * sizeof(T)));
#if !defined(__CUDA_ARCH__) && !defined(__SYCL_DEVICE_ONLY__) && \
	!defined(__METAL_VERSION__) // Host-only logging
			LOGTRACE("Allocated {} bytes", count_ * sizeof(T));
#endif
		}
	}

	void deallocate() {
		if (device_ptr_) {
			Policy::deallocate(device_ptr_);
			device_ptr_ = nullptr;
#if !defined(__CUDA_ARCH__) && !defined(__SYCL_DEVICE_ONLY__) && \
	!defined(__METAL_VERSION__) // Host-only logging
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
#endif