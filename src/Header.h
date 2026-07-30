#pragma once
/**
 * @file Header.h
 * @brief Common macros for all backends.
 * @version 0.1
 * @date 2025-08-22
 */

#define _GLIBCXX_USE_CXX11_ABI 1
#include "Constants.h"

#ifdef __CUDACC__
#define HOST __host__
#define DEVICE __device__
#define KERNEL_FUNC __device__
#include <vector_types.h>
#else
#define HOST
#define DEVICE
#endif

#ifdef __METAL_VERSION__
#include <metal_stdlib>
#define KERNEL_FUNC [[kernel]]
#endif

// Only include SYCL headers when actually compiling with SYCL support
// __SYCL_COMPILER_VERSION is defined by icpx when -fsycl is used
#if defined(USE_SYCL) && (defined(__SYCL_COMPILER_VERSION) || defined(__SYCL_DEVICE_ONLY__))
#include <sycl/sycl.hpp>
#include <type_traits>
#endif

#ifdef __CUDA_ARCH__
#include <cuda.h> // Or <cuda.h> depending on CUDAManager's needs
#include <cuda_runtime.h>
#endif

#if !defined(__CUDA_ARCH__) && !defined(__SYCL_DEVICE_ONLY__) && !defined(__METAL_VERSION__)
#define HOST_GUARD
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>
#ifndef __CUDACC__
#include <experimental/simd>
#include <type_traits>
namespace sx = std::experimental;
#endif

namespace ARBD {

/**
 * @brief Resolve file path relative to a configuration file
 * @param file_path The file path to resolve (can be absolute, relative, or ~/ home path)
 * @param config_file_path The path to the configuration file
 * @return Resolved absolute path
 */
inline std::string resolve_file_path(const std::string& file_path,
									 const std::string& config_file_path) {
	// If path is already absolute, return as-is
	if (!file_path.empty() && file_path[0] == '/') {
		return file_path;
	}

	// If path starts with ~, expand home directory
	if (!file_path.empty() && file_path[0] == '~') {
		const char* home = getenv("HOME");
		if (home) {
			return std::string(home) + file_path.substr(1);
		}
		return file_path; // Fallback if HOME not set
	}

	// For relative paths, resolve relative to config file directory
	std::string config_dir = config_file_path;
	size_t last_slash = config_dir.find_last_of('/');
	if (last_slash != std::string::npos) {
		config_dir = config_dir.substr(0, last_slash + 1);
	} else {
		config_dir = "./"; // Current directory if no path separators
	}

	return config_dir + file_path;
}

} // namespace ARBD

#elif defined(__CUDACC__) && !defined(__CUDA_ARCH__)
// Host code in CUDA files - need to include string for host-side usage
#include <string>
namespace ARBD {
inline std::string resolve_file_path(const std::string&, const std::string&) {
	return ""; // Stub for host code in .cu files
}
} // namespace ARBD
#elif defined(USE_SYCL) && defined(__SYCL_COMPILER_VERSION) && !defined(__SYCL_DEVICE_ONLY__)
// SYCL host code - need to include string and filesystem for host-side usage
#include <cstdlib>
#include <filesystem>
#include <string>
namespace ARBD {
/**
 * @brief Resolve file path relative to a configuration file
 * @param file_path The file path to resolve (can be absolute, relative, or ~/ home path)
 * @param config_file_path The path to the configuration file
 * @return Resolved absolute path
 */
inline std::string resolve_file_path(const std::string& file_path,
									 const std::string& config_file_path) {
	// If path is already absolute, return as-is
	if (!file_path.empty() && file_path[0] == '/') {
		return file_path;
	}

	// If path starts with ~, expand home directory
	if (!file_path.empty() && file_path[0] == '~') {
		const char* home = getenv("HOME");
		if (home) {
			return std::string(home) + file_path.substr(1);
		}
		return file_path; // Fallback if HOME not set
	}

	// For relative paths, resolve relative to config file directory
	std::string config_dir = config_file_path;
	size_t last_slash = config_dir.find_last_of('/');
	if (last_slash != std::string::npos) {
		config_dir = config_dir.substr(0, last_slash + 1);
	} else {
		config_dir = "./"; // Current directory if no path separators
	}

	return config_dir + file_path;
}
} // namespace ARBD
#elif defined(__SYCL_DEVICE_ONLY__)
// SYCL device compilation: provide a stub to satisfy parsing
#include <string>
namespace ARBD {
inline std::string resolve_file_path(const std::string&, const std::string&) {
	return "";
}
} // namespace ARBD
#else
// Device compilation: no declaration needed
namespace ARBD {
// resolve_file_path not available in device code
} // namespace ARBD
#endif

#ifndef KERNEL_FUNC
#define KERNEL_FUNC
#endif

#if defined(__CUDACC__)
// For CUDA, atomicAdd is a built-in function for floats
#define ATOMIC_ADD(ptr, val) atomicAdd((ptr), (val))
#elif defined(__SYCL_DEVICE_ONLY__)
// For SYCL: fetch_add available for integral types, CAS for floating-point
#define ATOMIC_ADD(ptr, val)                                                                      \
	([&]() -> std::remove_reference_t<decltype(*(ptr))> {                                         \
		using value_type = std::remove_reference_t<decltype(*(ptr))>;                             \
		if constexpr (std::is_same_v<value_type, float>) {                                        \
			auto atomic_ref =                                                                     \
				sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device>( \
					*(ptr));                                                                      \
			return atomic_ref.fetch_add(val);                                                     \
		} else if constexpr (std::is_integral_v<value_type>) {                                    \
			auto atomic_ref = sycl::atomic_ref<value_type,                                        \
											   sycl::memory_order::relaxed,                       \
											   sycl::memory_scope::device>(*(ptr));               \
			return atomic_ref.fetch_add(val);                                                     \
		} else {                                                                                  \
			auto atomic_ref = sycl::atomic_ref<value_type,                                        \
											   sycl::memory_order::relaxed,                       \
											   sycl::memory_scope::device>(*(ptr));               \
			auto old_val = atomic_ref.load();                                                     \
			while (!atomic_ref.compare_exchange_weak(old_val, old_val + (val))) {                 \
			}                                                                                     \
			return old_val;                                                                       \
		}                                                                                         \
	}())
#elif defined(__METAL_VERSION__)
#define ATOMIC_ADD(ptr, val)                                                              \
	atomic_fetch_add_explicit(                                                            \
		reinterpret_cast<device atomic<std::remove_reference_t<decltype(*(ptr))>>*>(ptr), \
		val,                                                                              \
		memory_order_relaxed)
#else
#define ATOMIC_ADD(ptr, val) (*(ptr) += (val))
#endif

// Suppress narrowing conversion warnings
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wc++11-narrowing"
#endif

#ifdef __METAL_VERSION__
#include <metal_stdlib>
using namespace metal;
template<typename T>
using device_ptr = device T*;
template<typename T>
using constant_ptr = constant T*;
template<typename T>
using thread_ptr = thread T*;
template<typename T>
using threadgroup_ptr = threadgroup T*;
#ifndef DEVICE_PTR
#define DEVICE_PTR(T) device T*
#endif
#ifndef CONSTANT_PTR
#define CONSTANT_PTR(T) constant T*
#endif
#ifndef THREAD_PTR
#define THREAD_PTR(T) thread T*
#endif
#ifndef THREADGROUP_PTR
#define THREADGROUP_PTR(T) threadgroup T*
#endif
#else
// Address space pointer macros for non-Metal backends
#ifndef DEVICE_PTR
#define DEVICE_PTR(T) T*
#endif
#ifndef CONSTANT_PTR
#define CONSTANT_PTR(T) const T*
#endif
#ifndef THREAD_PTR
#define THREAD_PTR(T) T*
#endif
#ifndef THREADGROUP_PTR
#define THREADGROUP_PTR(T) T*
#endif
#endif

#if defined(__METAL_VERSION__)
#ifndef __restrict__
#define __restrict__ __restrict
#endif
#elif !defined(__restrict__)
#define __restrict__
#endif

constexpr inline short NUM_QUEUES = 4;
// bump to 5 for cross-GPU gathering streams.

/**
 * @brief Optimized reduction helper for scenarios with many threads
 *
 * This function provides a pattern for reducing atomic contention by using
 * local accumulation and reduced frequency of atomic operations.
 *
 * @tparam T Arithmetic type
 * @param local_values Array of local values to reduce
 * @param count Number of local values
 * @param global_sum Pointer to global accumulator
 * @return Local thread's contribution to the sum
 */
template<typename T>
inline T atomic_reduce_batch(const T* local_values, size_t count, T* global_sum) {
	T local_total = T{0};
	for (size_t i = 0; i < count; ++i) {
		local_total += local_values[i];
	}
	atomic_add(global_sum, local_total);
	return local_total;
}
using idx_t = size_t;
using patch_t = size_t;
using particle_t = size_t;
using device_id_t = unsigned short;
using grid_t = size_t; // grid ids
using morton_t = uint32_t;
using coord_t = uint32_t;
