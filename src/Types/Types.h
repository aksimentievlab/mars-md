#pragma once
#include "Array.h"
#include "Bitmask.h"
#include "Header.h"
#include "IndexList.h"
#include "Math.h"
#include "Matrix3.h"
#include "TypeName.h"
#include "Vector2.h"
#include "Vector3.h"

namespace ARBD {

// Simplified string formatting function for CUDA compatibility
#ifdef HOST_GUARD
template<typename... Args>
inline std::string string_format(const char* format, Args... args) {
	// Calculate required size
	int size = snprintf(nullptr, 0, format, args...);
	if (size <= 0)
		return std::string();

	// Allocate buffer and format
	size++; // for null terminator
	std::string result(size, '\0');
	snprintf(&result[0], size, format, args...);
	result.resize(size - 1); // remove null terminator
	return result;
}
#endif

// Includes of various types (allows those to be used simply by including Types.h)
template<typename T>
struct GridSample {
	T value;
	Vector3_t<T> gradient; // raw gradient (∂V/∂x); caller negates for force
};

using Vector3 = Vector3_t<float>;
using Matrix3 = Matrix3_t<float>;
using NeighborList = IndexList<morton_t, 27>;
// For 3-component indices (Angles)
using int3 = ARBD::Vector3_t<int>;
// For 4-component indices (Dihedrals)
using int4 = ARBD::Vector3_t<int>;
using float4 = ARBD::Vector3_t<float>;

using int2 = Vec2<int>;
using float2 = Vec2<float>;

using arbd_int = int;
using arbd_real = float;

/**
 * @brief Backend-agnostic atomic add operation
 * @tparam T Arithmetic type (int, float, double, etc.)
 * @param ptr Pointer to the value to add to
 * @param value Value to add
 * @return The old value at ptr (before addition)
 *
 * @warning High contention scenarios will cause performance degradation.
 * Consider using optimized reduction patterns for better performance.
 */
template<typename T>
HOST DEVICE inline void atomic_add(T* ptr, T value) {
#ifdef USE_CUDA
#ifdef __CUDA_ARCH__
	atomicAdd(ptr, value);
#else
	// Host code fallback - not thread-safe but allows template instantiation
	*ptr += value;
#endif
#elif defined(USE_SYCL)
#ifdef __SYCL_DEVICE_ONLY__
	sycl::atomic_ref<T,
					 sycl::memory_order::relaxed,
					 sycl::memory_scope::device,
					 sycl::access::address_space::global_space>(*ptr)
		.fetch_add(value);
#else
	*ptr += value;
#endif
#elif defined(USE_METAL)
	atomic_fetch_add_explicit(reinterpret_cast<device atomic<T>*>(ptr),
							  value,
							  memory_order_relaxed);
#else
	(*(ptr) += (value));
#endif
};

/**
 * @brief Backend-agnostic atomic fetch-add: adds value to *ptr and returns
 *        the value at ptr *before* the add (unlike atomic_add(), which is
 *        void - useful for allocating a slot via a shared counter).
 * @tparam T Arithmetic type (int, unsigned int, float, etc.)
 *
 * @warning Must not be called from a non-template context with a T other
 * than what any backend-specific overload set actually supports (e.g. CUDA's
 * atomicAdd) - being a template itself, unlike Header.h's ATOMIC_ADD macro,
 * this dispatches per-backend without an `if constexpr` on a concrete type,
 * so it has no untaken-branch-still-must-typecheck pitfall to worry about.
 *
 * SYCL branch explicitly specifies address_space::global_space, matching
 * every existing counter/slot-allocation atomic_ref in this codebase (e.g.
 * ZOrderCellNeighborKernel::pair_count, AdaptiveKernels.h) - the default
 * (generic_space) silently misbehaves for USM device pointers on at least
 * the CUDA backend of SYCL (nvptx64-nvidia-cuda), even though it happens to
 * work for atomic_add()'s per-component float adds above.
 */
template<typename T>
HOST DEVICE inline T atomic_fetch_add(T* ptr, T value) {
#ifdef __CUDA_ARCH__
	return atomicAdd(ptr, value);
#elif defined(__SYCL_DEVICE_ONLY__)
	return sycl::atomic_ref<T,
							sycl::memory_order::relaxed,
							sycl::memory_scope::device,
							sycl::access::address_space::global_space>(*(ptr))
		.fetch_add(value);
#elif defined(USE_METAL)
	return atomic_fetch_add_explicit(reinterpret_cast<device atomic<T>*>(ptr),
									 value,
									 memory_order_relaxed);
#else
	const T old = *ptr;
	*ptr += value;
	return old;
#endif
};

// Specialized atomic_add for Vector3_t - do component-wise atomics
template<typename T>
HOST DEVICE inline void atomic_add(Vector3_t<T>* ptr, const Vector3_t<T>& value) {
#ifdef USE_CUDA
#ifdef __CUDA_ARCH__
	atomicAdd(&(ptr->x), value.x);
	atomicAdd(&(ptr->y), value.y);
	atomicAdd(&(ptr->z), value.z);
	atomicAdd(&(ptr->t), value.t);
#else
	*ptr += value;
#endif
#elif defined(USE_SYCL)
	atomic_add(&(ptr->x), value.x);
	atomic_add(&(ptr->y), value.y);
	atomic_add(&(ptr->z), value.z);
	atomic_add(&(ptr->t), value.t);
#elif defined(USE_METAL)
	atomic_add(&(ptr->x), value.x);
	atomic_add(&(ptr->y), value.y);
	atomic_add(&(ptr->z), value.z);
	atomic_add(&(ptr->t), value.t);
#else
	*ptr += value;
#endif
}

template<typename T>
inline T atomic_reduce_batch(const T* local_values, size_t count, T* global_sum) {
	T local_total = T{0};
	for (size_t i = 0; i < count; ++i) {
		local_total += local_values[i];
	}
	atomic_add(global_sum, local_total);
	return local_total;
}
} // namespace ARBD
