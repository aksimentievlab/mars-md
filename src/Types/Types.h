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
	sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device>(*(ptr)) += value;
#elif defined(USE_METAL)
	atomic_fetch_add_explicit(reinterpret_cast<device atomic<T>*>(ptr),
							  value,
							  memory_order_relaxed);
#else
	(*(ptr) += (value));
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

} // namespace ARBD
