#pragma once
#include "Header.h"

namespace ARBD {
namespace math {

// ============================================================================
// Device-agnostic math function wrappers
// ============================================================================

#ifdef __SYCL_DEVICE_ONLY__
// SYCL device code
using sycl::acos;
using sycl::asin;
using sycl::atan;
using sycl::atan2;
using sycl::cos;
using sycl::exp;
using sycl::floor;
using sycl::log;
using sycl::sin;
using sycl::sqrt;

#elif defined(__CUDA_ARCH__)
// CUDA device code - use fast float intrinsics
template<typename T>
HOST DEVICE inline T sin(T x) {
	if constexpr (sizeof(T) == sizeof(float))
		return ::sinf(x);
	else
		return ::sin(x);
}

template<typename T>
HOST DEVICE inline T cos(T x) {
	if constexpr (sizeof(T) == sizeof(float))
		return ::cosf(x);
	else
		return ::cos(x);
}

template<typename T>
HOST DEVICE inline T asin(T x) {
	if constexpr (sizeof(T) == sizeof(float))
		return ::asinf(x);
	else
		return ::asin(x);
}

template<typename T>
HOST DEVICE inline T acos(T x) {
	if constexpr (sizeof(T) == sizeof(float))
		return ::acosf(x);
	else
		return ::acos(x);
}

template<typename T>
HOST DEVICE inline T atan(T x) {
	if constexpr (sizeof(T) == sizeof(float))
		return ::atanf(x);
	else
		return ::atan(x);
}

template<typename T>
HOST DEVICE inline T atan2(T y, T x) {
	if constexpr (sizeof(T) == sizeof(float))
		return ::atan2f(y, x);
	else
		return ::atan2(y, x);
}

template<typename T>
HOST DEVICE inline T sqrt(T x) {
	if constexpr (sizeof(T) == sizeof(float))
		return ::sqrtf(x);
	else
		return ::sqrt(x);
}

template<typename T>
HOST DEVICE inline T floor(T x) {
	if constexpr (sizeof(T) == sizeof(float))
		return ::floorf(x);
	else
		return ::floor(x);
}

#elif defined(__METAL_VERSION__)
// Metal Shading Language
using metal::acos;
using metal::asin;
using metal::atan;
using metal::atan2;
using metal::cos;
using metal::exp;
using metal::floor;
using metal::log;
using metal::sin;
using metal::sqrt;

#else
// Host code - use std library
using std::acos;
using std::asin;
using std::atan;
using std::atan2;
using std::cos;
using std::exp;
using std::floor;
using std::log;
using std::sin;
using std::sqrt;
#endif

// ============================================================================
// Safe inverse trig functions with clamping
// ============================================================================

template<typename T>
HOST DEVICE inline T safe_acos(T x) {
	// Clamp to [-1, 1] to prevent NaN from floating point errors
	if (x < T(-1))
		x = T(-1);
	if (x > T(1))
		x = T(1);
	return acos(x);
}

template<typename T>
HOST DEVICE inline T safe_asin(T x) {
	// Clamp to [-1, 1] to prevent NaN from floating point errors
	if (x < T(-1))
		x = T(-1);
	if (x > T(1))
		x = T(1);
	return asin(x);
}

} // namespace math
} // namespace ARBD
