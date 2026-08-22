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
using sycl::cosh;
using sycl::exp;
using sycl::floor;
using sycl::log;
using sycl::round;
using sycl::sin;
using sycl::sinh;
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

template<typename T>
HOST DEVICE inline T round(T x) {
	if constexpr (sizeof(T) == sizeof(float))
		return ::roundf(x);
	else
		return ::round(x);
}

template<typename T>
HOST DEVICE inline T exp(T x) {
	if constexpr (sizeof(T) == sizeof(float))
		return ::expf(x);
	else
		return ::exp(x);
}

template<typename T>
HOST DEVICE inline T log(T x) {
	if constexpr (sizeof(T) == sizeof(float))
		return ::logf(x);
	else
		return ::log(x);
}

template<typename T>
HOST DEVICE inline T sinh(T x) {
	if constexpr (sizeof(T) == sizeof(float))
		return ::sinhf(x);
	else
		return ::sinh(x);
}

template<typename T>
HOST DEVICE inline T cosh(T x) {
	if constexpr (sizeof(T) == sizeof(float))
		return ::coshf(x);
	else
		return ::cosh(x);
}

#elif defined(__METAL_VERSION__)
// Metal Shading Language
using metal::acos;
using metal::asin;
using metal::atan;
using metal::atan2;
using metal::cos;
using metal::cosh;
using metal::exp;
using metal::floor;
using metal::log;
using metal::sin;
using metal::sinh;
using metal::sqrt;

#else
// Host code - use std library
using std::acos;
using std::asin;
using std::atan;
using std::atan2;
using std::cos;
using std::cosh;
using std::exp;
using std::floor;
using std::log;
using std::round;
using std::sin;
using std::sinh;
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

// ============================================================================
// Cancellation-free differences
// ============================================================================

/**
 * @brief sinh(x) - x
 * @details Direct subtraction loses ~3 significant digits below |x| ~ 1, where
 *          sinh(x) -> x. Uses the series there instead.
 */
template<typename T>
HOST DEVICE inline T sinh_minus_x(T x) {
	if (x >= T(1) || x <= T(-1))
		return sinh(x) - x;
	const T w = x * x;
	return x * w * (T(1.0 / 6) + w * (T(1.0 / 120) + w * (T(1.0 / 5040) + w * T(1.0 / 362880))));
}

/**
 * @brief x cosh(x) - sinh(x)
 * @details Same cancellation as sinh_minus_x; both terms tend to x.
 */
template<typename T>
HOST DEVICE inline T x_cosh_minus_sinh(T x) {
	if (x >= T(1) || x <= T(-1))
		return x * cosh(x) - sinh(x);
	const T w = x * x;
	return x * w * (T(1.0 / 3) + w * (T(1.0 / 30) + w * (T(1.0 / 840) + w * T(1.0 / 45360))));
}

} // namespace math
} // namespace ARBD
