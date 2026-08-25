#pragma once
#include "Header.h"

namespace MARS {

template<typename T>
class alignas(2 * sizeof(T)) Vec2 {
  public:
	union {
#if defined(__CUDACC__)
		typename std::conditional<std::is_same<T, int>::value, ::int2, ::float2>::type native;
#elif defined(__SYCL_DEVICE_ONLY__)
		sycl::vec<T, 2> native;
#elif defined(__METAL_VERSION__)
		typename std::conditional<std::is_same<T, int>::value, simd::int2, simd::float2>::type
			native;
#elif defined(HOST_GUARD)
		sx::simd<T, sx::simd_abi::fixed_size<2> > native;
#endif

		struct {
			T x, y;
		};
	};

	// --- Constructors ---
	HOST DEVICE constexpr Vec2() : x(T(0)), y(T(0)) {}
	HOST DEVICE constexpr Vec2(T x_val, T y_val) : x(x_val), y(y_val) {}

	// Explicit copy/move for union compatibility
	HOST DEVICE constexpr Vec2(const Vec2& other) : x(other.x), y(other.y) {}
	HOST DEVICE constexpr Vec2(Vec2&& other) : x(other.x), y(other.y) {}
	HOST DEVICE constexpr Vec2& operator=(const Vec2& other) {
		x = other.x;
		y = other.y;
		return *this;
	}
	HOST DEVICE constexpr Vec2& operator=(Vec2&& other) {
		x = other.x;
		y = other.y;
		return *this;
	}
};

} // namespace MARS

// SYCL specialization
#ifdef USE_SYCL
#include <sycl/sycl.hpp>
template<typename T>
struct sycl::is_device_copyable<MARS::Vec2<T> > : std::true_type {};
#endif
