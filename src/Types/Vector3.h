/*********************************************************************
 * @file  Vector3.h
 *
 * @brief Declaration of templated Vector3_t class with conditional Metal support.
 *********************************************************************/
#pragma once
#include "Header.h"
#include "Math.h"

// Metal-specific implementation
#ifdef __METAL_VERSION__
#include "METAL/Vector3.h"
#else

// Standard implementation for CUDA, SYCL, and HOST
#ifdef HOST_GUARD
#include "ARBDException.h"
#include "ARBDLogger.h"
#include <limits>
#include <sstream>
#include <type_traits>
#endif

#ifdef __CUDA_ARCH__
#include <cuda/std/limits>
#include <cuda/std/type_traits>
#endif

namespace ARBD {

/**
 * 3D vector utility class with common operations implemented on CPU and GPU.
 *
 * Implemented with 4D data storage for better GPU alignment; extra
 * data can be stored in fourth variable this->w
 *
 * @tparam T the type of data stored in the four fields, x,y,z,w; T
 * Should usually be float or double.
 */
template<typename T>
class alignas(4 * sizeof(T)) Vector3_t {
  public:
	union {

#if defined(__CUDACC__)
		typename std::conditional<std::is_same<T, int>::value, ::int4, ::float4>::type native;
#elif defined(__SYCL_DEVICE_ONLY__)
		sycl::vec<T, 4> native;
#elif defined(HOST_GUARD)
		sx::simd<T, sx::simd_abi::fixed_size<4> > native;
#endif
		struct {
			T x, y, z, t;
		};
	};
	// Constructors
	HOST DEVICE constexpr Vector3_t() : x(T(0)), y(T(0)), z(T(0)), t(T(0)) {}
	HOST DEVICE constexpr Vector3_t(T s) : x(s), y(s), z(s), t(s) {}
	HOST DEVICE constexpr Vector3_t(const Vector3_t<T>& v) : x(v.x), y(v.y), z(v.z), t(v.t) {}
	HOST DEVICE constexpr Vector3_t(T x, T y, T z) : x(x), y(y), z(z), t(0) {}
	HOST DEVICE constexpr Vector3_t(T x, T y, T z, T t) : x(x), y(y), z(z), t(t) {}
	HOST DEVICE constexpr Vector3_t(Vector3_t<T>&&) = default; // move

	// Backend vector conversion constructor
	template<typename BackendVec>
	HOST DEVICE constexpr Vector3_t(const BackendVec& v)
		: x(static_cast<T>(v.x)), y(static_cast<T>(v.y)), z(static_cast<T>(v.z)), t(0) {}

	// Cross product with template type deduction
	template<typename U>
	HOST DEVICE constexpr auto cross(const Vector3_t<U>& w) const {
#ifdef __CUDA_ARCH__
		using TU = T; // CUDA limitation for device code
#else
		using TU = typename std::common_type<T, U>::type;
#endif
		Vector3_t<TU> v;
		v.x = y * w.z - z * w.y;
		v.y = z * w.x - x * w.z;
		v.z = x * w.y - y * w.x;
		return v;
	}

	// Assignment operators
	HOST DEVICE constexpr T& operator[](int i) {
		if (i == 0)
			return x;
		if (i == 1)
			return y;
		if (i == 2)
			return z;
		if (i == 3)
			return t;
#ifdef HOST_GUARD
		throw_value_error("Invalid index for Vector3_t");
#endif
		return t;
	}

	HOST DEVICE constexpr Vector3_t<T>& operator=(const Vector3_t<T>& v) {
		x = v.x;
		y = v.y;
		z = v.z;
		t = v.t;
		return *this;
	}

	HOST DEVICE constexpr Vector3_t<T>& operator=(Vector3_t<T>&& v) {
		x = v.x;
		y = v.y;
		z = v.z;
		t = v.t;
		return *this;
	}

	HOST DEVICE constexpr Vector3_t<T>& operator+=(const Vector3_t<T>& v) {
		x += v.x;
		y += v.y;
		z += v.z;
		return *this;
	}

	HOST DEVICE constexpr Vector3_t<T>& operator-=(const Vector3_t<T>& v) {
		x -= v.x;
		y -= v.y;
		z -= v.z;
		return *this;
	}

	template<typename S>
	HOST DEVICE constexpr Vector3_t<T>& operator*=(S s) {
		x *= s;
		y *= s;
		z *= s;
		return *this;
	}

	template<typename S>
	HOST DEVICE constexpr Vector3_t<T>& operator/=(S s) {
		const auto inv = S(1) / s;
		x *= inv;
		y *= inv;
		z *= inv;
		return *this;
	}

	HOST DEVICE constexpr Vector3_t<T> operator-() const {
		return Vector3_t<T>(-x, -y, -z);
	}

	// Binary operators with type deduction
	template<typename U>
	HOST DEVICE constexpr auto operator+(const Vector3_t<U>& w) const {
#ifdef __CUDA_ARCH__
		using TU = T; // CUDA limitation for device code
#else
		using TU = typename std::common_type<T, U>::type;
#endif
		return Vector3_t<TU>(x + w.x, y + w.y, z + w.z);
	}

	template<typename U>
	HOST DEVICE constexpr auto operator-(const Vector3_t<U>& w) const {
#ifdef __CUDA_ARCH__
		using TU = T;
#else
		using TU = typename std::common_type<T, U>::type;
#endif
		return Vector3_t<TU>(x - w.x, y - w.y, z - w.z);
	}

	template<typename U>
	HOST DEVICE constexpr auto operator*(U s) const {
#ifdef __CUDA_ARCH__
		using TU = T;
#else
		using TU = typename std::common_type<T, U>::type;
#endif
		return Vector3_t<TU>(s * x, s * y, s * z);
	}

	template<typename U>
	HOST DEVICE constexpr auto operator/(U s) const {
		const auto inv = U(1) / s;
		return (*this) * inv;
	}

	template<typename U>
	HOST DEVICE constexpr auto dot(const Vector3_t<U>& w) const {
		return x * w.x + y * w.y + z * w.z;
	}

	// Length operations
	HOST DEVICE constexpr auto length2() const {
		return x * x + y * y + z * z;
	}

	HOST DEVICE T angle_between(const Vector3_t<T>& other) const {
		T dot_prod = dot(other);
		T len_product = length() * other.length();
		T cos_angle = dot_prod / len_product;
		return math::safe_acos(cos_angle);
	}

	HOST DEVICE auto length() const {
		return math::sqrt(length2());
	}

	HOST DEVICE static auto element_sqrt(const Vector3_t<T>& w) {
		return Vector3_t<T>(math::sqrt(w.x), math::sqrt(w.y), math::sqrt(w.z));
	}

	HOST DEVICE auto element_floor() const {
		return Vector3_t<T>(math::floor(x), math::floor(y), math::floor(z));
	}

	HOST DEVICE auto rLength() const {
		auto l = length();
		return (l != T(0)) ? T(1) / l : T(0);
	}

	HOST DEVICE constexpr auto rLength2() const {
		auto l2 = length2();
		return (l2 != T(0)) ? T(1) / l2 : T(0);
	}

	template<typename U>
	HOST DEVICE constexpr auto element_mult(const U w[]) const {
#ifdef __CUDA_ARCH__
		using TU = T;
#elif defined(__SYCL_DEVICE_ONLY__)
		using TU = typename std::common_type<T, U>::type;
#else
		using TU = typename std::common_type<T, U>::type;
#endif
		return Vector3_t<TU>(x * w[0], y * w[1], z * w[2]);
	}

	template<typename U>
	HOST DEVICE constexpr auto element_mult(const Vector3_t<U>& w) const {
#ifdef __CUDA_ARCH__
		using TU = T;
#elif defined(__SYCL_DEVICE_ONLY__)
		using TU = typename std::common_type<T, U>::type;
#else
		using TU = typename std::common_type<T, U>::type;
#endif
		return Vector3_t<TU>(x * w.x, y * w.y, z * w.z);
	}

	// Static element-wise operations
	template<typename U>
	HOST DEVICE static constexpr auto element_mult(const Vector3_t<T>& v, const Vector3_t<U>& w) {
#ifdef __CUDA_ARCH__
		using TU = T;
#elif defined(__SYCL_DEVICE_ONLY__)
		using TU = typename std::common_type<T, U>::type;
#else
		using TU = typename std::common_type<T, U>::type;
#endif
		return Vector3_t<TU>(v.x * w.x, v.y * w.y, v.z * w.z);
	}

	template<typename U>
	HOST DEVICE static constexpr auto element_mult(const Vector3_t<T>& v, const U w[]) {
#ifdef __CUDA_ARCH__
		using TU = T;
#elif defined(__SYCL_DEVICE_ONLY__)
		using TU = typename std::common_type<T, U>::type;
#else
		using TU = typename std::common_type<T, U>::type;
#endif
		return Vector3_t<TU>(v.x * w[0], v.y * w[1], v.z * w[2]);
	}

	// Numeric limits
	HOST DEVICE static constexpr T highest() {
#ifdef __CUDA_ARCH__
		return cuda::std::numeric_limits<T>::max();
#else
		return std::numeric_limits<T>::max();
#endif
	}

	HOST DEVICE static constexpr T lowest() {
#ifdef __CUDA_ARCH__
		return cuda::std::numeric_limits<T>::lowest();
#else
		return std::numeric_limits<T>::lowest();
#endif
	}

	// Comparison operators
	template<typename U>
	HOST DEVICE constexpr bool operator==(const Vector3_t<U>& b) const {
		return x == b.x && y == b.y && z == b.z && t == b.t;
	}

	template<typename U>
	HOST DEVICE constexpr bool operator!=(const Vector3_t<U>& b) const {
		return !(*this == b);
	}
#ifdef HOST_GUARD
	HOST void print() const;

	std::string to_string() const;
#endif
};

// Free function operators
template<typename T, typename U>
HOST DEVICE constexpr T operator/(U s, const Vector3_t<T>& v) {
#ifdef __CUDA_ARCH__
	using TU = T;
#elif defined(__SYCL_DEVICE_ONLY__)
	using TU = typename std::common_type<T, U>::type;
#else
	using TU = typename std::common_type<T, U>::type;
#endif
	return Vector3_t<TU>(s / v.x, s / v.y, s / v.z);
}

template<typename T>
HOST DEVICE constexpr auto operator*(const float& s, const Vector3_t<T>& v) {
	return v * s;
}

// Helpful index conversion routines
HOST DEVICE inline Vector3_t<idx_t> index_to_ijk(idx_t idx, idx_t nx, idx_t ny, idx_t nz) {
	Vector3_t<idx_t> res;
	res.z = idx % nz;
	res.y = (idx / nz) % ny;
	res.x = (idx / (ny * nz)) % nx;
	return res;
}

HOST DEVICE inline Vector3_t<idx_t> index_to_ijk(idx_t idx, const idx_t n[]) {
	return index_to_ijk(idx, n[0], n[1], n[2]);
}

HOST DEVICE inline Vector3_t<idx_t> index_to_ijk(idx_t idx, const Vector3_t<idx_t>& n) {
	return index_to_ijk(idx, n.x, n.y, n.z);
}

} // namespace ARBD

// Backend-specific specializations
#ifdef USE_SYCL
#include <sycl/sycl.hpp>
template<typename T>
struct sycl::is_device_copyable<ARBD::Vector3_t<T> > : std::true_type {};
#endif

// Provide common type for vectors (only when std:: is available)
#if !defined(__SYCL_DEVICE_ONLY__) && !defined(__CUDA_ARCH__)
namespace std {
template<typename T, typename U>
struct common_type<ARBD::Vector3_t<T>, ARBD::Vector3_t<U> > {
	using type = ARBD::Vector3_t<typename common_type<T, U>::type>;
};
} // namespace std
#endif

// fmt::formatter specialization for Vector3_t (host-only)
#ifdef HOST_GUARD
#ifndef FMT_HEADER_ONLY
#define FMT_HEADER_ONLY
#endif
#include "../extern/fmt/include/fmt/format.h"
namespace fmt {
template<typename T>
struct formatter<ARBD::Vector3_t<T> > {
	template<typename ParseContext>
	constexpr auto parse(ParseContext& ctx) {
		return ctx.begin();
	}

	template<typename FormatContext>
	auto format(const ARBD::Vector3_t<T>& v, FormatContext& ctx) const {
		return format_to(ctx.out(),
						 "({:.3f}, {:.3f}, {:.3f})",
						 static_cast<float>(v.x),
						 static_cast<float>(v.y),
						 static_cast<float>(v.z));
	}
};
} // namespace fmt
#endif // HOST_GUARD

#endif // __METAL_VERSION__
