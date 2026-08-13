/*********************************************************************
 * @file  Vector3_metal.h
 *
 * @brief Metal-specific implementation of Vector3_t class.
 *
 * This implementation avoids all references and uses value passing
 * to comply with Metal's strict address space requirements.
 *********************************************************************/
#pragma once
#ifdef __METAL_VERSION__
#include <metal_stdlib>
using namespace metal;

namespace ARBD {

/**
 * Metal-specific 3D vector utility class.
 * Uses value passing for all parameters to avoid reference issues.
 */
template<typename T>
class alignas(4 * sizeof(T)) Vector3_t {
  public:
	// Constructors
	constexpr Vector3_t() : x(T(0)), y(T(0)), z(T(0)), t(T(0)) {}
	constexpr Vector3_t(T s) : x(s), y(s), z(s), t(s) {}
	constexpr Vector3_t(T x, T y, T z) : x(x), y(y), z(z), t(0) {}
	constexpr Vector3_t(T x, T y, T z, T t) : x(x), y(y), z(z), t(t) {}

	// Copy constructor - still needs to be by reference in Metal
	constexpr Vector3_t(thread const Vector3_t<T>& v) : x(v.x), y(v.y), z(v.z), t(v.t) {}

	// Device address space copy constructor
	constexpr Vector3_t(device const Vector3_t<T>& v) : x(v.x), y(v.y), z(v.z), t(v.t) {}

	// Constant address space copy constructor
	constexpr Vector3_t(constant const Vector3_t<T>& v) : x(v.x), y(v.y), z(v.z), t(v.t) {}

	// Threadgroup address space copy constructor
	constexpr Vector3_t(threadgroup const Vector3_t<T>& v) : x(v.x), y(v.y), z(v.z), t(v.t) {}

	// Backend vector conversion constructor
	template<typename BackendVec>
	constexpr Vector3_t(BackendVec v)
		: x(static_cast<T>(v.x)), y(static_cast<T>(v.y)), z(static_cast<T>(v.z)), t(0) {}

	// Cross product
	template<typename U>
	constexpr auto cross(Vector3_t<U> w) const {
		using TU = T; // Metal limitation - use same type
		Vector3_t<TU> v;
		v.x = y * w.z - z * w.y;
		v.y = z * w.x - x * w.z;
		v.z = x * w.y - y * w.x;
		return v;
	}

	// Assignment methods (void return to avoid reference issues)
	void assign(Vector3_t<T> v) {
		x = v.x;
		y = v.y;
		z = v.z;
		t = v.t;
	}

	void add_assign(Vector3_t<T> v) {
		x += v.x;
		y += v.y;
		z += v.z;
	}

	void sub_assign(Vector3_t<T> v) {
		x -= v.x;
		y -= v.y;
		z -= v.z;
	}

	template<typename S>
	void mul_assign(S s) {
		x *= s;
		y *= s;
		z *= s;
	}

	template<typename S>
	void div_assign(S s) {
		const auto inv = S(1) / s;
		x *= inv;
		y *= inv;
		z *= inv;
	}

	// Unary operators
	constexpr Vector3_t<T> operator-() const {
		return Vector3_t<T>(-x, -y, -z);
	}

	// Binary operators - value passing for parameters
	template<typename U>
	constexpr auto operator+(Vector3_t<U> w) const {
		using TU = T; // Metal limitation
		return Vector3_t<TU>(x + w.x, y + w.y, z + w.z);
	}

	template<typename U>
	constexpr auto operator-(Vector3_t<U> w) const {
		using TU = T;
		return Vector3_t<TU>(x - w.x, y - w.y, z - w.z);
	}

	template<typename U>
	constexpr auto operator*(U s) const {
		using TU = T;
		return Vector3_t<TU>(s * x, s * y, s * z);
	}

	template<typename U>
	constexpr auto operator/(U s) const {
		const auto inv = U(1) / s;
		return (*this) * inv;
	}

	// Dot product
	template<typename U>
	constexpr auto dot(Vector3_t<U> w) const {
		return x * w.x + y * w.y + z * w.z;
	}

	// Length operations
	constexpr auto length2() const {
		return x * x + y * y + z * z;
	}

	auto length() const {
		return sqrt(length2());
	}

	auto rLength() const {
		auto l = length();
		return (l != T(0)) ? T(1) / l : T(0);
	}

	constexpr auto rLength2() const {
		auto l2 = length2();
		return (l2 != T(0)) ? T(1) / l2 : T(0);
	}

	// Element-wise operations
	auto element_floor() const {
		return Vector3_t<T>(floor(x), floor(y), floor(z));
	}

	template<typename U>
	constexpr auto element_mult(const U w[]) const {
		using TU = T;
		return Vector3_t<TU>(x * w[0], y * w[1], z * w[2]);
	}

	template<typename U>
	constexpr auto element_mult(Vector3_t<U> w) const {
		using TU = T;
		return Vector3_t<TU>(x * w.x, y * w.y, z * w.z);
	}

	// Static element-wise operations
	template<typename U>
	static constexpr auto element_mult(Vector3_t<T> v, Vector3_t<U> w) {
		using TU = T;
		return Vector3_t<TU>(v.x * w.x, v.y * w.y, v.z * w.z);
	}

	template<typename U>
	static constexpr auto element_mult(Vector3_t<T> v, const U w[]) {
		using TU = T;
		return Vector3_t<TU>(v.x * w[0], v.y * w[1], v.z * w[2]);
	}

	static auto element_sqrt(Vector3_t<T> w) {
		return Vector3_t<T>(sqrt(w.x), sqrt(w.y), sqrt(w.z));
	}

	// Numeric limits (Metal-compatible values)
	static constexpr T highest() {
		if (sizeof(T) == 4)
			return T(3.40282347e+38f); // FLT_MAX
		return T(1);				   // Fallback for other types
	}

	static constexpr T lowest() {
		if (sizeof(T) == 4)
			return T(-3.40282347e+38f); // -FLT_MAX
		return T(-1);					// Fallback for other types
	}

	// Comparison operators
	template<typename U>
	constexpr bool operator==(Vector3_t<U> b) const {
		return x == b.x && y == b.y && z == b.z && t == b.t;
	}

	template<typename U>
	constexpr bool operator!=(Vector3_t<U> b) const {
		return !(*this == b);
	}

	T x, y, z, t;
};

// Free function operators
template<typename T, typename U>
constexpr auto operator/(U s, Vector3_t<T> v) {
	using TU = T;
	return Vector3_t<TU>(s / v.x, s / v.y, s / v.z);
}

template<typename T>
constexpr auto operator*(float s, Vector3_t<T> v) {
	return v * s;
}

// Helpful index conversion routines
inline Vector3_t<size_t> index_to_ijk(size_t idx, size_t nx, size_t ny, size_t nz) {
	Vector3_t<size_t> res;
	res.z = idx % nz;
	res.y = (idx / nz) % ny;
	res.x = (idx / (ny * nz)) % nx;
	return res;
}

inline Vector3_t<size_t> index_to_ijk(size_t idx, const size_t n[]) {
	return index_to_ijk(idx, n[0], n[1], n[2]);
}

inline Vector3_t<size_t> index_to_ijk(size_t idx, Vector3_t<size_t> n) {
	return index_to_ijk(idx, n.x, n.y, n.z);
}

// Metal type aliases for common usage
using MetalVector3 = Vector3_t<arbd_real>;

} // namespace ARBD
#endif
