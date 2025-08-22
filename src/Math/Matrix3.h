/*********************************************************************
 * @file  Matrix3.h
 *
 * @brief Declaration of templated Matrix3_t class, Metal-compatible version
 * using column-major layout for cross-platform GPU compatibility.
 *********************************************************************/
#pragma once
#include "Backend/Header.h"

#ifdef __METAL_VERSION__
#include "METAL/Matrix3.h"
#else
#ifdef HOST_GUARD
#include "ARBDException.h"
#include "ARBDLogger.h"
#include "Backend/Resource.h"
#include <type_traits>
#endif

#include "Vector3.h"

namespace ARBD {
/**
 * @brief A 3x3 matrix class optimized for Metal GPU kernels and cross-backend use.
 *
 * This matrix is stored in column-major order to align with conventions
 * in OpenGL, Vulkan, and Metal. Simplified for Metal C++17 compatibility.
 */
template<typename T>
struct alignas(16) Matrix3_t {
	using Matrix3 = Matrix3_t<T>;
	using Vector3 = Vector3_t<T>;

	// Simple conditional compilation - Metal supports this
	static constexpr bool is_float = sizeof(T) == sizeof(float);

  private:
	// Column-major storage: three column vectors
	Vector3 cols[3];

  public:
	// Constructors - simplified to avoid complex template resolution
	constexpr Matrix3_t() {
		cols[0] = Vector3(T(1), T(0), T(0));
		cols[1] = Vector3(T(0), T(1), T(0));
		cols[2] = Vector3(T(0), T(0), T(1));
	}

	constexpr Matrix3_t(T s) {
		cols[0] = Vector3(s, T(0), T(0));
		cols[1] = Vector3(T(0), s, T(0));
		cols[2] = Vector3(T(0), T(0), s);
	}

	// Diagonal constructor
	constexpr Matrix3_t(T x, T y, T z) {
		cols[0] = Vector3(x, T(0), T(0));
		cols[1] = Vector3(T(0), y, T(0));
		cols[2] = Vector3(T(0), T(0), z);
	}

	// Column vector constructor
	constexpr Matrix3_t(const Vector3& c0, const Vector3& c1, const Vector3& c2) {
		cols[0] = c0;
		cols[1] = c1;
		cols[2] = c2;
	}

	// Simplified operators - avoid std::common_type_t
	Matrix3 operator*(T s) const {
		return Matrix3(cols[0] * s, cols[1] * s, cols[2] * s);
	}

	// Vector transformation
	Vector3 transform(const Vector3& v) const {
		return cols[0] * v.x + cols[1] * v.y + cols[2] * v.z;
	}

	Vector3 operator*(const Vector3& v) const {
		return transform(v);
	}

	// Matrix multiplication - same type only for Metal safety
	Matrix3 operator*(const Matrix3& m) const {
		Vector3 new_c0 = transform(m.cols[0]);
		Vector3 new_c1 = transform(m.cols[1]);
		Vector3 new_c2 = transform(m.cols[2]);
		return Matrix3(new_c0, new_c1, new_c2);
	}

	// Matrix addition
	Matrix3 operator+(const Matrix3& m) const {
		return Matrix3(cols[0] + m.cols[0], cols[1] + m.cols[1], cols[2] + m.cols[2]);
	}

	// Matrix transpose
	Matrix3 transpose() const {
		Vector3 r0(cols[0].x, cols[1].x, cols[2].x);
		Vector3 r1(cols[0].y, cols[1].y, cols[2].y);
		Vector3 r2(cols[0].z, cols[1].z, cols[2].z);
		return Matrix3(r0, r1, r2);
	}

	// Matrix inverse - simplified
	Matrix3 inverse() const {
		Vector3 c0 = cols[0], c1 = cols[1], c2 = cols[2];
		Vector3 r0 = c1.cross(c2);
		Vector3 r1 = c2.cross(c0);
		Vector3 r2 = c0.cross(c1);
		T inv_det = T(1) / c0.dot(r0);

		return Matrix3(Vector3(r0.x, r1.x, r2.x) * inv_det,
					   Vector3(r0.y, r1.y, r2.y) * inv_det,
					   Vector3(r0.z, r1.z, r2.z) * inv_det);
	}

	// Determinant
	T det() const {
		return cols[0].dot(cols[1].cross(cols[2]));
	}

	// Column accessors
	const Vector3& ex() const {
		return cols[0];
	}
	const Vector3& ey() const {
		return cols[1];
	}
	const Vector3& ez() const {
		return cols[2];
	}

	Vector3& ex() {
		return cols[0];
	}
	Vector3& ey() {
		return cols[1];
	}
	Vector3& ez() {
		return cols[2];
	}

#ifdef HOST_GUARD

	auto to_string() const {
		Matrix3 transposed = transpose();
		return string_format("%2.8f %2.8f %2.8f\n%2.8f %2.8f %2.8f\n%2.8f %2.8f %2.8f",
							 transposed.cols[0].x,
							 transposed.cols[1].x,
							 transposed.cols[2].x,
							 transposed.cols[0].y,
							 transposed.cols[1].y,
							 transposed.cols[2].y,
							 transposed.cols[0].z,
							 transposed.cols[1].z,
							 transposed.cols[2].z);
	}
#endif
};

// Free function for scalar multiplication
template<typename T>
Matrix3_t<T> operator*(T s, const Matrix3_t<T>& m) {
	return m * s;
}

// Common type aliases
using Matrix3f = Matrix3_t<float>;

} // namespace ARBD

// SYCL specialization
#ifdef USE_SYCL
#include <sycl/sycl.hpp>
template<typename T>
struct sycl::is_device_copyable<ARBD::Matrix3_t<T> > : std::true_type {};
#endif
#endif
