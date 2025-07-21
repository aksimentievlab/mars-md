/*********************************************************************
 * @file  Matrix3.h
 *
 * @brief Declaration of templated Matrix3_t class.
 *********************************************************************/
#pragma once
#include "ARBDException.h"
#include "ARBDLogger.h"
#include "Backend/Resource.h"
#include "Vector3.h"
#include <cassert>
#include <memory>
#include <type_traits>

// C++20 Concepts for Matrix constraints

namespace ARBD {

// Forward declaration for string_format function
template<typename... Args>
std::string string_format(const char* format, Args... args);

template <typename T, bool is_diag = false, bool check_diag = false>
  requires Arithmetic<T>
struct alignas(16 * sizeof(T)) Matrix3_t {
  using Matrix3 = Matrix3_t<T, is_diag, check_diag>;
  using Vector3 = Vector3_t<T>;

  // Constructors
  HOST DEVICE constexpr Matrix3_t() noexcept { (*this) = Matrix3(T(1)); }
  HOST DEVICE constexpr Matrix3_t(T s) noexcept { (*this) = Matrix3(s, s, s); }
  HOST DEVICE constexpr Matrix3_t(T x, T y, T z) noexcept {
    (*this) = Matrix3(x, T(0), T(0), T(0), y, T(0), T(0), T(0), z);
  }
  HOST DEVICE constexpr Matrix3_t(const T *d) noexcept {
    (*this) = Matrix3(d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8]);
  }
  HOST DEVICE constexpr Matrix3_t(const Vector3 &ex, const Vector3 &ey,
                                  const Vector3 &ez) noexcept {
    (*this) = Matrix3(ex.x, ex.y, ex.z, ey.x, ey.y, ey.z, ez.x, ez.y, ez.z);
  }
  HOST DEVICE constexpr Matrix3_t(T xx, T xy, T xz, T yx, T yy, T yz, T zx,
                                  T zy, T zz) noexcept
      : xx(xx), xy(xy), xz(xz), yx(yx), yy(yy), yz(yz), zx(zx), zy(zy), zz(zz) {
    diag_check();
  }

  HOST DEVICE constexpr void diag_check() const noexcept {
    if constexpr (check_diag && is_diag) {
      assert(xy == T(0) && xz == T(0) && yx == T(0) && yz == T(0) &&
             zx == T(0) && zy == T(0));
    }
  }

  // Scalar multiplication
  template <typename U>
    requires Arithmetic<U>
  HOST DEVICE constexpr auto operator*(U s) const noexcept {
    using TU = ::std::common_type_t<T, U>;
    Matrix3_t<TU, is_diag, check_diag> m;
    m.xx = s * xx;
    m.xy = s * xy;
    m.xz = s * xz;
    m.yx = s * yx;
    m.yy = s * yy;
    m.yz = s * yz;
    m.zx = s * zx;
    m.zy = s * zy;
    m.zz = s * zz;
    return m;
  }

  // Vector transformation
  template <typename U>
    requires Arithmetic<U>
  HOST DEVICE constexpr auto operator*(const Vector3_t<U> &v) const noexcept {
    return this->transform(v);
  }

  // Matrix multiplication
  template <typename U, bool is_diag2, bool check_diag2>
    requires Arithmetic<U>
  HOST DEVICE constexpr auto
  operator*(const Matrix3_t<U, is_diag2, check_diag2> &m) const noexcept {
    return this->transform(m);
  }

  // Matrix addition
  template <typename U, bool is_diag2, bool check_diag2>
    requires Arithmetic<U>
  HOST DEVICE constexpr auto
  operator+(const Matrix3_t<U, is_diag2, check_diag2> &m) const noexcept {
    using TU = ::std::common_type_t<T, U>;
    constexpr bool result_is_diag = is_diag && is_diag2;
    constexpr bool result_check_diag = check_diag || check_diag2;

    Matrix3_t<TU, result_is_diag, result_check_diag> ret;
    ret.xx = xx + m.xx;
    ret.yy = yy + m.yy;
    ret.zz = zz + m.zz;

    if constexpr (!is_diag || !is_diag2) {
      ret.xy = xy + m.xy;
      ret.xz = xz + m.xz;
      ret.yx = yx + m.yx;
      ret.yz = yz + m.yz;
      ret.zx = zx + m.zx;
      ret.zy = zy + m.zy;
    }
    return ret;
  }

  // Matrix subtraction
  template <typename U, bool is_diag2, bool check_diag2>
    requires Arithmetic<U>
  HOST DEVICE constexpr auto
  operator-(const Matrix3_t<U, is_diag2, check_diag2> &m) const noexcept {
    return (*this) + (-m);
  }

  HOST DEVICE constexpr Matrix3 operator-() const noexcept {
    return Matrix3(-xx, -xy, -xz, -yx, -yy, -yz, -zx, -zy, -zz);
  }

  HOST DEVICE constexpr Matrix3 transpose() const noexcept {
    return Matrix3(xx, yx, zx, xy, yy, zy, xz, yz, zz);
  }

  // Vector transformation
  template <typename U>
    requires Arithmetic<U>
  HOST DEVICE constexpr auto transform(const Vector3_t<U> &v) const noexcept {
    using TU = ::std::common_type_t<T, U>;
    Vector3_t<TU> w;
    if constexpr (is_diag) {
      w.x = xx * v.x;
      w.y = yy * v.y;
      w.z = zz * v.z;
    } else {
      w.x = xx * v.x + xy * v.y + xz * v.z;
      w.y = yx * v.x + yy * v.y + yz * v.z;
      w.z = zx * v.x + zy * v.y + zz * v.z;
    }
    return w;
  }

  // Matrix transformation (multiplication)
  template <typename U, bool is_diag2, bool check_diag2>
    requires Arithmetic<U>
  HOST DEVICE constexpr auto
  transform(const Matrix3_t<U, is_diag2, check_diag2> &m) const noexcept {
    using TU = ::std::common_type_t<T, U>;
    constexpr bool result_is_diag = is_diag && is_diag2;
    constexpr bool result_check_diag = check_diag || check_diag2;

    Matrix3_t<TU, result_is_diag, result_check_diag> ret;

    if constexpr (is_diag && is_diag2) {
      // Both diagonal - simple element-wise multiplication
      ret.xx = xx * m.xx;
      ret.yy = yy * m.yy;
      ret.zz = zz * m.zz;
    } else {
      // General matrix multiplication
      ret.xx = xx * m.xx + xy * m.yx + xz * m.zx;
      ret.yy = yx * m.xy + yy * m.yy + yz * m.zy;
      ret.zz = zx * m.xz + zy * m.yz + zz * m.zz;

      if constexpr (!result_is_diag) {
        ret.yx = yx * m.xx + yy * m.yx + yz * m.zx;
        ret.zx = zx * m.xx + zy * m.yx + zz * m.zx;
        ret.xy = xx * m.xy + xy * m.yy + xz * m.zy;
        ret.zy = zx * m.xy + zy * m.yy + zz * m.zy;
        ret.xz = xx * m.xz + xy * m.yz + xz * m.zz;
        ret.yz = yx * m.xz + yy * m.yz + yz * m.zz;
      }
    }
    return ret;
  }

  // Matrix inverse
  HOST DEVICE constexpr Matrix3 inverse() const noexcept {
    if constexpr (is_diag) {
      return Matrix3(T(1) / xx, T(1) / yy, T(1) / zz);
    } else {
      T det_val = det();
      return Matrix3(
          (yy * zz - yz * zy) / det_val, -(xy * zz - xz * zy) / det_val,
          (xy * yz - xz * yy) / det_val, -(yx * zz - yz * zx) / det_val,
          (xx * zz - xz * zx) / det_val, -(xx * yz - xz * yx) / det_val,
          (yx * zy - yy * zx) / det_val, -(xx * zy - xy * zx) / det_val,
          (xx * yy - xy * yx) / det_val);
    }
  }

  // Determinant
  HOST DEVICE constexpr T det() const noexcept {
    if constexpr (is_diag) {
      return xx * yy * zz;
    } else {
      return xx * (yy * zz - yz * zy) - xy * (yx * zz - yz * zx) +
             xz * (yx * zy - yy * zx);
    }
  }

  // Gram-Schmidt orthonormalization
  HOST DEVICE Matrix3 normalized() const noexcept
    requires FloatingPoint<T>
  {
    Vector3 x = ex();
    Vector3 y = ey();
    Vector3 z = x.cross(y);

    auto l = z.length();
    z = (l > T(0)) ? z / l : Vector3(T(0));

    l = x.length();
    x = (l > T(0)) ? x / l : Vector3(T(0));

    y = z.cross(x);
    l = y.length();
    y = (l > T(0)) ? y / l : Vector3(T(0));

    return Matrix3(x, y, z);
  }

  // Extract basis vectors
  HOST DEVICE constexpr Vector3 ex() const noexcept {
    return Vector3(xx, yx, zx);
  }
  HOST DEVICE constexpr Vector3 ey() const noexcept {
    return Vector3(xy, yy, zy);
  }
  HOST DEVICE constexpr Vector3 ez() const noexcept {
    return Vector3(xz, yz, zz);
  }

  // String representation
  auto to_string() const {
    return string_format(
        "%2.8f %2.8f %2.8f\n%2.8f %2.8f %2.8f\n%2.8f %2.8f %2.8f", xx, xy, xz,
        yx, yy, yz, zx, zy, zz);
  }

  // Query functions
  HOST DEVICE static constexpr bool is_diagonal() noexcept { return is_diag; }
  HOST DEVICE static constexpr bool check_diagonal() noexcept {
    return check_diag;
  }

  // Comparison operators
  template <typename U, bool is_diag2, bool check_diag2>
    requires Arithmetic<U>
  HOST DEVICE constexpr bool
  operator==(const Matrix3_t<U, is_diag2, check_diag2> &b) const noexcept {
    return xx == b.xx && xy == b.xy && xz == b.xz && yx == b.yx && yy == b.yy &&
           yz == b.yz && zx == b.zx && zy == b.zy && zz == b.zz;
  }

  template <typename U, bool is_diag2, bool check_diag2>
    requires Arithmetic<U>
  HOST DEVICE constexpr bool
  operator!=(const Matrix3_t<U, is_diag2, check_diag2> &b) const noexcept {
    return !(*this == b);
  }

  // Helper function for testing
  template <typename U, bool is_diag2, bool check_diag2>
  void test_equal(const Matrix3_t<U, is_diag2, check_diag2> &b) const {
    CHECK(xx == b.xx);
    CHECK(xy == b.xy);
    CHECK(xz == b.xz);
    CHECK(yx == b.yx);
    CHECK(yy == b.yy);
    CHECK(yz == b.yz);
    CHECK(zx == b.zx);
    CHECK(zy == b.zy);
    CHECK(zz == b.zz);
  }

  T xx, xy, xz;
  T yx, yy, yz;
  T zx, zy, zz;
};

// Free function for scalar-matrix multiplication
template <typename S, typename T, bool is_diag, bool check_diag>
  requires Arithmetic<S> && Arithmetic<T>
HOST DEVICE constexpr auto
operator*(S s, const Matrix3_t<T, is_diag, check_diag> &m) noexcept {
  return m * s;
}

// Matrix division by scalar
template <typename T, typename S, bool is_diag, bool check_diag>
  requires Arithmetic<T> && Arithmetic<S>
HOST DEVICE constexpr auto operator/(const Matrix3_t<T, is_diag, check_diag> &m,
                                     S s) noexcept {
  return m * (S(1) / s);
}
} // namespace ARBD

//void example_matrix_kernel() {
  //Resource gpu_resource{Resource::SYCL, 0};
  
  //Matrix3Buffer<double> transforms(100, gpu_resource);
  //Vector3Buffer<double> points(100, gpu_resource);
  //Vector3Buffer<double> results(100, gpu_resource);
  
  //kernel_call(gpu_resource,
  //    MultiRef{transforms, points},
  //    MultiRef{results},
  //      100,
  //    [](size_t i, const Matrix3_t<double>* transform, 
  //       const Vector3_t<double>* point, Vector3_t<double>* result) {
  //        // Use your Matrix3_t transform operation
  //        result[i] = transform[i].transform(point[i]);
  //    });
//}