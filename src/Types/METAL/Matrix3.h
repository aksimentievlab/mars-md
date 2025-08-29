/*********************************************************************
 * @file  Matrix3_metal.h
 *
 * @brief Metal-specific implementation of Matrix3_t class.
 * 
 * This implementation avoids all references and uses value passing
 * to comply with Metal's strict address space requirements.
 *********************************************************************/
 #pragma once
 #ifdef __METAL_VERSION__
 #include <metal_stdlib>
 #include "Vector3.h"
 using namespace metal;
 
 namespace ARBD {
 
 /**
  * @brief A 3x3 matrix class optimized for Metal GPU kernels.
  *
  * This matrix is stored in column-major order to align with conventions
  * in OpenGL, Vulkan, and Metal. Uses value passing for Metal compatibility.
  */
 template<typename T>
 struct alignas(16) Matrix3_t {
     using Matrix3 = Matrix3_t<T>;
     using Vector3 = Vector3_t<T>;
 
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
 
     // Column vector constructor - Metal-compatible parameter passing
     constexpr Matrix3_t(Vector3 c0, Vector3 c1, Vector3 c2) {
         cols[0] = c0;
         cols[1] = c1;
         cols[2] = c2;
     }
 
     // Simplified operators - avoid std::common_type_t
     Matrix3 operator*(T s) const {
         return Matrix3(cols[0] * s, cols[1] * s, cols[2] * s);
     }
 
     // Vector transformation - Metal-compatible parameter passing
     Vector3 transform(Vector3 v) const {
         return cols[0] * v.x + cols[1] * v.y + cols[2] * v.z;
     }
 
     Vector3 operator*(Vector3 v) const {
         return transform(v);
     }
 
     // Matrix multiplication - same type only for Metal safety
     Matrix3 operator*(Matrix3 m) const {
         Vector3 new_c0 = transform(m.cols[0]);
         Vector3 new_c1 = transform(m.cols[1]);
         Vector3 new_c2 = transform(m.cols[2]);
         return Matrix3(new_c0, new_c1, new_c2);
     }
 
     // Matrix addition
     Matrix3 operator+(Matrix3 m) const {
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
 
     // Column accessors - Metal-compatible return types (value returns only)
     Vector3 ex() const { return cols[0]; }
     Vector3 ey() const { return cols[1]; }
     Vector3 ez() const { return cols[2]; }
     
     // Setters for Metal compatibility
     void set_ex(Vector3 v) { cols[0] = v; }
     void set_ey(Vector3 v) { cols[1] = v; }
     void set_ez(Vector3 v) { cols[2] = v; }
     
     // Element access
     Vector3 col(int i) const { return cols[i]; }
     void set_col(int i, Vector3 v) { cols[i] = v; }
 };
 
 // Free function for scalar multiplication - Metal-compatible parameter passing
 template<typename T>
 Matrix3_t<T> operator*(T s, Matrix3_t<T> m) {
     return m * s;
 }
 
 // Common type aliases for Metal
 using MetalMatrix3 = Matrix3_t<float>;
 
 } // namespace ARBD
 #endif