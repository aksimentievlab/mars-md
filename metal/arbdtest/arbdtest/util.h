// @HEADER
// *******************************************************************************
//                                OpenRAND                                       *
//   A Performance Portable, Reproducible Random Number Generation Library       *
//                                                                               *
// Copyright (c) 2023, Michigan State University                                 *
//                                                                               *
// Permission is hereby granted, free of charge, to any person obtaining a copy  *
// of this software and associated documentation files (the "Software"), to deal *
// in the Software without restriction, including without limitation the rights  *
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell     *
// copies of the Software, and to permit persons to whom the Software is         *
// furnished to do so, subject to the following conditions:                      *
//                                                                               *
// The above copyright notice and this permission notice shall be included in    *
// all copies or substantial portions of the Software.                           *
//                                                                               *
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR    *
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,      *
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE   *
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER        *
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, *
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE *
// SOFTWARE.                                                                     *
//********************************************************************************
// @HEADER
//  Created by PinYi on 8/3/25.
//

#ifndef OPENRAND_UTIL_H_
#define OPENRAND_UTIL_H_
#define M_PI        3.14159265358979323846264338327950288   /* pi             */
#define M_PI_2      1.57079632679489661923132169163975144   /* pi/2           */
#define M_PI_4      0.785398163397448309615660845819875721  /* pi/4           */
// ============================================================================
// Platform-Specific Macros and Headers
// ============================================================================

#ifdef __METAL_VERSION__
// METAL SHADING LANGUAGE (MSL) COMPILATION
#include <metal_stdlib>
using namespace metal;

#define OPENRAND_DEVICE         // MSL functions are device-only by default
#define OPENRAND_HOST_DEVICE
#define OPENRAND_HOST           // Host-only constructs are not compiled

#else
// HOST (C++) COMPILATION
#include <cmath>
#include <cstdint>
#include <type_traits>

#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
#define OPENRAND_DEVICE      __host__ __device__
#define OPENRAND_HOST_DEVICE __host__ __device__
#define OPENRAND_HOST        __host__
#else
#define OPENRAND_DEVICE
#define OPENRAND_HOST_DEVICE
#define OPENRAND_HOST
#endif

#endif // __METAL_VERSION__


namespace openrand {

// ============================================================================
// Common Constants and Math Functions
// ============================================================================

#ifdef __METAL_VERSION__
constant uint32_t DEFAULT_GLOBAL_SEED = 0xAAAAAAAA;
#else
const uint32_t DEFAULT_GLOBAL_SEED = 0xAAAAAAAA;
#endif

// Generic math functions that dispatch to the correct backend implementation.
template <typename T>
inline OPENRAND_HOST_DEVICE T sin(T x) {
#ifdef __METAL_VERSION__
    return metal::sin(x); // MSL's sin is overloaded for float, half, etc.
#else
    if constexpr (std::is_same_v<T, float>) return sinf(x);
    else return std::sin(x);
#endif
}

template <typename T>
inline OPENRAND_HOST_DEVICE T cos(T x) {
#ifdef __METAL_VERSION__
    return metal::cos(x);
#else
    if constexpr (std::is_same_v<T, float>) return cosf(x);
    else return std::cos(x);
#endif
}

template <typename T>
inline OPENRAND_HOST_DEVICE T log(T x) {
#ifdef __METAL_VERSION__
    return metal::log(x);
#else
    if constexpr (std::is_same_v<T, float>) return logf(x);
    else return std::log(x);
#endif
}

template <typename T>
inline OPENRAND_HOST_DEVICE T sqrt(T x) {
#ifdef __METAL_VERSION__
    return metal::sqrt(x);
#else
    if constexpr (std::is_same_v<T, float>) return sqrtf(x);
    else return std::sqrt(x);
#endif
}


// ============================================================================
// Vector Type Definitions
// ============================================================================

#ifdef __METAL_VERSION__
// For Metal, use the built-in, optimized vector types directly.
// These are just type aliases for clarity and consistency with host code.
template <typename T> using vec2 = metal::vec<T, 2>;
template <typename T> using vec3 = metal::vec<T, 3>;
template <typename T> using vec4 = metal::vec<T, 4>;

using uint2 = metal::uint2;
using uint3 = metal::uint3;
using uint4 = metal::uint4;

using float2 = metal::float2;
using float3 = metal::float3;
using float4 = metal::float4;

// Metal does not have a native double vector type, but can be defined if needed.
// using double2 = metal::vec<double, 2>;

#else
// For the host (C++), define simple struct-based vector types.
template <typename T> struct vec2 { T x, y; };
template <typename T> struct vec3 { T x, y, z; };
template <typename T> struct vec4 { T x, y, z, w; };

// Type aliases for convenience on the host.
using uint2 = vec2<uint32_t>;
using uint3 = vec3<uint32_t>;
using uint4 = vec4<uint32_t>;

using float2 = vec2<float>;
using float3 = vec3<float>;
using float4 = vec4<float>;

using double2 = vec2<double>;
using double3 = vec3<double>;
using double4 = vec4<double>;

#endif // __METAL_VERSION__


// ============================================================================
// Type Traits (CRTP Helper)
// ============================================================================

#ifdef __METAL_VERSION__
// MSL does not support advanced SFINAE/type_traits like std::void_t.
// We use a simplified version that assumes the counter exists for RNGs.
template <typename T>
struct has_counter {
    const bool value = true;
};
#else
// C++17 version for the host compiler.
template <typename T, typename = std::void_t<>>
struct has_counter : std::false_type {};

template <typename T>
struct has_counter<T, std::void_t<decltype(std::declval<T>()._ctr)>>
    : std::true_type {};
#endif // __METAL_VERSION__

}  // namespace openrand

#endif // OPENRAND_UTIL_H_

