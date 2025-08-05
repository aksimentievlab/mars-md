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

#ifndef OPENRAND_UTIL_H_
#define OPENRAND_UTIL_H_
#include "Backend/Header.h"

#ifdef __METAL_VERSION__
#include <metal_stdlib>
using namespace metal;
#else
#include <cmath>
#include <cstdint>
#include <type_traits>
#endif

// Metal-aware device annotations
#ifdef __METAL_VERSION__
#define OPENRAND_DEVICE
#define OPENRAND_HOST_DEVICE
#define OPENRAND_HOST
#elif defined(__CUDA_ARCH__)
#define OPENRAND_DEVICE __host__ __device__
#define OPENRAND_HOST_DEVICE __host__ __device__
#define OPENRAND_HOST __host__
#elif defined(__HIP_DEVICE_COMPILE__)
#define OPENRAND_DEVICE __device__ __host__
#define OPENRAND_HOST_DEVICE __device__ __host__
#define OPENRAND_HOST __host__
#elif defined(__SYCL_DEVICE_ONLY__)
#define OPENRAND_DEVICE
#define OPENRAND_HOST_DEVICE
#define OPENRAND_HOST
#else
#define OPENRAND_DEVICE
#define OPENRAND_HOST_DEVICE
#define OPENRAND_HOST
#endif

namespace openrand {

// NOTE: nvcc compiler replaces floating point variants with cuda built-in
// versions

constexpr uint32_t DEFAULT_GLOBAL_SEED =
    0xAAAAAAAA;  // equal number of 0 and 1 bits

template <typename T>
inline OPENRAND_DEVICE T sin(T x) {
#ifdef __METAL_VERSION__
  if (sizeof(T) == sizeof(float))
    return static_cast<T>(metal::sin(static_cast<float>(x)));
  else
    return static_cast<T>(metal::sin(static_cast<double>(x)));
#else
  if constexpr (std::is_same_v<T, float>)
    return sinf(x);
  else if constexpr (std::is_same_v<T, double>)
    return std::sin(x);
#endif
}

template <typename T>
inline OPENRAND_DEVICE T cos(T x) {
#ifdef __METAL_VERSION__
  if (sizeof(T) == sizeof(float))
    return static_cast<T>(metal::cos(static_cast<float>(x)));
  else
    return static_cast<T>(metal::cos(static_cast<double>(x)));
#else
  if constexpr (std::is_same_v<T, float>)
    return cosf(x);
  else if constexpr (std::is_same_v<T, double>)
    return std::cos(x);
#endif
}

template <typename T>
inline OPENRAND_DEVICE T log(T x) {
#ifdef __METAL_VERSION__
  if (sizeof(T) == sizeof(float))
    return static_cast<T>(metal::log(static_cast<float>(x)));
  else
    return static_cast<T>(metal::log(static_cast<double>(x)));
#else
  if constexpr (std::is_same_v<T, float>)
    return logf(x);
  else if constexpr (std::is_same_v<T, double>)
    return std::log(x);
#endif
}

template <typename T>
inline OPENRAND_DEVICE T sqrt(T x) {
#ifdef __METAL_VERSION__
  if (sizeof(T) == sizeof(float))
    return static_cast<T>(metal::sqrt(static_cast<float>(x)));
  else
    return static_cast<T>(metal::sqrt(static_cast<double>(x)));
#else
  if constexpr (std::is_same_v<T, float>)
    return sqrtf(x);
  else if constexpr (std::is_same_v<T, double>)
    return std::sqrt(x);
#endif
}

template <typename T>
struct vec2 {
  T x, y;
};

template <typename T>
struct vec3 {
  T x, y, z;
};

template <typename T>
struct vec4 {
  T x, y, z, w;
};

// for GPU, better to be explicit about the type and size
using uint2 = vec2<uint32_t>;
using uint3 = vec3<uint32_t>;
using uint4 = vec4<uint32_t>;

using float2 = vec2<float>;
using float3 = vec3<float>;
using float4 = vec4<float>;

using double2 = vec2<double>;
using double3 = vec3<double>;
using double4 = vec4<double>;

// CRTP: helper struct to check if Derived has internal counter
// that enables O(1) state forwarding
#ifdef __METAL_VERSION__
// Simplified version for Metal (no std::void_t support)
template <typename T>
struct has_counter {
    static constexpr bool value = true; // Assume all RNGs have counter for Metal
};
#else
template <typename T, typename = std::void_t<>>
struct has_counter : std::false_type {};

template <typename T>
struct has_counter<T, std::void_t<decltype(std::declval<T>()._ctr)>>
    : std::true_type {};
#endif

}  // namespace openrand

#endif
