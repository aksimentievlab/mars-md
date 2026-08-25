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
#ifndef OPENRAND_BASE_STATE_H_
#define OPENRAND_BASE_STATE_H_

#include "util.h"

#ifdef __METAL_VERSION__
#include <metal_stdlib>
using namespace metal;
#else
#include <cstdint>
#include <limits>
#include <type_traits>
#endif

namespace openrand {

#ifndef __METAL_VERSION__
// Host-side (C++) type traits
template<typename T>
constexpr bool is_integral_v = std::is_integral_v<T>;
template<typename T>
constexpr bool is_floating_point_v = std::is_floating_point_v<T>;
#else
// Metal-safe type traits implementation
template<typename T> struct is_integral { const bool value = false; };
template<> struct is_integral<int> { const bool value = true; };
template<> struct is_integral<uint> { const bool value = true; };
#endif


/**
 * @brief Base class for random number generators.
 * (CRTP pattern)
 */
template <typename RNG>
class BaseRNG {
 public:
  using result_type = uint32_t;

  OPENRAND_HOST_DEVICE static constexpr result_type min() { return 0u; }
  OPENRAND_HOST_DEVICE static constexpr result_type max() { return ~((result_type)0); }

  /**
   * @brief Generates a 32 bit unsigned integer (C++ engine interface).
   */
  OPENRAND_DEVICE result_type operator()() {
    return gen().template draw<uint32_t>();
  }

  /**
   * @brief Generates a random number from a uniform distribution between 0 and 1.
   */
  template <typename T = float>
  OPENRAND_DEVICE T rand() {
#ifdef __METAL_VERSION__
    // MSL does not have if constexpr, use standard if
    if (sizeof(T) <= 4) {
      const uint32_t x = gen().template draw<uint32_t>();
      if (is_integral_v<T>) return static_cast<T>(x);
      else return u01<float, uint32_t>(x);
    } else {
      const uint64_t x = gen().template draw<uint64_t>();
      if (is_integral_v<T>) return static_cast<T>(x);
      else return u01<float, uint64_t>(x);
    }
#else
    if constexpr (sizeof(T) <= 4) {
      const uint32_t x = gen().template draw<uint32_t>();
      if constexpr (is_integral_v<T>) return static_cast<T>(x);
      else return u01<float, uint32_t>(x);
    } else {
      const uint64_t x = gen().template draw<uint64_t>();
      if constexpr (is_integral_v<T>) return static_cast<T>(x);
      else return u01<double, uint64_t>(x);
    }
#endif
  }

  /**
   * @brief Generates a number from a uniform distribution between a and b.
   */
  template <typename T = float>
  OPENRAND_DEVICE T uniform(const T low, const T high) {
#ifndef __METAL_VERSION__
    static_assert(!(std::is_integral_v<T> && sizeof(T) > sizeof(int32_t)),
                  "64 bit int not yet supported");
#endif
    T r = high - low;
#ifdef __METAL_VERSION__
    if (is_floating_point_v<T>) {
        return low + r * rand<T>();
    } else if (is_integral_v<T>) {
        return low + range<true, T>(r);
    }
#else
    if constexpr (is_floating_point_v<T>) {
      return low + r * rand<T>();
    } else if constexpr (is_integral_v<T>) {
      return low + range<true, T>(r);
    }
#endif
    return T(0); // Should be unreachable
  }

  /**
   * @brief Fills an array with random numbers from a uniform distribution [0, 1)
   */
  template <typename T = float>
  OPENRAND_DEVICE void fill_random(thread T *array, const int N) {
    for (int i = 0; i < N; i++) array[i] = rand<T>();
  }

  /**
   * @brief Generates a random number from a normal distribution (Box-Muller).
   */
  template <typename T = float>
  OPENRAND_DEVICE T randn() {
#ifndef __METAL_VERSION__
    static_assert(std::is_floating_point_v<T>);
#endif
    constexpr T M_PI2 = 2 * M_PI;
    T u = rand<T>();
    T v = rand<T>();
    T r = openrand::sqrt(T(-2.0) * openrand::log(u));
    T theta = v * M_PI2;
    return r * openrand::cos(theta);
  }

  /**
   * @brief More efficient version of randn, returns two values at once.
   */
  template <typename T = float>
  OPENRAND_DEVICE vec2<T> randn2() {
#ifndef __METAL_VERSION__
    static_assert(std::is_floating_point_v<T>);
#endif
    constexpr T M_PI2 = 2 * M_PI;
    T u = rand<T>();
    T v = rand<T>();
    T r = sqrt(T(-2.0) * log(u));
    T theta = v * M_PI2;
    vec2<T> result = {r * cos(theta), r * sin(theta)};
    return result;
  }

  /**
   * @brief Generates a random number from a normal distribution with mean and std.
   */
  template <typename T = float>
  OPENRAND_DEVICE T randn(const T mean, const T std_dev) {
    return mean + randn<T>() * std_dev;
  }

  /**
   * @brief Generates a random integer of certain range [0..N)
   */
  template <bool biased = true, typename T = int>
  OPENRAND_DEVICE T range(const T N) {
    uint32_t x = gen().template draw<uint32_t>();
    uint64_t res = static_cast<uint64_t>(x) * static_cast<uint64_t>(N);

#ifdef __METAL_VERSION__
    if (biased) {
        return static_cast<T>(res >> 32);
    }
#else
    if constexpr (biased) {
      return static_cast<T>(res >> 32);
    }
#endif
    else {
      uint32_t leftover = static_cast<uint32_t>(res);
      if (leftover < N) {
        uint32_t threshold = -N % N;
        while (leftover < threshold) {
          x = gen().template draw<uint32_t>();
          res = static_cast<uint64_t>(x) * static_cast<uint64_t>(N);
          leftover = static_cast<uint32_t>(res);
        }
      }
      return static_cast<T>(res >> 32); // Return the high part
    }
  }

  /**
   * @brief Generates a random number from a gamma distribution.
   */
  template <typename T = float>
  OPENRAND_DEVICE T gamma(T alpha, T b) {
    T d = alpha - T(1.0/3.0);
    T c = T(1.0) / sqrt(9.0f * d);
    T v, x;
    while (true) {
      do {
        x = randn<T>();
        v = T(1.0) + c * x;
      } while (v <= T(0.));
      v = v * v * v;
      T u = rand<T>();

      const T x2 = x * x;
      if (u < 1.0f - 0.0331f * x2 * x2) return (d * v * b);
      if (log(u) < 0.5f * x2 + d * (1.0f - v + log(v))) return (d * v * b);
    }
  }

  /**
   * @brief Returns a new generator with the internal state forwarded by n steps (O(1)).
   */
#ifdef __METAL_VERSION__
  // Simplified version for Metal without enable_if_t
  template <typename T = RNG>
  RNG forward_state(int n) const {
      RNG rng = *static_cast<thread const RNG *>(this);
      rng._ctr += n;
      return rng;
  }
#else
  template <typename T = RNG>
  typename std::enable_if_t<has_counter<T>::value, RNG>
  forward_state(int n) const {
    RNG rng = *static_cast<const RNG *>(this);  // copy
    rng._ctr += n;
    return rng;
  }
#endif

 protected:
  /**
   * @brief Converts a random integer to a floating point number in [0., 1.).
   */
  template <typename Ftype, typename Utype>
  inline OPENRAND_HOST_DEVICE Ftype u01(const Utype in) const {
    constexpr Ftype factor = Ftype(1.) / (Ftype(~static_cast<Utype>(0)) + Ftype(1.));
    constexpr Ftype halffactor = Ftype(0.5) * factor;
    return static_cast<Ftype>(in) * factor + halffactor;
  }

 private:
#ifdef __METAL_VERSION__
  OPENRAND_DEVICE thread RNG &gen() {
    return *static_cast<thread RNG *>(this);
  }
  OPENRAND_DEVICE thread const RNG &gen() const {
    return *static_cast<thread const RNG *>(this);
  }
#else
  OPENRAND_DEVICE RNG &gen() {
    return *static_cast<RNG *>(this);
  }
  OPENRAND_DEVICE const RNG &gen() const {
    return *static_cast<const RNG *>(this);
  }
#endif

};

}  // namespace openrand

#endif  // OPENRAND_BASE_STATE_H_

