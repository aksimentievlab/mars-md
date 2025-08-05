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

#ifndef OPENRAND_PHILOX_H_
#define OPENRAND_PHILOX_H_

#include "base_state.h"

#ifdef __METAL_VERSION__
#include <metal_stdlib>
using namespace metal;
#else
#include <array>
#include <cstdint>
#include <iostream>
#include <limits>
#include <type_traits> // For std::is_same_v
#endif

// Constants are safe for both host and device
#define PHILOX_W0 0x9E3779B9
#define PHILOX_W1 0xBB67AE85
#define PHILOX_M0 0xD2511F53
#define PHILOX_M1 0xCD9E8D57

namespace openrand {

/**
 * @class Philox
 * @brief A Metal-safe Philox 4x32 random number generator.
 */
#ifdef __METAL_VERSION__
class Philox {
#else
class Philox : public BaseRNG<Philox> {
#endif

#ifdef __METAL_VERSION__
  // Metal-compatible version of u01
  template <typename Ftype, typename Utype>
  static inline OPENRAND_DEVICE Ftype u01(const Utype in) {
    constexpr Ftype factor = Ftype(1.) / (Ftype(~static_cast<Utype>(0)) + Ftype(1.));
    constexpr Ftype halffactor = Ftype(0.5) * factor;
    return static_cast<Ftype>(in) * factor + halffactor;
  }
#endif // __METAL_VERSION__

 public:
  /**
   * @brief Constructor for the Philox generator.
   */
  OPENRAND_HOST_DEVICE Philox(uint64_t seed, uint32_t ctr,
                              uint32_t global_seed = openrand::DEFAULT_GLOBAL_SEED,
                              uint32_t ctr1 = 0x12345)
      : seed_hi(static_cast<uint32_t>(seed >> 32)),
        seed_lo(static_cast<uint32_t>(seed & 0xFFFFFFFF)),
        ctr0(ctr),
        ctr1(ctr1),
        ctr2(global_seed),
        _ctr(0) {}

  /**
   * @brief Draws a 32-bit or 64-bit random unsigned integer.
   */
  template <typename T = uint32_t>
  OPENRAND_DEVICE T draw() {
    generate();

#ifdef __METAL_VERSION__
    // MSL path: Use sizeof to check the return type since std::is_same_v is not available.
    if (sizeof(T) == sizeof(uint32_t)) {
      return static_cast<T>(_out[0]);
    } else {
      // Assuming the only other option is uint64_t
      uint64_t res = (static_cast<uint64_t>(_out[0]) << 32) | static_cast<uint64_t>(_out[1]);
      return static_cast<T>(res);
    }
#else
    // Host C++ path: We can use modern C++ features for compile-time checks.
    static_assert(std::is_same_v<T, uint32_t> || std::is_same_v<T, uint64_t>,
                  "Philox::draw() only supports uint32_t and uint64_t");
                  
    if constexpr (std::is_same_v<T, uint32_t>) {
      return _out[0];
    } else {
      uint64_t res = (static_cast<uint64_t>(_out[0]) << 32) | static_cast<uint64_t>(_out[1]);
      return res;
    }
#endif
  }

  /**
   * @brief Draws four 32-bit random unsigned integers.
   */
  OPENRAND_DEVICE openrand::uint4 draw_int4() {
    generate();
    // Use the Metal-safe uint4 type from util.h
    openrand::uint4 result = {_out[0], _out[1], _out[2], _out[3]};
    return result;
  }

  /**
   * @brief Draws four random floats in [0, 1).
   */
  OPENRAND_DEVICE openrand::float4 draw_float4() {
    generate();
#ifdef __METAL_VERSION__
    openrand::float4 result = {
        u01<float, uint32_t>(_out[0]), u01<float, uint32_t>(_out[1]),
        u01<float, uint32_t>(_out[2]), u01<float, uint32_t>(_out[3])};
#else
    openrand::float4 result = {
        this->u01<float, uint32_t>(_out[0]), this->u01<float, uint32_t>(_out[1]),
        this->u01<float, uint32_t>(_out[2]), this->u01<float, uint32_t>(_out[3])};
#endif
    return result;
  }
  
// Public member for O(1) state forwarding (used by BaseRNG)
// TODO: Implement or provide alternative for Metal
uint32_t _ctr;

 private:
  OPENRAND_DEVICE void generate() {
    // This logic is safe for both MSL and C++
    uint32_t key[2] = {seed_hi, seed_lo};
    _out[0] = ctr0;
    _out[1] = ctr1;
    _out[2] = ctr2;
    _out[3] = _ctr;

    // Philox uses 10 rounds
    for (int r = 0; r < 10; r++) {
      round(key, _out);
      if (r < 9) { // Avoid key update on the last round
          key[0] += PHILOX_W0;
          key[1] += PHILOX_W1;
      }
    }
    _ctr++;
  }

  OPENRAND_DEVICE void mulhilo(uint32_t L, uint32_t R, thread uint32_t &hip) {
    const uint64_t product = static_cast<uint64_t>(L) * static_cast<uint64_t>(R);
    hip = static_cast<uint32_t>(product >> 32);
    // Return low part
    // On MSL we would need to pass hip as a pointer/reference.
    // Let's modify the signature to be compatible.
    _out[0] = static_cast<uint32_t>(product);
  }

  inline OPENRAND_DEVICE void round(const thread uint32_t (&key)[2], thread uint32_t (&ctr)[4]) {
    uint32_t hi0, hi1;
    uint32_t lo0, lo1;

    // Manually handle mulhilo logic since we can't return a value and modify a pointer simultaneously
    // in a clean way across platforms.
    uint64_t p0 = static_cast<uint64_t>(PHILOX_M0) * static_cast<uint64_t>(ctr[0]);
    hi0 = p0 >> 32;
    lo0 = p0;

    uint64_t p1 = static_cast<uint64_t>(PHILOX_M1) * static_cast<uint64_t>(ctr[2]);
    hi1 = p1 >> 32;
    lo1 = p1;

    ctr[0] = hi1 ^ ctr[1] ^ key[0];
    ctr[1] = lo1;
    ctr[2] = hi0 ^ ctr[3] ^ key[1];
    ctr[3] = lo0;
  }

  // User-provided seeds and counters, constant for the object's lifetime.
  const uint32_t seed_hi, seed_lo;
  const uint32_t ctr0, ctr1, ctr2;
  
  // Internal buffer for generated numbers
  uint32_t _out[4];

};  // class Philox

}  // namespace openrand

#endif  // OPENRAND_PHILOX_H_


