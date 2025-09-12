#include "Header.h"

#ifdef HOST_GUARD
#include <experimental/simd>
namespace sx = std::experimental;
using simd_float2 = sx::simd<float, sx::simd_abi::fixed_size<2>>;
using simd_int2 = sx::simd<int, sx::simd_abi::fixed_size<2>>;
#elif defined(__METAL_VERSION__)
#include <metal_stdlib>
using simd_float2 = simd::float2;
using simd_int2 = simd::int2;
#endif

namespace ARBD {

// SIMD-friendly types for different backends
#ifdef __CUDACC__
// CUDA warp-aligned types
using simd_int = int;
using simd_float = float;
using simd_double = double;
using simd_int2 = int2;
using simd_float2 = float2;
#elif defined(__SYCL_DEVICE_ONLY__)
// SYCL SIMD types
using simd_int = int;
using simd_float = float;
using simd_double = double;
using simd_int2 = sycl::vec<int, 2>;
using simd_float2 = sycl::vec<float, 2>;
#elif defined(__METAL_VERSION__)
// Metal SIMD types
using simd_int = int;
using simd_float = float;
using simd_double = double;
using simd_int2 = int2;
using simd_float2 = float2;
#else
// CPU SIMD types (could use std::simd in C++20)
using simd_int = int;
using simd_float = float;
using simd_double = double;
using simd_int2 = simd_int2;
using simd_float2 = simd_float2;
#endif

// Backend-specific alignment requirements
#ifdef __CUDACC__
static constexpr size_t MEMORY_ALIGNMENT = 128; // CUDA warp size alignment
#elif defined(__SYCL_DEVICE_ONLY__)
static constexpr size_t MEMORY_ALIGNMENT = 64; // SYCL typical alignment
#elif defined(__METAL_VERSION__)
static constexpr size_t MEMORY_ALIGNMENT = 64; // Metal typical alignment
#else
static constexpr size_t MEMORY_ALIGNMENT = 32; // CPU cache line alignment
#endif

// Backend-specific optimal sizes
#ifdef __CUDACC__
static constexpr size_t OPTIMAL_BLOCK_SIZE = 256; // CUDA optimal block size
static constexpr size_t OPTIMAL_WARP_SIZE = 32;	  // CUDA warp size
#elif defined(__SYCL_DEVICE_ONLY__)
static constexpr size_t OPTIMAL_BLOCK_SIZE = 256; // SYCL typical work group size
static constexpr size_t OPTIMAL_WARP_SIZE = 32;	  // SYCL subgroup size
#elif defined(__METAL_VERSION__)
static constexpr size_t OPTIMAL_BLOCK_SIZE = 256; // Metal threadgroup size
static constexpr size_t OPTIMAL_WARP_SIZE = 32;	  // Metal SIMD group size
#else
static constexpr size_t OPTIMAL_BLOCK_SIZE = 64; // CPU cache-friendly size
static constexpr size_t OPTIMAL_WARP_SIZE = 8;	 // CPU SIMD width
#endif

} // namespace ARBD
