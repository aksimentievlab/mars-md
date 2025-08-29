#pragma once
/**
 * @file Header.h
 * @brief Common macros for all backends.
 * @version 0.1
 * @date 2025-08-22
 */
#include <cstddef>
#ifdef __CUDACC__
#define HOST __host__
#define DEVICE __device__
#else
#define HOST
#define DEVICE
#endif

#if !defined(__CUDA_ARCH__) && !defined(__SYCL_DEVICE_ONLY__) && !defined(__METAL_VERSION__)
#define HOST_GUARD
#endif

// Suppress narrowing conversion warnings
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wc++11-narrowing"
#endif

using idx_t = size_t;
using device_id_t = size_t;
