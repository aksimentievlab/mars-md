#pragma once
/**
 * @file Header.h
 * @brief Common macros for all backends.
 * @version 0.1
 * @date 2025-08-22
 */

#ifdef __CUDACC__
#define HOST __host__
#define DEVICE __device__
#else
#define HOST
#define DEVICE
#endif

#ifdef __METAL_VERSION__
#include <metal_stdlib>
#endif

#ifdef USE_SYCL
#include <sycl/sycl.hpp>
#endif

#ifdef __CUDA_ARCH__
#include <cuda.h> // Or <cuda.h> depending on CUDAManager's needs
#include <cuda_runtime.h>
#endif

#if !defined(__CUDA_ARCH__) && !defined(__SYCL_DEVICE_ONLY__) && !defined(__METAL_VERSION__)
#define HOST_GUARD
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#endif

// Suppress narrowing conversion warnings
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wc++11-narrowing"
#endif

#ifdef __METAL_VERSION__
#include <metal_stdlib>
using namespace metal;
template<typename T>
using device_ptr = device T*;
template<typename T>
using constant_ptr = constant T*;
template<typename T>
using thread_ptr = thread T*;
template<typename T>
using threadgroup_ptr = threadgroup T*;
#ifndef DEVICE_PTR
#define DEVICE_PTR(T) device T*
#endif
#ifndef CONSTANT_PTR
#define CONSTANT_PTR(T) constant T*
#endif
#ifndef THREAD_PTR
#define THREAD_PTR(T) thread T*
#endif
#ifndef THREADGROUP_PTR
#define THREADGROUP_PTR(T) threadgroup T*
#endif
#else
// Address space pointer macros for non-Metal backends
#ifndef DEVICE_PTR
#define DEVICE_PTR(T) T*
#endif
#ifndef CONSTANT_PTR
#define CONSTANT_PTR(T) const T*
#endif
#ifndef THREAD_PTR
#define THREAD_PTR(T) T*
#endif
#ifndef THREADGROUP_PTR
#define THREADGROUP_PTR(T) T*
#endif
#endif
using idx_t = size_t;
using device_id_t = size_t;
