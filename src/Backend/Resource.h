#pragma once
#include "ARBDLogger.h"
#include <type_traits>

// Define HOST and DEVICE macros
#ifdef __CUDACC__
#define HOST __host__
#define DEVICE __device__
#else
#define HOST
#define DEVICE
#endif

#ifdef USE_CUDA
#include "Backend/CUDA/CUDAManager.h"
#endif

#ifdef USE_SYCL
#include "Backend/SYCL/SYCLManager.h"
#endif

#ifdef USE_METAL
#include "Backend/METAL/METALManager.h"
#endif

/**
 * @brief Get current device ID for device computing backends
 */
inline size_t get_device_id() {
#ifdef USE_CUDA
  if (cudaGetDevice(nullptr) == cudaSuccess) {
    int device;
    cudaGetDevice(&device);
    return static_cast<size_t>(device);
  }
#endif

#ifdef USE_SYCL
  try {
    return static_cast<size_t>(
        ARBD::SYCL::SYCLManager::get_current_device().id());
  } catch (...) {
    return 0;
  }
#endif

#ifdef USE_METAL
  try {
    return static_cast<size_t>(
        ARBD::METAL::METALManager::get_current_device().id());
  } catch (...) {
    return 0;
  }
#endif

  return 0;
}

/**
 * @brief Resource representation for heterogeneous computing
 * Supports CUDA, SYCL, and MPI resource types
 */
namespace ARBD {

enum class ResourceType {CPU, CUDA, SYCL, METAL };

/**
 * @brief Memory transfer policies
 */
enum class TransferType {
  HOST_TO_DEVICE,
  DEVICE_TO_HOST,
  DEVICE_TO_DEVICE,
  HOST_TO_HOST
};

/**
 * @brief Backend capability traits for compile-time feature detection
 */
template <typename Backend> struct BackendTraits {
  static constexpr bool supports_device_memory = false;
  static constexpr bool supports_async_execution = false;
  static constexpr bool supports_peer_access = false;
  static constexpr bool requires_explicit_sync = false;

  using context_type = void;
  using event_type = void;
  using stream_type = void;
};

/**
 * @brief CUDA Backend Traits
 */
struct CUDABackend {
  static constexpr const char *name = "CUDA";
  static constexpr ResourceType resource_type = ResourceType::CUDA;
};

template <> struct BackendTraits<CUDABackend> {
  static constexpr bool supports_device_memory = true;
  static constexpr bool supports_async_execution = true;
  static constexpr bool supports_peer_access = true;
  static constexpr bool requires_explicit_sync = true;

#ifdef USE_CUDA
  using context_type = int; // CUDA device ID
  using event_type = cudaEvent_t;
  using stream_type = cudaStream_t;
#else
  using context_type = void;
  using event_type = void;
  using stream_type = void;
#endif
};

/**
 * @brief SYCL Backend Traits
 */
struct SYCLBackend {
  static constexpr const char *name = "SYCL";
  static constexpr ResourceType resource_type = ResourceType::SYCL;
};

template <> struct BackendTraits<SYCLBackend> {
  static constexpr bool supports_device_memory = true;
  static constexpr bool supports_async_execution = true;
  static constexpr bool supports_peer_access = false;
  static constexpr bool requires_explicit_sync = false;

#ifdef USE_SYCL
  using context_type = sycl::queue*; // void*
  using event_type = sycl::event*;   // void*
  using stream_type = sycl::queue*;  // void*
#else
  using context_type = void;
  using event_type = void;
  using stream_type = void;
#endif
};

/**
 * @brief METAL Backend Traits
 */
struct METALBackend {
  static constexpr const char *name = "METAL";
  static constexpr ResourceType resource_type = ResourceType::METAL;
};

template <> struct BackendTraits<METALBackend> {
  static constexpr bool supports_device_memory = true;
  static constexpr bool supports_async_execution = true;
  static constexpr bool supports_peer_access = false;
  static constexpr bool requires_explicit_sync = false;

  using context_type = void; // METAL doesn't expose contexts directly
  using event_type = void;   // METAL command buffer events
  using stream_type = void;  // METAL command queues
};

/**
 * @brief Concept to check if a type is a valid backend
 */
template <typename T>
concept ValidBackend = requires {
  typename BackendTraits<T>;
  { T::name } -> std::convertible_to<const char *>;
  { T::resource_type } -> std::same_as<ResourceType>;
};

/**
 * @brief Resource representation for device computing environments
 *
 * The Resource class provides a unified interface for representing and managing
 * computational resources for device computing (CUDA, SYCL, METAL).
 * For distributed computing (MPI), use MPIResource from MPIBackend.h instead.
 *
 * @details This class manages different compute devices on a single machine:
 * - CUDA GPU devices for NVIDIA GPU computing
 * - SYCL devices for cross-platform parallel computing
 * - METAL devices for Apple GPU computing
 *
 * @example Basic Usage:
 * ```cpp
 * // Create a CUDA resource for device 0
 * ARBD::Resource cuda_res(ARBD::Resource::CUDA, 0);
 *
 * // Create a SYCL resource for device 1
 * ARBD::Resource sycl_res(ARBD::Resource::SYCL, 1);
 *
 * // Check if resource is local to current execution context
 * if (cuda_res.is_local()) {
 *     // Perform local operations
 * }
 * ```
 *
 * @see MPIBackend.h for distributed computing resources
 * @see ResourceType for available device resource types
 * @see is_local() for locality checking
 * @see getTypeString() for human-readable type names
 */

struct Resource {
#ifdef USE_CUDA
  static constexpr ResourceType DEFAULT_DEVICE = ResourceType::CUDA;
#elif defined(USE_SYCL)
  static constexpr ResourceType DEFAULT_DEVICE = ResourceType::SYCL;
#elif defined(USE_METAL)
  static constexpr ResourceType DEFAULT_DEVICE = ResourceType::METAL;
#else 
  LOGERROR("Resource::Resource(): No device backend defined, using HOST only");
#endif

  ResourceType type;
  size_t id;
  Resource *parent;

  HOST DEVICE Resource() : type(DEFAULT_DEVICE), id(0), parent(nullptr) {}
  HOST DEVICE Resource(ResourceType t, size_t i)
      : type(t), id(i), parent(nullptr) {}
  HOST DEVICE Resource(ResourceType t, size_t i, Resource *p)
      : type(t), id(i), parent(p) {}

  HOST DEVICE constexpr const char *getTypeString() const {
    switch (type) {
    case ResourceType::CUDA:
      return "CUDA";
    case ResourceType::SYCL:
      return "SYCL";
    case ResourceType::METAL:
      return "METAL";
    default:
      return "Unknown";
    }
  }

  HOST DEVICE bool is_local() const {
#if defined(__CUDA_ARCH__) || defined(__SYCL_DEVICE_ONLY__)
    // We are executing on a device
    if (type == ResourceType::CPU) {
      return false;
    }
    // Check if the resource's device type and ID match the current device
    // This part requires device-specific ways to get current device ID,
    // which can be complex. Assuming for now that if it's not CPU,
    // and we are on a device, we are on the right one if the code is launched correctly.
    // A more robust implementation might be needed here.
    return true; // Simplified for device code
#else
    // We are executing on the host
    if (type == ResourceType::CPU) {
      return true;
    }

    bool ret = false;
#ifdef USE_CUDA
    if (type == ResourceType::CUDA) {
      int current_device;
      if (cudaGetDevice(&current_device) == cudaSuccess) {
        ret = (current_device == static_cast<int>(id));
      }
    }
#endif
#ifdef USE_SYCL
    if (type == ResourceType::SYCL) {
      try {
        auto &current_device = ARBD::SYCL::SYCLManager::get_current_device();
        ret = (current_device.id() == id);
      } catch (...) {
        ret = false;
      }
    }
#endif
#ifdef USE_METAL
    if (type == ResourceType::METAL) {
          try {
      auto &current_device = ARBD::METAL::METALManager::get_current_device();
      ret = (current_device.id() == id);
    } catch (...) {
      ret = false;
    }
    }
#endif
    return ret;
#endif
  }

  static Resource Local() {
#ifdef USE_CUDA
    int device;
    if (cudaGetDevice(&device) == cudaSuccess) {
      return Resource{ResourceType::CUDA, static_cast<size_t>(device)};
    }
#endif
#ifdef USE_SYCL
    try {
      auto &current_device = ARBD::SYCL::SYCLManager::get_current_device();
      return Resource{ResourceType::SYCL, static_cast<size_t>(current_device.id())};
    } catch (...) {
    }
#endif
#ifdef USE_METAL
    try {
      auto &current_device = ARBD::METAL::METALManager::get_current_device();
      return Resource{ResourceType::METAL,
                      static_cast<size_t>(current_device.id())};
    } catch (...) {
    }
#endif
    // Default to HOST if no device context is active.
    return Resource{ResourceType::CPU, 0};
  }

  HOST DEVICE bool operator==(const Resource &other) const {
    return type == other.type && id == other.id;
  }

  HOST std::string toString() const {
    return std::string(getTypeString()) + "[" + std::to_string(id) + "]";
  }

  /**
   * @brief Check if the resource supports asynchronous operations
   */
  HOST DEVICE bool supports_async() const {
    switch (type) {
    case ResourceType::CUDA:
    case ResourceType::SYCL:
      return true;

    default:
      return false;
    }
  }

  /**
   * @brief Get the memory space type for this resource
   */
  HOST DEVICE constexpr const char *getMemorySpace() const {
    switch (type) {
    case ResourceType::CPU:
      return "host";
    case ResourceType::CUDA:
      return "device";
    case ResourceType::SYCL:
      return "device";

    case ResourceType::METAL:
      return "device";
    default:
      return "host";
    }
  }

  /**
   * @brief Check if this resource represents a device (GPU)
   */
  HOST DEVICE bool is_device() const {
    return type == ResourceType::CUDA || type == ResourceType::SYCL ||
           type == ResourceType::METAL;
  }

  /**
   * @brief Check if this resource represents a host (CPU)
   * @note Always returns false since Resource only handles device computing.
   *       For distributed computing, use MPIResource from MPIBackend.h
   */
  HOST DEVICE bool is_host() const { return type == ResourceType::CPU; }
};

} // namespace ARBD
