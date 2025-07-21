#pragma once

#include "ARBDException.h"
#include "ARBDLogger.h"
#include "Resource.h"
#include <cstring>
#include <future>
#include <typeinfo>
#include <span>
#include <functional>
#include <tuple>
#include "Kernels.h"
/**
 * @file Proxy.h
 * @brief Backend-agnostic proxy system for remote object method calls
 * 
 * This file provides the Proxy template class and supporting infrastructure for
 * making synchronous and asynchronous method calls on objects located on different
 * compute backends (CUDA, SYCL, Metal, etc.). The proxy system handles serialization,
 * backend-specific dispatch, and result retrieval transparently.
 */

namespace ARBD {

// Helper functions for transfer type
inline TransferType get_transfer_type(const Resource& src, const Resource& dest) {
    if (src.is_host() && dest.is_device()) return TransferType::HOST_TO_DEVICE;
    if (src.is_device() && dest.is_host()) return TransferType::DEVICE_TO_HOST;
    if (src.is_device() && dest.is_device()) return TransferType::DEVICE_TO_DEVICE;
    return TransferType::HOST_TO_HOST;
}

inline const char* to_string(TransferType type) {
    switch (type) {
        case TransferType::HOST_TO_DEVICE: return "HOST_TO_DEVICE";
        case TransferType::DEVICE_TO_HOST: return "DEVICE_TO_HOST";
        case TransferType::DEVICE_TO_DEVICE: return "DEVICE_TO_DEVICE";
        case TransferType::HOST_TO_HOST: return "HOST_TO_HOST";
    }
    return "UNKNOWN_TRANSFER";
}

/**
 * @brief Concept to check if a type has a 'send_children' method that accepts a
 * Resource parameter.
 */
template <typename T>
concept has_send_children = requires(T t, Resource r) { t.send_children(r); };

/**
 * @brief Concept to check if a type has a 'no_send' member.
 */
template <typename T>
concept has_no_send = requires(T t) { t.no_send; };

// Helper alias for C++14 compatibility (still needed for Metadata_t)
template <typename...> using void_t = void;

// Used by Proxy class
template <typename T, typename = void> struct Metadata_t {
  Metadata_t(const T &obj){};
  Metadata_t(const Metadata_t<T> &other){};
};

template <typename T>
struct Metadata_t<T, void_t<typename T::Metadata>> : T::Metadata {
  Metadata_t(const T &obj) : T::Metadata(obj){};
  Metadata_t(const Metadata_t<T> &other) : T::Metadata(other){};
};

// Forward declarations for backend-specific implementations
namespace ProxyImpl {
// CUDA-specific implementations (defined in Proxy.cu when USE_CUDA is enabled)
#ifdef USE_CUDA
void *cuda_call_sync(void *addr, void *func_ptr, void *args, size_t args_size,
                     const Resource &location, size_t result_size);
std::future<void *> cuda_call_async(void *addr, void *func_ptr, void *args,
                                    size_t args_size, const Resource &location,
                                    size_t result_size);
#else
inline void *cuda_call_sync(void *addr, void *func_ptr, void *args, size_t args_size,
                            const Resource &location, size_t result_size) {
    ARBD::throw_not_implemented("CUDA support not enabled");
}
inline std::future<void *> cuda_call_async(void *addr, void *func_ptr, void *args,
                                           size_t args_size, const Resource &location,
                                           size_t result_size) {
    ARBD::throw_not_implemented("CUDA support not enabled");
}
#endif

// SYCL-specific implementations (defined in Proxy_sycl.cpp when USE_SYCL is enabled)
#ifdef USE_SYCL
void *sycl_call_sync(void *addr, void *func_ptr, void *args, size_t args_size,
                     const Resource &location, size_t result_size);
std::future<void *> sycl_call_async(void *addr, void *func_ptr, void *args,
                                    size_t args_size, const Resource &location,
                                    size_t result_size);
#else
inline void *sycl_call_sync(void *addr, void *func_ptr, void *args, size_t args_size,
                            const Resource &location, size_t result_size) {
    ARBD::throw_not_implemented("SYCL support not enabled");
}
inline std::future<void *> sycl_call_async(void *addr, void *func_ptr, void *args,
                                           size_t args_size, const Resource &location,
                                           size_t result_size) {
    ARBD::throw_not_implemented("SYCL support not enabled");
}
#endif

// METAL-specific implementations (defined in Proxy.mm when USE_METAL is enabled)
#if defined(USE_METAL) && defined(__OBJC__)
void *metal_call_sync(void *addr, void *func_ptr, void *args, size_t args_size,
                      const Resource &location, size_t result_size);
std::future<void *> metal_call_async(void *addr, void *func_ptr, void *args,
                                     size_t args_size, const Resource &location,
                                     size_t result_size);
#else
inline void *metal_call_sync(void *addr, void *func_ptr, void *args, size_t args_size,
                             const Resource &location, size_t result_size) {
    ARBD::throw_not_implemented("METAL support not enabled");
}
inline std::future<void *> metal_call_async(void *addr, void *func_ptr, void *args,
                                            size_t args_size, const Resource &location,
                                            size_t result_size) {
    ARBD::throw_not_implemented("METAL support not enabled");
}
#endif

// General send implementations
template <typename T>
void *send_ignoring_children(const Resource &location, T &obj, T *dest);

template <typename T>
void *construct_remote(const Resource &location, void *args, size_t args_size);
} // namespace ProxyImpl

// ============================================================================
// Enhanced Kernel and Remote Call Infrastructure
// ============================================================================

/**
 * @brief Result wrapper for remote method calls
 */
template<typename T>
struct RemoteResult {
    T result;
    std::future<void> completion_future;
    
    RemoteResult(T&& res, std::future<void>&& future) 
        : result(std::forward<T>(res)), completion_future(std::move(future)) {}
    
    void wait() { completion_future.wait(); }
    bool is_ready() const { 
        return completion_future.wait_for(std::chrono::seconds(0)) == std::future_status::ready; 
    }
    
    T get() {
        wait();
        return std::move(result);
    }
};

// Specialization for void type
template<>
struct RemoteResult<void> {
    std::future<void> completion_future;
    
    RemoteResult(std::future<void>&& future) 
        : completion_future(std::move(future)) {}
    
    void wait() { completion_future.wait(); }
    bool is_ready() const { 
        return completion_future.wait_for(std::chrono::seconds(0)) == std::future_status::ready; 
    }
    
    void get() {
        wait();
    }
};

/**
 * @brief Enhanced kernel dispatcher for remote calls
 */
class RemoteKernelDispatcher {
public:
    template<typename KernelFunc, typename... Args>
    static std::future<void> dispatch_cuda(const Resource& resource,
                                          size_t num_elements,
                                          const Kernels::KernelConfig& config,
                                          KernelFunc&& kernel,
                                          Args&&... args) {
        return std::async(std::launch::async, [=, kernel = std::forward<KernelFunc>(kernel)]() mutable {
            for (size_t i = 0; i < num_elements; ++i) {
                kernel(i, args...);
            }
        });
    }

    template<typename KernelFunc, typename... Args>
    static std::future<void> dispatch_sycl(const Resource& resource,
                                          size_t num_elements,
                                          const Kernels::KernelConfig& config,
                                          KernelFunc&& kernel,
                                          Args&&... args) {
        return std::async(std::launch::async, [=, kernel = std::forward<KernelFunc>(kernel)]() mutable {
            for (size_t i = 0; i < num_elements; ++i) {
                kernel(i, args...);
            }
        });
    }

    template<typename KernelFunc, typename... Args>
    static std::future<void> dispatch_metal(const Resource& resource,
                                           size_t num_elements,
                                           const Kernels::KernelConfig& config,
                                           KernelFunc&& kernel,
                                           Args&&... args) {
        return std::async(std::launch::async, [=, kernel = std::forward<KernelFunc>(kernel)]() mutable {
            for (size_t i = 0; i < num_elements; ++i) {
                kernel(i, args...);
            }
        });
    }

    template<typename KernelFunc, typename... Args>
    static std::future<void> dispatch_cpu(size_t num_elements,
                                         const Kernels::KernelConfig& config,
                                         KernelFunc&& kernel,
                                         Args&&... args) {
        if (config.async) {
            return std::async(std::launch::async, [=, kernel = std::forward<KernelFunc>(kernel)]() mutable {
                for (size_t i = 0; i < num_elements; ++i) {
                    kernel(i, args...);
                }
            });
        } else {
            for (size_t i = 0; i < num_elements; ++i) {
                kernel(i, args...);
            }
            std::promise<void> promise;
            promise.set_value();
            return promise.get_future();
        }
    }
};

// Helper to decay and remove pointer/reference to get base type
template <typename T>
using base_type_t = std::remove_cv_t<std::remove_pointer_t<std::decay_t<T>>>;

// Helper to check if a type is a vector
template <typename T>
struct is_vector : std::false_type {};

template <typename T, typename A>
struct is_vector<std::vector<T, A>> : std::true_type {};

template<typename T>
static constexpr bool is_vector_v = is_vector<std::decay_t<T>>::value;

// Argument wrapper for remote_kernel_call
template<typename T>
auto get_kernel_arg(T&& arg) {
    if constexpr (is_vector_v<T>) {
        return arg.data();
    } else {
        return arg;
    }
}

/**
 * @brief Enhanced kernel launch function that works with proxy objects
 */
template<typename KernelFunc, typename... Args>
RemoteResult<void> remote_kernel_call(const Resource& resource,
                                    size_t num_elements,
                                    KernelFunc&& kernel,
                                    Args&&... args) {
    Kernels::KernelConfig config; // Default config for now
    std::future<void> future;

    switch (resource.type) {
        case ResourceType::CPU:
            future = RemoteKernelDispatcher::dispatch_cpu(num_elements, config,
                kernel, get_kernel_arg(std::forward<Args>(args))...
            );
            break;
        case ResourceType::CUDA:
            #ifdef USE_CUDA
                future = RemoteKernelDispatcher::dispatch_cuda(resource, num_elements, config, kernel, get_kernel_arg(std::forward<Args>(args))...);
            #else
                throw_not_implemented("CUDA support not enabled");
            #endif
            break;
        case ResourceType::SYCL:
            #ifdef USE_SYCL
                future = RemoteKernelDispatcher::dispatch_sycl(resource, num_elements, config, kernel, get_kernel_arg(std::forward<Args>(args))...);
            #else
                throw_not_implemented("SYCL support not enabled");
            #endif
            break;
        case ResourceType::METAL:
            #if defined(USE_METAL) && defined(__OBJC__)
                future = RemoteKernelDispatcher::dispatch_metal(resource, num_elements, config, kernel, get_kernel_arg(std::forward<Args>(args))...);
            #else
                throw_not_implemented("METAL support not enabled");
            #endif
            break;
    }

    return RemoteResult<void>(std::move(future));
}

/**
 * @brief Kernel chain for sequential remote execution
 */
class RemoteKernelChain {
private:
    Resource resource_;
    std::vector<std::future<void>> futures_;
    
public:
    explicit RemoteKernelChain(const Resource& resource) : resource_(resource) {}
    
    template<typename KernelFunc, typename... Objects>
    RemoteKernelChain& then(size_t num_elements, KernelFunc&& kernel, Objects&... objects) {
        // Wait for previous operations to complete
        for (auto& future : futures_) {
            future.wait();
        }
        futures_.clear();
        
        auto result = remote_kernel_call(resource_, num_elements,
                                       std::forward<KernelFunc>(kernel), objects...);
        futures_.push_back(std::move(result.completion_future));
        
        return *this;
    }
    
    void wait() {
        for (auto& future : futures_) {
            future.wait();
        }
    }
    
    bool is_ready() const {
        for (const auto& future : futures_) {
            if (future.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
                return false;
            }
        }
        return true;
    }
};

// ============================================================================
// Main Proxy Template Class
// ============================================================================

template <typename T, typename Enable = void> struct Proxy {
  static_assert(!std::is_same<T, Proxy>::value,
                "Cannot make a Proxy of a Proxy object");

  // Constructors
  Proxy()
      : location(Resource::Local()), addr(nullptr),
        metadata(nullptr) {
    LOGINFO("Constructing Proxy<{}> @{}", typeid(T).name(),
            static_cast<void *>(this));
  };

  Proxy(const Resource &r) : location(r), addr(nullptr), metadata(nullptr) {
    LOGINFO("Constructing Proxy<{}> @{}", typeid(T).name(),
            static_cast<void *>(this));
  };

  Proxy(const Resource &r, T &obj, T *dest = nullptr)
      : location(r), addr(dest == nullptr ? &obj : dest) {
    if (dest == nullptr)
      metadata = nullptr;
    else
      metadata = new Metadata_t<T>(obj);
    LOGINFO("Constructing Proxy<{}> @{} wrapping @{} with metadata @{}",
            typeid(T).name(), static_cast<void *>(this),
            static_cast<void *>(&obj), static_cast<void *>(metadata));
  };

  // Copy constructor
  Proxy(const Proxy<T> &other)
      : location(other.location), addr(other.addr), metadata(nullptr) {
    LOGINFO("Copy Constructing Proxy<{}> @{}", typeid(T).name(),
            static_cast<void *>(this));
    if (other.metadata != nullptr) {
      const Metadata_t<T> &tmp = *(other.metadata);
      metadata = new Metadata_t<T>(tmp);
    }
  };

  // Copy assignment
  Proxy<T> &operator=(const Proxy<T> &other) {
    if (this != &other) {
      if (metadata != nullptr)
        delete metadata;
      location = other.location;
      addr = other.addr;
      if (other.metadata != nullptr) {
        const Metadata_t<T> &tmp = *(other.metadata);
        metadata = new Metadata_t<T>(tmp);
      } else {
        metadata = nullptr;
      }
    }
    return *this;
  };

  // Move constructor
  Proxy(Proxy<T> &&other) : addr(nullptr), metadata(nullptr) {
    LOGINFO("Move Constructing Proxy<{}> @{}", typeid(T).name(),
            static_cast<void *>(this));
    location = other.location;
    addr = other.addr;
    metadata = other.metadata;
    other.metadata = nullptr;
  };

  // Destructor
  ~Proxy() {
    LOGINFO("Deconstructing Proxy<{}> @{} with metadata @{}", typeid(T).name(),
            static_cast<void *>(this), static_cast<void *>(metadata));
    if (metadata != nullptr)
      delete metadata;
  };

  /**
   * @brief Overloaded operator-> returns the address of the underlying object.
   */
  auto operator->() { return addr; };
  auto operator->() const { return addr; };

  /**
   * @brief Enhanced synchronous method call implementation with better error handling
   */
  template <typename RetType, typename... Args1, typename... Args2>
  RetType callSync(RetType (T::*memberFunc)(Args1...), Args2 &&...args) {
    struct ArgPack {
      std::tuple<Args2...> arguments;
      RetType (T::*func)(Args1...);
    };

    ArgPack pack{std::make_tuple(std::forward<Args2>(args)...), memberFunc};

    void *result_ptr = nullptr;

    try {
      if (location.is_host()) {
        // Host execution - direct function call
        if constexpr (!std::is_void_v<RetType>) {
          auto result = std::apply(
              [&](auto &&...a) {
                return (static_cast<T *>(addr)->*memberFunc)(
                    std::forward<decltype(a)>(a)...);
              },
              pack.arguments);
          result_ptr = new RetType(std::move(result));
        } else {
          std::apply(
              [&](auto &&...a) {
                (static_cast<T *>(addr)->*memberFunc)(
                    std::forward<decltype(a)>(a)...);
              },
              pack.arguments);
        }
      } else if (location.is_device()) {
        // Device execution - dispatch to backend
        switch (location.type) {
#ifdef USE_CUDA
        case ResourceType::CUDA:
          result_ptr = ProxyImpl::cuda_call_sync(
              addr, reinterpret_cast<void *>(&memberFunc), &pack, sizeof(pack),
              location, sizeof(RetType));
          break;
#endif
#ifdef USE_SYCL
        case ResourceType::SYCL:
          result_ptr = ProxyImpl::sycl_call_sync(
              addr, reinterpret_cast<void *>(&memberFunc), &pack, sizeof(pack),
              location, sizeof(RetType));
          break;
#endif
#ifdef USE_METAL
        case ResourceType::METAL:
          result_ptr = ProxyImpl::metal_call_sync(
              addr, reinterpret_cast<void *>(&memberFunc), &pack, sizeof(pack),
              location, sizeof(RetType));
          break;
#endif
        default:
          ARBD::throw_value_error(
              "Proxy::callSync(): Unsupported device type");
        }
      } else {
        ARBD::throw_value_error("Proxy::callSync(): Invalid resource location");
      }
    } catch (const std::exception &e) {
      LOGERROR("Proxy::callSync() failed: {}", e.what());
      throw;
    }

    if constexpr (!std::is_void_v<RetType>) {
      if (!result_ptr) {
        throw std::runtime_error("Proxy::callSync(): Null result pointer");
      }
      RetType result = *static_cast<RetType *>(result_ptr);
      delete static_cast<RetType *>(result_ptr);
      return result;
    }
  }

  /**
   * @brief Enhanced asynchronous method call implementation with RemoteResult
   */
  template <typename RetType, typename... Args1, typename... Args2>
  RemoteResult<RetType> callAsync(RetType (T::*memberFunc)(Args1...),
                                 Args2 &&...args) {
    // Pack arguments for implementation functions
    struct ArgPack {
      std::tuple<Args2...> arguments;
      RetType (T::*func)(Args1...);
    };

    ArgPack pack{std::make_tuple(std::forward<Args2>(args)...), memberFunc};

    auto future_result = std::async(std::launch::async, [this, pack, memberFunc]() -> void* {
      void *result_ptr = nullptr;
      
      try {
        if (location.is_host()) {
            // Host execution
            if constexpr (!std::is_void_v<RetType>) {
                auto result = std::apply([&](auto&&... a) {
                    return (static_cast<T*>(addr)->*memberFunc)(std::forward<decltype(a)>(a)...);
                }, pack.arguments);
                result_ptr = new RetType(std::move(result));
            } else {
                std::apply([&](auto&&... a) {
                    (static_cast<T*>(addr)->*memberFunc)(std::forward<decltype(a)>(a)...);
                }, pack.arguments);
            }
        } else if (location.is_device()) {
          // Device execution
          std::future<void *> backend_future;
          switch (location.type) {
#ifdef USE_CUDA
          case ResourceType::CUDA:
            backend_future = ProxyImpl::cuda_call_async(
                addr, reinterpret_cast<void *>(&memberFunc),
                const_cast<ArgPack *>(&pack), sizeof(pack), location,
                sizeof(RetType));
            break;
#endif
#ifdef USE_SYCL
          case ResourceType::SYCL:
            backend_future = ProxyImpl::sycl_call_async(
                addr, reinterpret_cast<void *>(&memberFunc),
                const_cast<ArgPack *>(&pack), sizeof(pack), location,
                sizeof(RetType));
            break;
#endif
#ifdef USE_METAL
          case ResourceType::METAL:
            backend_future = ProxyImpl::metal_call_async(
                addr, reinterpret_cast<void *>(&memberFunc),
                const_cast<ArgPack *>(&pack), sizeof(pack), location,
                sizeof(RetType));
            break;
#endif
          default:
            throw std::runtime_error(
                "Proxy::callAsync(): Unsupported device type");
          }
          if (backend_future.valid()) {
            result_ptr = backend_future.get();
          }
        } else {
            throw std::runtime_error("Proxy::callAsync(): Invalid resource location");
        }
      } catch (const std::exception& e) {
        LOGERROR("Proxy::callAsync() failed: {}", e.what());
        throw;
      }
      return result_ptr;
    });

    if constexpr (!std::is_void_v<RetType>) {
        void* result_ptr = future_result.get();
        if (!result_ptr) {
            throw std::runtime_error("Proxy::callAsync(): Null result pointer");
        }
        RetType result = *static_cast<RetType *>(result_ptr);
        delete static_cast<RetType *>(result_ptr);
        
        std::promise<void> promise;
        promise.set_value();
        return RemoteResult<RetType>(std::move(result), promise.get_future());
    } else {
        future_result.wait();
        std::promise<void> promise;
        promise.set_value();
        return RemoteResult<void>(promise.get_future());
    }
  }

  /**
   * @brief Batch method call for multiple operations
   */
  template <typename... Methods>
  auto callBatch(Methods&&... methods) {
    return std::make_tuple(methods(*this)...);
  }

  /**
   * @brief Get the resource location of this proxy
   */
  const Resource& get_location() const { return location; }

  /**
   * @brief Check if the proxy is valid (has a valid address)
   */
  bool is_valid() const { return addr != nullptr; }

  /**
   * @brief Get the raw address (for debugging)
   */
  void* get_address() const { return addr; }

  Resource location;
  T *addr;
  Metadata_t<T> *metadata;
};

// Specialization for arithmetic types
template <typename T>
struct Proxy<T, typename std::enable_if_t<std::is_arithmetic<T>::value>> {
  Proxy() : location{Resource::Local()}, addr{nullptr} {};
  Proxy(const Resource &r, T *obj) : location{r}, addr{obj} {};

  auto operator->() { return addr; }
  auto operator->() const { return addr; }

  const Resource& get_location() const { return location; }
  bool is_valid() const { return addr != nullptr; }
  void* get_address() const { return addr; }

  Resource location;
  T *addr;
};

// ============================================================================
// Template Function Implementations
// ============================================================================

template <typename T>
HOST inline Proxy<T> _send_ignoring_children(const Resource &location, T &obj,
                                             T *dest = nullptr) {
  LOGTRACE("   _send_ignoring_children...");

  void *result_ptr = ProxyImpl::send_ignoring_children(location, obj, dest);
  T *typed_dest = static_cast<T *>(result_ptr);

  LOGINFO("   creating Proxy...");
  return Proxy<T>(location, obj, typed_dest);
}

template <typename T>
  requires(!has_send_children<T>)
HOST inline Proxy<T> send(const Resource &location, T &obj, T *dest = nullptr) {
  Resource src = Resource::Local();
  TransferType transfer_type = get_transfer_type(src, location);
  LOGINFO("...Sending object {} @{} to device at {} (transfer type: {})",
          typeid(T).name(), static_cast<void *>(&obj),
          static_cast<void *>(dest), to_string(transfer_type));

  Proxy<T> ret = _send_ignoring_children(location, obj, dest);
  LOGTRACE("...done sending");
  return ret;
}

template <typename T>
  requires has_send_children<T>
HOST inline Proxy<T> send(const Resource &location, T &obj, T *dest = nullptr) {
  Resource src = Resource::Local();
  TransferType transfer_type = get_transfer_type(src, location);
  LOGINFO("Sending complex object {} @{} to device at {} (transfer type: {})",
          typeid(T).name(), static_cast<void *>(&obj),
          static_cast<void *>(dest), to_string(transfer_type));

  auto dummy = obj.send_children(location);
  Proxy<T> ret = _send_ignoring_children<T>(location, dummy, dest);
  LOGTRACE("... clearing dummy complex object");
  dummy.clear();
  LOGTRACE("... done sending");
  return ret;
}

template <typename T, typename... Args>
Proxy<T> construct_remote(Resource location, Args &&...args) {
  struct ArgPack {
    std::tuple<Args...> arguments;
  };

  ArgPack pack{std::make_tuple(std::forward<Args>(args)...)};

  void *result_ptr =
      ProxyImpl::construct_remote<T>(location, &pack, sizeof(pack));
  T *typed_ptr = static_cast<T *>(result_ptr);

  // Create a temporary object for metadata (this is a limitation of the current
  // design)
  T temp_obj{std::forward<Args>(args)...};
  return Proxy<T>(location, temp_obj, typed_ptr);
}

// ============================================================================
// Utility Functions for Proxy Management
// ============================================================================

/**
 * @brief Create a proxy chain for sequential operations
 */
template<typename T>
RemoteKernelChain make_proxy_chain(const Proxy<T>& proxy) {
    return RemoteKernelChain(proxy.get_location());
}

/**
 * @brief Batch send multiple objects to the same location
 */
template<typename... Types>
auto send_batch(const Resource& location, Types&... objects) {
    return std::make_tuple(send(location, objects)...);
}

/**
 * @brief Wait for multiple proxy operations to complete
 */
template<typename... Results>
void wait_all(Results&... results) {
    (results.wait(), ...);
}

} // namespace ARBD
