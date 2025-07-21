#pragma once

#include <cstring>
#include <memory>
#include <type_traits>
#include <unordered_map>
#include <vector>
#include <tuple>
#include <functional>
#include <future>
#include <optional>
#include <map>

#ifdef USE_CUDA
#include "CUDA/CUDAManager.h"
#endif

#ifdef USE_SYCL
#include "SYCL/SYCLManager.h"
#endif

#ifdef USE_METAL
#include "METAL/METALManager.h"
#endif

#include "ARBDLogger.h"
#include "Resource.h"
#include "Events.h"

namespace ARBD {

/**
 * @brief Memory location tracking
 */
struct MemoryLocation {
  Resource resource;
  void *ptr;
  size_t size;
  bool is_valid;
  bool is_owned; // Track if this memory was allocated by UnifiedBuffer

  MemoryLocation() : ptr(nullptr), size(0), is_valid(false), is_owned(false) {}
  MemoryLocation(const Resource &res, void *p, size_t s, bool owned = true)
      : resource(res), ptr(p), size(s), is_valid(true), is_owned(owned) {}
};

/**
 * @brief Backend-agnostic memory operations
 */
class MemoryOps {
public:
    /**
     * @brief Allocate memory on specified backend
     */
    static void* allocate(size_t size, const Resource& resource) {
        if (resource.is_device()) {
            // Try each available device backend
#ifdef USE_CUDA
            void* ptr = nullptr;
            cudaMalloc(&ptr, size);
            return ptr;
#elif defined(USE_SYCL)
            auto& device = ARBD::SYCL::SYCLManager::get_current_device();
            auto& queue = device.get_next_queue();
            return sycl::malloc_device<char>(size, queue.get());
#elif defined(USE_METAL)
            return ARBD::METAL::METALManager::allocate_raw(size);
#else
            // CPU fallback for device resource when no device backend available
            return std::malloc(size);
#endif
        } else {
            // Host allocation
            return std::malloc(size);
        }
    }

    /**
     * @brief Deallocate memory on specified backend
     */
    static void deallocate(void* ptr, const Resource& resource) {
        if (!ptr) return;

        if (resource.is_device()) {
            // Try each available device backend
#ifdef USE_CUDA
            cudaFree(ptr);
#elif defined(USE_SYCL)
            auto& device = ARBD::SYCL::SYCLManager::get_current_device();
            auto& queue = device.get_next_queue();
            sycl::free(ptr, queue.get());
#elif defined(USE_METAL)
            ARBD::METAL::METALManager::deallocate_raw(ptr);
#else
            // CPU fallback for device resource when no device backend available
            std::free(ptr);
#endif
        } else {
            // Host deallocation
            std::free(ptr);
        }
    }

    /**
     * @brief Copy memory between resources (synchronous)
     */
    static void copy(void* dst, const void* src, size_t size,
                    const Resource& dst_res, const Resource& src_res) {
        // Same resource type - use backend-specific copy
        if (dst_res.is_device() && src_res.is_device()) {
            // Device-to-device copy within same backend
#ifdef USE_CUDA
            cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice);
            return;
#elif defined(USE_SYCL)
            auto& device = ARBD::SYCL::SYCLManager::get_current_device();
            auto& queue = device.get_next_queue();
            queue.get().memcpy(dst, src, size).wait();
            return;
#elif defined(USE_METAL)
            // Metal device-to-device copy would go here
            // For now, fall through to cross-backend copy
#else
            // CPU-to-CPU copy for device resource when no device backend
            std::memcpy(dst, src, size);
            return;
#endif
        } else if (dst_res.is_host() && src_res.is_host()) {
            // Host-to-host copy
            std::memcpy(dst, src, size);
            return;
        }

        // Cross-backend copy via host
        copy_cross_backend(dst, src, size, dst_res, src_res);
    }

    /**
     * @brief Copy from host to device
     */
    static void copy_from_host(void* device_dst, const void* host_src, size_t size,
                               const Resource& dst_res) {
        if (dst_res.is_device()) {
            // Host-to-device copy
#ifdef USE_CUDA
            cudaMemcpy(device_dst, host_src, size, cudaMemcpyHostToDevice);
#elif defined(USE_SYCL)
            auto& device = ARBD::SYCL::SYCLManager::get_current_device();
            auto& queue = device.get_next_queue();
            queue.get().memcpy(device_dst, host_src, size).wait();
#elif defined(USE_METAL)
            // Metal host-to-device copy would go here
            // For now, use regular memcpy
            std::memcpy(device_dst, host_src, size);
#else
            // CPU copy for device resource when no device backend
            std::memcpy(device_dst, host_src, size);
#endif
        } else {
            // Host-to-host copy
            std::memcpy(device_dst, host_src, size);
        }
    }

    /**
     * @brief Copy from device to host
     */
    static void copy_to_host(void* host_dst, const void* device_src, size_t size,
                            const Resource& src_res) {
        if (src_res.is_device()) {
            // Device-to-host copy
#ifdef USE_CUDA
            cudaMemcpy(host_dst, device_src, size, cudaMemcpyDeviceToHost);
#elif defined(USE_SYCL)
            auto& device = ARBD::SYCL::SYCLManager::get_current_device();
            auto& queue = device.get_next_queue();
            queue.get().memcpy(host_dst, device_src, size).wait();
#elif defined(USE_METAL)
            // Metal device-to-host copy would go here
            // For now, use regular memcpy
            std::memcpy(host_dst, device_src, size);
#else
            // CPU copy for device resource when no device backend
            std::memcpy(host_dst, device_src, size);
#endif
        } else {
            // Host-to-host copy
            std::memcpy(host_dst, device_src, size);
        }
    }

private:
    static void copy_cross_backend(void* dst, const void* src, size_t size,
                                  const Resource& dst_res, const Resource& src_res) {
        // Allocate temporary host buffer
        std::vector<char> temp(size);
        
        // Copy from source to host
        copy_to_host(temp.data(), src, size, src_res);
        
        // Copy from host to destination
        copy_from_host(dst, temp.data(), size, dst_res);
    }
};

/**
 * @brief Enhanced device buffer with event tracking for single-location memory
 */
template <typename T> 
class DeviceBuffer {
public:
    using value_type = T;
    
    // Constructors
    DeviceBuffer() : size_(0), device_ptr_(nullptr) {}
    
    explicit DeviceBuffer(size_t count, const Resource& resource) 
        : size_(count), resource_(resource), device_ptr_(nullptr) {
        if (count > 0) {
            allocate_device_memory();
        }
    }
    
    // Move semantics
    DeviceBuffer(DeviceBuffer&& other) noexcept 
        : size_(other.size_), resource_(other.resource_), 
          device_ptr_(other.device_ptr_), proxy_(std::move(other.proxy_)),
          last_event_(std::move(other.last_event_)) {
        other.device_ptr_ = nullptr;
        other.size_ = 0;
    }
    
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            deallocate_device_memory();
            size_ = other.size_;
            resource_ = other.resource_;
            device_ptr_ = other.device_ptr_;
            proxy_ = std::move(other.proxy_);
            last_event_ = std::move(other.last_event_);
            other.device_ptr_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }
    
    // Delete copy operations
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    
    ~DeviceBuffer() {
        deallocate_device_memory();
    }
    
    // Access methods with event tracking
    T* get_write_access(EventList& depends_list) {
        depends_list.wait_all();  // Wait for dependencies
        return device_ptr_;
    }
    
    const T* get_read_access(EventList& depends_list) const {
        depends_list.wait_all();  // Wait for dependencies
        return device_ptr_;
    }
    
    void complete_event_state(const Event& e) {
        last_event_ = e;
    }
    
    Event get_last_event() const { return last_event_.value_or(Event{}); }
    
    // Properties
    size_t size() const { return size_; }
    bool empty() const { return size_ == 0; }
    const Resource& get_resource() const { return resource_; }
    T* data() { return device_ptr_; }
    const T* data() const { return device_ptr_; }
    
    // Enhanced data transfer operations with proper backend handling
    void copy_from_host(const T* host_data, size_t count = 0) {
        if (count == 0) count = size_;
        if (count > size_) {
            throw std::runtime_error("Copy size exceeds buffer size");
        }
        
        MemoryOps::copy_from_host(device_ptr_, host_data, count * sizeof(T), resource_);
    }
    
    void copy_to_host(T* host_data, size_t count = 0) const {
        if (count == 0) count = size_;
        if (count > size_) {
            throw std::runtime_error("Copy size exceeds buffer size");
        }
        
        MemoryOps::copy_to_host(host_data, device_ptr_, count * sizeof(T), resource_);
    }

private:
    size_t size_;
    Resource resource_;
    T* device_ptr_;                    // Generic device pointer
    std::optional<void*> proxy_;       // Type-erased proxy pointer 
    std::optional<Event> last_event_;  // Track last event for synchronization
    
    void allocate_device_memory() {
        device_ptr_ = static_cast<T*>(MemoryOps::allocate(size_ * sizeof(T), resource_));
    }
    
    void deallocate_device_memory() {
        if (device_ptr_) {
            MemoryOps::deallocate(device_ptr_, resource_);
            device_ptr_ = nullptr;
        }
    }
};

/**
 * @brief Unified buffer for cross-backend memory management with enhanced event tracking
 */
template <typename T> class UnifiedBuffer {
private:
  std::unordered_map<size_t, MemoryLocation> locations_;
  size_t count_;
  Resource primary_location_;
  Event last_event_; // Add event tracking to UnifiedBuffer

  /**
   * @brief Allocate memory using appropriate backend manager
   */
  void *allocate_memory(size_t size, const Resource &resource) {
    return MemoryOps::allocate(size, resource);
  }

  /**
   * @brief Deallocate memory using appropriate backend manager
   */
  void deallocate_memory(void *ptr, const Resource &resource) {
    MemoryOps::deallocate(ptr, resource);
  }

  /**
   * @brief Enhanced copy memory between resources with proper backend handling
   */
  void copy_memory(void *dst, const void *src, size_t size,
                   const Resource &src_res, const Resource &dst_res) {
    MemoryOps::copy(dst, src, size, dst_res, src_res);
  }

public:
  /**
   * @brief Construct unified buffer
   */
  explicit UnifiedBuffer(size_t count, const Resource &primary_res = Resource())
      : count_(count), primary_location_(primary_res) {
    if (count_ > 0) {
      void *ptr = allocate_memory(sizeof(T) * count_, primary_res);
      locations_[primary_location_.id] =
          MemoryLocation(primary_location_, ptr, sizeof(T) * count_, true); // owned=true
    }
  }

  /**
   * @brief Construct from existing data
   */
  UnifiedBuffer(size_t count, T *existing_data, const Resource &location)
      : count_(count), primary_location_(location) {
    locations_[location.id] =
        MemoryLocation(location, existing_data, sizeof(T) * count_, false); // owned=false
  }

  ~UnifiedBuffer() {
    for (auto &[id, loc] : locations_) {
      if (loc.is_owned) { // Only deallocate memory we allocated
        deallocate_memory(loc.ptr, loc.resource);
      }
    }
  }

  // No copy constructor - use explicit clone()
  UnifiedBuffer(const UnifiedBuffer &) = delete;
  UnifiedBuffer &operator=(const UnifiedBuffer &) = delete;

  // Move constructor
  UnifiedBuffer(UnifiedBuffer &&other) noexcept
      : locations_(std::move(other.locations_)), count_(other.count_),
        primary_location_(other.primary_location_), last_event_(std::move(other.last_event_)) {
    other.count_ = 0;
  }

  UnifiedBuffer &operator=(UnifiedBuffer &&other) noexcept {
    if (this != &other) {
      for (auto &[id, loc] : locations_) {
        if (loc.is_owned) { // Only deallocate memory we allocated
          deallocate_memory(loc.ptr, loc.resource);
        }
      }
      locations_ = std::move(other.locations_);
      count_ = other.count_;
      primary_location_ = other.primary_location_;
      last_event_ = std::move(other.last_event_);
      other.count_ = 0;
    }
    return *this;
  }

  /**
   * @brief Get pointer for specific resource with event tracking
   */
  T *get_ptr(const Resource &resource) {
    ensure_available_at(resource);
    auto it = locations_.find(resource.id);
    return (it != locations_.end()) ? static_cast<T *>(it->second.ptr)
                                    : nullptr;
  }

  /**
   * @brief Get const pointer for specific resource
   */
  const T *get_ptr(const Resource &resource) const {
    auto it = locations_.find(resource.id);
    return (it != locations_.end()) ? static_cast<const T *>(it->second.ptr)
                                    : nullptr;
  }

  /**
   * @brief Get write access with event dependency tracking
   */
  T* get_write_access(const Resource &resource, EventList& depends_list) {
    depends_list.wait_all();  // Wait for dependencies
    return get_ptr(resource);
  }

  /**
   * @brief Get read access with event dependency tracking
   */
  const T* get_read_access(const Resource &resource, EventList& depends_list) const {
    depends_list.wait_all();  // Wait for dependencies
    return get_ptr(resource);
  }

  /**
   * @brief Complete event state tracking
   */
  void complete_event_state(const Event& e) {
    last_event_ = e;
  }

  /**
   * @brief Get last event
   */
  Event get_last_event() const { return last_event_; }

  /**
   * @brief Ensure data is available at target resource
   */
  void ensure_available_at(const Resource &target) {
    if (locations_.find(target.id) != locations_.end()) {
      return; // Already available
    }

    if (count_ == 0) {
      return; // No data to migrate
    }

    // Find a source location to copy from
    MemoryLocation *source = nullptr;
    for (auto &[id, loc] : locations_) {
      if (loc.is_valid && loc.ptr) {
        source = &loc;
        break;
      }
    }

    if (!source) {
      ARBD::throw_value_error(
          "No valid source location found for data migration");
    }

    // Allocate at target
    void *target_ptr = allocate_memory(sizeof(T) * count_, target);

    // Copy data
    copy_memory(target_ptr, source->ptr, sizeof(T) * count_, source->resource,
                target);

    // Store location
    locations_[target.id] =
        MemoryLocation(target, target_ptr, sizeof(T) * count_, true);

    LOGINFO("UnifiedBuffer: migrated data from {} to {}",
            source->resource.getTypeString(), target.getTypeString());
  }

  /**
   * @brief Release memory at specific resource
   */
  void release_at(const Resource &resource) {
    auto it = locations_.find(resource.id);
    if (it != locations_.end()) {
      if (it->second.is_owned) { // Only deallocate memory we allocated
        deallocate_memory(it->second.ptr, resource);
      }
      locations_.erase(it);
    }
  }

  /**
   * @brief Get element count
   */
  size_t size() const { return count_; }

  /**
   * @brief Check if empty
   */
  bool empty() const { return count_ == 0; }

  /**
   * @brief Get primary location
   */
  const Resource &primary_location() const { return primary_location_; }

  /**
   * @brief Get all locations where data is available
   */
  std::vector<Resource> available_locations() const {
    std::vector<Resource> result;
    for (const auto &[id, loc] : locations_) {
      if (loc.is_valid) {
        result.push_back(loc.resource);
      }
    }
    return result;
  }

  /**
   * @brief Synchronize all locations
   */
  void synchronize_all() { ensure_available_at(primary_location_); }

  /**
   * @brief Enhanced data transfer operations
   */
  void copy_from_host(const T* host_data, const Resource &resource, size_t count = 0) {
    if (count == 0) count = count_;
    if (count > count_) {
      throw std::runtime_error("Copy size exceeds buffer size");
    }
    
    ensure_available_at(resource);
    T* ptr = get_ptr(resource);
    MemoryOps::copy_from_host(ptr, host_data, count * sizeof(T), resource);
  }
  
  void copy_to_host(T* host_data, const Resource &resource, size_t count = 0) const {
    if (count == 0) count = count_;
    if (count > count_) {
      throw std::runtime_error("Copy size exceeds buffer size");
    }
    
    const T* ptr = get_ptr(resource);
    if (!ptr) {
      throw std::runtime_error("Data not available at specified resource");
    }
    
    MemoryOps::copy_to_host(host_data, ptr, count * sizeof(T), resource);
  }
};

/**
 * @brief Generic multi-reference adapter that works with any type providing access methods
 */
template<typename... Args>
struct MultiRef {
    std::tuple<Args&...> storage;
    
    MultiRef(Args&... args) : storage(args...) {}
    
    template<size_t I>
    auto& get() { return std::get<I>(storage); }
    
    template<size_t I>
    const auto& get() const { return std::get<I>(storage); }
    
    static constexpr size_t size() { return sizeof...(Args); }
    
    auto get_read_access(EventList& depends_list) {
        return std::apply([&](auto&... args) {
            return std::make_tuple(args.get_read_access(depends_list)...);
        }, storage);
    }
    
    auto get_write_access(EventList& depends_list) {
        return std::apply([&](auto&... args) {
            return std::make_tuple(args.get_write_access(depends_list)...);
        }, storage);
    }
    
    void complete_event_state(const Event& e) {
        std::apply([&](auto&... args) {
            (args.complete_event_state(e), ...);
        }, storage);
    }
    
    // Get raw pointers for kernel launch (for compatibility)
    template<typename EventListType>
    auto get_pointers(EventListType& deps) {
        return get_write_access(deps);
    }
};

// Specialization for const buffers (read-only access)
template<typename... Args>
struct MultiRef<const Args...> {
    std::tuple<const Args&...> storage;
    
    MultiRef(const Args&... args) : storage(args...) {}
    
    template<size_t I>
    const auto& get() const { return std::get<I>(storage); }
    
    static constexpr size_t size() { return sizeof...(Args); }
    
    auto get_read_access(EventList& depends_list) const {
        return std::apply([&](auto&... args) {
            return std::make_tuple(args.get_read_access(depends_list)...);
        }, storage);
    }
    
    void complete_event_state(const Event& e) {
        std::apply([&](auto&... args) {
            (args.complete_event_state(e), ...);
        }, storage);
    }
    
    // Get raw pointers for kernel launch (for compatibility)
    template<typename EventListType>
    auto get_pointers(EventListType& deps) const {
        return get_read_access(deps);
    }
};

// Helper to create MultiRef with type deduction
template<typename... Args>
auto make_multi_ref(Args&... args) {
    return MultiRef<Args...>(args...);
}

// ============================================================================
// Generic Allocator
// ============================================================================

template<typename T>
class GenericAllocator {
public:
    static DeviceBuffer<T> allocate(size_t count, const Resource& resource) {
        return DeviceBuffer<T>(count, resource);
    }
    
    static DeviceBuffer<T> allocate_zeroed(size_t count, const Resource& resource) {
        DeviceBuffer<T> buffer(count, resource);
        zero_device_memory(buffer, resource);
        return buffer;
    }
    
    template<typename InitFunc>
    static DeviceBuffer<T> allocate_initialized(size_t count, 
                                               const Resource& resource,
                                               InitFunc&& init_func) {
        DeviceBuffer<T> buffer(count, resource);
        
        // Initialize on host then copy to device
        std::vector<T> host_data(count);
        for (size_t i = 0; i < count; ++i) {
            host_data[i] = init_func(i);
        }
        
        buffer.copy_from_host(host_data.data(), count);
        return buffer;
    }

    // Specialized allocator for common patterns
    static DeviceBuffer<T> allocate_sequence(size_t count, 
                                            const Resource& resource,
                                            T start = T{0},
                                            T step = T{1}) {
        return allocate_initialized(count, resource,
            [start, step](size_t i) { return start + static_cast<T>(i) * step; });
    }

private:
    static void zero_device_memory(DeviceBuffer<T>& buffer, const Resource& resource) {
        // Zero on host then copy to device
        std::vector<T> zero_data(buffer.size(), T{});
        buffer.copy_from_host(zero_data.data(), buffer.size());
    }
};


template<typename T>
class TypedAllocator { // TODO: implement this
public:
    static DeviceBuffer<T> allocate(size_t count, const Resource& resource) {
        return DeviceBuffer<T>(count, resource);
    }

    static DeviceBuffer<T> allocate_zeroed(size_t count, const Resource& resource) {
        DeviceBuffer<T> buffer(count, resource);

        // Zero initialization kernel
        kernel_call(resource,
                   MultiRef<>{},  // No inputs
                   make_multi_ref(buffer),
                   count,
                   [](size_t i, T* output) {
                       output[i] = T{};
                   });

        return buffer;
    }

    template<typename InitFunc>
    static DeviceBuffer<T> allocate_initialized(size_t count,
                                               const Resource& resource,
                                               InitFunc&& init_func) {
        DeviceBuffer<T> buffer(count, resource);

        // Custom initialization kernel
        kernel_call(resource,
                   MultiRef<>{},  // No inputs
                   make_multi_ref(buffer),
                   count,
                   [init_func](size_t i, T* output) {
                       output[i] = init_func(i);
                   });

        return buffer;
    }

    // Specialized allocator for common patterns
    static DeviceBuffer<T> allocate_sequence(size_t count,
                                            const Resource& resource,
                                            T start = T{0},
                                            T step = T{1}) {
        return allocate_initialized(count, resource,
            [start, step](size_t i) { return start + static_cast<T>(i) * step; });
    }
};

// Convenience aliases
template<typename T>
using Allocator = GenericAllocator<T>;

template<typename T>
using AdvancedAllocator = TypedAllocator<T>;

// Specialized aliases for common types
using FloatAllocator = TypedAllocator<float>;
using DoubleAllocator = TypedAllocator<double>;
using IntAllocator = TypedAllocator<int>;
/**
 * @brief Copy data between buffers - type-agnostic
 * Fallback for fill_async in Kernels.h
 */
template<typename T>
void copy_buffer(const DeviceBuffer<T>& source,
                DeviceBuffer<T>& destination,
                const Resource& resource) {
    if (source.size() != destination.size()) {
        throw std::runtime_error("Buffer size mismatch in copy operation");
    }
    
    // Use host as intermediate for cross-device copy
    std::vector<T> temp_data(source.size());
    source.copy_to_host(temp_data.data());
    destination.copy_from_host(temp_data.data());
}

/**
 * @brief Fill buffer with value - type-agnostic
 */
template<typename T>
void fill_buffer(DeviceBuffer<T>& buffer,
                const T& value,
                const Resource& resource) {
    // Fill on host then copy to device
    std::vector<T> fill_data(buffer.size(), value);
    buffer.copy_from_host(fill_data.data());
}

} // namespace ARBD
