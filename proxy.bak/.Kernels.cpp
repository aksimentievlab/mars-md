#ifdef USE_SYCL

#include "Backend/Resource.h"
#include <sycl/sycl.hpp>
#include <mutex>
#include <atomic>
#include "SYCLManager.h"
#include "Backend/Events.h"

namespace ARBD {
namespace Backend {

// ============================================================================
// SYCL Memory Operations
// ============================================================================

void sycl_zero_memory(void* ptr, size_t bytes, const Resource& resource) {
    auto& device = SYCL::SYCLManager::get_device(resource.id);
    auto& queue = device.get_next_queue();
    
    queue.get().memset(ptr, 0, bytes).wait();
}

template<typename T>
void sycl_zero_typed_memory(T* ptr, size_t count, const Resource& resource) {
    auto& device = SYCL::SYCLManager::get_device(resource.id);
    auto& queue = device.get_next_queue();
    
    queue.get().parallel_for(sycl::range<1>(count), [=](sycl::id<1> idx) {
        ptr[idx] = T{};
    }).wait();
}

// ============================================================================
// SYCL Event Pool for efficient event management
// ============================================================================

class SYCLEventPool {
private:
    std::vector<std::shared_ptr<sycl::event>> available_events_;
    std::mutex mutex_;
    
public:
    static SYCLEventPool& instance() {
        static SYCLEventPool pool;
        return pool;
    }
    
    std::shared_ptr<sycl::event> acquire() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (available_events_.empty()) {
            return std::make_shared<sycl::event>();
        }
        
        auto event = available_events_.back();
        available_events_.pop_back();
        return event;
    }
    
    void release(std::shared_ptr<sycl::event> event) {
        std::lock_guard<std::mutex> lock(mutex_);
        available_events_.push_back(event);
    }
};

// ============================================================================
// SYCL Queue Manager for load balancing
// ============================================================================

class SYCLQueueManager {
private:
    struct QueueInfo {
        size_t device_id;
        size_t queue_id;
        std::atomic<size_t> pending_work{0};
    };
    
    std::vector<QueueInfo> queues_;
    
public:
    static SYCLQueueManager& instance() {
        static SYCLQueueManager manager;
        return manager;
    }
    
    SYCLQueueManager() {
        // Use SYCLManager's device management instead of creating raw queues
        if (SYCL::SYCLManager::devices().empty()) {
            return; // No devices available
        }
        
        // For each device, use multiple queue IDs for load balancing
        const size_t queues_per_device = 4;
        for (size_t dev_id = 0; dev_id < SYCL::SYCLManager::devices().size(); ++dev_id) {
            for (size_t q_id = 0; q_id < queues_per_device; ++q_id) {
                queues_.push_back({dev_id, q_id, 0});
            }
        }
    }
    
    SYCL::Queue& get_least_loaded_queue() {
        if (queues_.empty()) {
            // Fallback to current device
            return SYCL::SYCLManager::get_current_device().get_next_queue();
        }
        
        size_t min_idx = 0;
        size_t min_work = queues_[0].pending_work.load();
        
        for (size_t i = 1; i < queues_.size(); ++i) {
            size_t work = queues_[i].pending_work.load();
            if (work < min_work) {
                min_work = work;
                min_idx = i;
            }
        }
        
        queues_[min_idx].pending_work++;
        auto& device = SYCL::SYCLManager::get_device(queues_[min_idx].device_id);
        return device.get_queue(queues_[min_idx].queue_id);
    }
    
    void mark_complete(SYCL::Queue& queue) {
        // Just decrement any pending work since we can't match specific queues
        for (auto& q : queues_) {
            if (q.pending_work.load() > 0) {
                q.pending_work--;
                break;
            }
        }
    }
};

// ============================================================================
// SYCL Kernel Launch Helpers
// ============================================================================

// Helper to handle SYCL dependencies
std::vector<sycl::event> convert_to_sycl_events(const EventList& deps) {
    std::vector<sycl::event> sycl_deps;
    
    for (const auto& event : deps.get_events()) {
        if (event.get_resource().type == Resource::SYCL && event.is_valid()) {
            auto sycl_event_ptr = static_cast<sycl::event*>(event.event_impl_.get());
            if (sycl_event_ptr) {
                sycl_deps.push_back(*sycl_event_ptr);
            }
        }
    }
    
    return sycl_deps;
}

// Generic SYCL kernel launcher
template<typename KernelFunc, typename... Args>
Event launch_sycl_kernel(const Resource& resource,
                        size_t n,
                        KernelFunc&& kernel,
                        const LaunchConfig& config,
                        const EventList& deps,
                        Args*... args) {
    auto& device = SYCL::SYCLManager::get_device(resource.id);
    auto& queue = config.async ? 
        SYCLQueueManager::instance().get_least_loaded_queue() : 
        device.get_next_queue().get();
    
    // Convert dependencies
    auto sycl_deps = convert_to_sycl_events(deps);
    
    // Launch kernel
    auto event = queue.submit([&](sycl::handler& h) {
        // Set dependencies
        for (const auto& dep : sycl_deps) {
            h.depends_on(dep);
        }
        
        // Launch parallel kernel
        h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx) {
            kernel(idx[0], args...);
        });
    });
    
    if (!config.async) {
        event.wait();
    }
    
    return Event(event, resource);
}

// ============================================================================
// SYCL Reduction Operations
// ============================================================================

template<typename T, typename BinaryOp>
Event reduce_async_sycl(const DeviceBuffer<T>& input,
                       DeviceBuffer<T>& output,
                       const Resource& resource,
                       BinaryOp op,
                       T identity = T{}) {
    size_t n = input.size();
    auto& device = SYCL::SYCLManager::get_device(resource.id);
    auto& queue = device.get_next_queue().get();
    
    EventList deps;
    const T* input_ptr = input.get_read_access(deps);
    T* output_ptr = output.get_write_access(deps);
    
    // Convert dependencies
    auto sycl_deps = convert_to_sycl_events(deps);
    
    auto event = queue.submit([&](sycl::handler& h) {
        // Set dependencies
        for (const auto& dep : sycl_deps) {
            h.depends_on(dep);
        }
        
        // Use SYCL reduction
        auto reduction = sycl::reduction(output_ptr, identity, op);
        
        h.parallel_for(sycl::range<1>(n), reduction,
            [=](sycl::id<1> idx, auto& sum) {
                sum.combine(input_ptr[idx]);
            });
    });
    
    return Event(event, resource);
}

// ============================================================================
// SYCL Matrix Operations
// ============================================================================

template<typename T>
Event matmul_sycl(const DeviceBuffer<T>& A,
                 const DeviceBuffer<T>& B,
                 DeviceBuffer<T>& C,
                 size_t M, size_t N, size_t K,
                 const Resource& resource) {
    auto& device = SYCL::SYCLManager::get_device(resource.id);
    auto& queue = device.get_next_queue().get();
    
    EventList deps;
    const T* A_ptr = A.get_read_access(deps);
    const T* B_ptr = B.get_read_access(deps);
    T* C_ptr = C.get_write_access(deps);
    
    // Convert dependencies
    auto sycl_deps = convert_to_sycl_events(deps);
    
    // Tile size for better cache usage
    constexpr size_t TILE_SIZE = 16;
    
    auto event = queue.submit([&](sycl::handler& h) {
        // Set dependencies
        for (const auto& dep : sycl_deps) {
            h.depends_on(dep);
        }
        
        // Local memory for tiles
        sycl::local_accessor<T, 2> A_tile(sycl::range<2>(TILE_SIZE, TILE_SIZE), h);
        sycl::local_accessor<T, 2> B_tile(sycl::range<2>(TILE_SIZE, TILE_SIZE), h);
        
        h.parallel_for(
            sycl::nd_range<2>(
                sycl::range<2>((M + TILE_SIZE - 1) / TILE_SIZE * TILE_SIZE,
                              (N + TILE_SIZE - 1) / TILE_SIZE * TILE_SIZE),
                sycl::range<2>(TILE_SIZE, TILE_SIZE)
            ),
            [=](sycl::nd_item<2> item) {
                const size_t row = item.get_global_id(0);
                const size_t col = item.get_global_id(1);
                const size_t local_row = item.get_local_id(0);
                const size_t local_col = item.get_local_id(1);
                
                T sum = T{};
                
                // Iterate over tiles
                for (size_t tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
                    // Load tiles into local memory
                    if (row < M && tile * TILE_SIZE + local_col < K) {
                        A_tile[local_row][local_col] = 
                            A_ptr[row * K + tile * TILE_SIZE + local_col];
                    } else {
                        A_tile[local_row][local_col] = T{};
                    }
                    
                    if (col < N && tile * TILE_SIZE + local_row < K) {
                        B_tile[local_row][local_col] = 
                            B_ptr[(tile * TILE_SIZE + local_row) * N + col];
                    } else {
                        B_tile[local_row][local_col] = T{};
                    }
                    
                    item.barrier(sycl::access::fence_space::local_space);
                    
                    // Compute partial dot product
                    for (size_t k = 0; k < TILE_SIZE; ++k) {
                        sum += A_tile[local_row][k] * B_tile[k][local_col];
                    }
                    
                    item.barrier(sycl::access::fence_space::local_space);
                }
                
                // Write result
                if (row < M && col < N) {
                    C_ptr[row * N + col] = sum;
                }
            }
        );
    });
    
    return Event(event, resource);
}

// ============================================================================
// SYCL Scan (Prefix Sum) Operations
// ============================================================================

template<typename T, typename BinaryOp>
Event scan_async_sycl(const DeviceBuffer<T>& input,
                     DeviceBuffer<T>& output,
                     const Resource& resource,
                     BinaryOp op,
                     bool inclusive = true) {
    size_t n = input.size();
    auto& device = SYCL::SYCLManager::get_device(resource.id);
    auto& queue = device.get_next_queue().get();
    
    EventList deps;
    const T* input_ptr = input.get_read_access(deps);
    T* output_ptr = output.get_write_access(deps);
    
    // For simplicity, using sequential scan for small arrays
    // For large arrays, would implement parallel scan algorithm
    if (n < 1024) {
        auto event = queue.submit([&](sycl::handler& h) {
            h.single_task([=]() {
                if (n == 0) return;
                
                if (inclusive) {
                    output_ptr[0] = input_ptr[0];
                    for (size_t i = 1; i < n; ++i) {
                        output_ptr[i] = op(output_ptr[i-1], input_ptr[i]);
                    }
                } else {
                    T acc = T{};
                    for (size_t i = 0; i < n; ++i) {
                        output_ptr[i] = acc;
                        acc = op(acc, input_ptr[i]);
                    }
                }
            });
        });
        
        return Event(event, resource);
    }
    
    // For larger arrays, implement work-efficient parallel scan
    // This is a simplified version - production code would use a more efficient algorithm
    size_t work_group_size = 256;
    size_t num_work_groups = (n + work_group_size - 1) / work_group_size;
    
    // Temporary buffer for partial sums
    DeviceBuffer<T> partial_sums(num_work_groups, resource);
    T* partial_sums_ptr = partial_sums.get_write_access(deps);
    
    // Phase 1: Local scan within work groups
    auto event1 = queue.submit([&](sycl::handler& h) {
        sycl::local_accessor<T, 1> local_data(sycl::range<1>(work_group_size), h);
        
        h.parallel_for(
            sycl::nd_range<1>(num_work_groups * work_group_size, work_group_size),
            [=](sycl::nd_item<1> item) {
                size_t global_id = item.get_global_id(0);
                size_t local_id = item.get_local_id(0);
                size_t group_id = item.get_group(0);
                
                // Load data
                T val = (global_id < n) ? input_ptr[global_id] : T{};
                local_data[local_id] = val;
                item.barrier();
                
                // Up-sweep phase
                for (size_t offset = 1; offset < work_group_size; offset *= 2) {
                    if (local_id >= offset) {
                        val = op(local_data[local_id - offset], val);
                    }
                    item.barrier();
                    local_data[local_id] = val;
                    item.barrier();
                }
                
                // Write output
                if (global_id < n) {
                    output_ptr[global_id] = inclusive ? val : 
                        (local_id > 0 ? local_data[local_id - 1] : T{});
                }
                
                // Save partial sum for next phase
                if (local_id == work_group_size - 1) {
                    partial_sums_ptr[group_id] = val;
                }
            }
        );
    });
    
    // Phase 2: Scan partial sums (recursive call for large arrays)
    // ... Implementation continues ...
    
    return Event(event1, resource);
}

// ============================================================================
// SYCL Memory Allocator Implementations
// ============================================================================

template<typename T>
void TypedAllocator<T>::zero_sycl_memory(void* ptr, size_t bytes, const Resource& resource) {
    sycl_zero_memory(ptr, bytes, resource);
}

// ============================================================================
// Explicit instantiations
// ============================================================================

#define INSTANTIATE_SYCL_KERNELS(T) \
    template void sycl_zero_typed_memory<T>(T*, size_t, const Resource&); \
    template Event reduce_async_sycl<T, sycl::plus<T>>(const DeviceBuffer<T>&, DeviceBuffer<T>&, const Resource&, sycl::plus<T>, T); \
    template Event matmul_sycl<T>(const DeviceBuffer<T>&, const DeviceBuffer<T>&, DeviceBuffer<T>&, size_t, size_t, size_t, const Resource&); \
    template Event scan_async_sycl<T, sycl::plus<T>>(const DeviceBuffer<T>&, DeviceBuffer<T>&, const Resource&, sycl::plus<T>, bool);

INSTANTIATE_SYCL_KERNELS(float)
INSTANTIATE_SYCL_KERNELS(double)
INSTANTIATE_SYCL_KERNELS(int)
INSTANTIATE_SYCL_KERNELS(unsigned int)

} // namespace Backend
} // namespace ARBD

#endif // USE_SYCL
#endif // USE_SYCL