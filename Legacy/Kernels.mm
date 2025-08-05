#ifdef USE_METAL
#include "Backend/Buffer.h"
#include "Backend/Resource.h"
#include "Backend/Events.h"
#include "Backend/Kernels.h"
#include "METALManager.h"
#include <Metal/Metal.hpp>
#include <memory>
#include <vector>
#include <span>
#include "Metal/MTLComputeCommandEncoder.hpp"
namespace ARBD {
namespace Backend {

// ============================================================================
// C++20 Memory Operations
// ============================================================================

void metal_zero_memory(std::span<std::byte> memory, const Resource& resource) {
    std::ranges::fill(memory, std::byte{0});
}

template<typename T>
void metal_zero_typed_memory(std::span<T> data, const Resource& resource) {
    // Use actual Metal compute kernel for zeroing
    auto& device = METAL::METALManager::get_current_device();
    auto& queue = device.get_next_queue();
    
    void* cmd_buffer_ptr = queue.create_command_buffer();
    auto* cmd_buffer = static_cast<MTL::CommandBuffer*>(cmd_buffer_ptr);
    auto* encoder = cmd_buffer->computeCommandEncoder();
    
    // Get zero kernel pipeline
    MTL::ComputePipelineState* pipeline = METAL::METALManager::get_compute_pipeline_state("zero_buffer");
    encoder->setComputePipelineState(pipeline);
    
    // Set buffer argument
    MTL::Buffer* buffer = static_cast<MTL::Buffer*>(METAL::METALManager::get_metal_buffer_from_ptr(data.data()));
    encoder->setBuffer(buffer, 0, 0);
    
    // Dispatch
    MTL::Size grid_size = MTL::Size::Make(data.size(), 1, 1);
    NS::UInteger threadgroup_size = std::min(static_cast<NS::UInteger>(256), 
                                           pipeline->maxTotalThreadsPerThreadgroup());
    MTL::Size threadgroup = MTL::Size::Make(threadgroup_size, 1, 1);
    
    encoder->dispatchThreads(grid_size, threadgroup);
    encoder->endEncoding();
    
    ARBD::METAL::Event metal_event(cmd_buffer_ptr);
    metal_event.commit();
    metal_event.wait();
}

// ============================================================================
// C++20 Event Pool using RAII
// ============================================================================

class MetalEventPool {
private:
    std::vector<std::unique_ptr<ARBD::METAL::Event>> available_events_;
    std::mutex mutex_;
    
public:
    static MetalEventPool& instance() {
        static MetalEventPool pool;
        return pool;
    }
    
    std::unique_ptr<ARBD::METAL::Event> acquire() {
        std::lock_guard lock{mutex_};
        
        if (available_events_.empty()) {
            auto& device = METAL::METALManager::get_current_device();
            auto& queue = device.get_next_queue();
            void* cmd_buffer = queue.create_command_buffer();
            return std::make_unique<ARBD::METAL::Event>(cmd_buffer);
        }
        
        auto event = std::move(available_events_.back());
        available_events_.pop_back();
        return event;
    }
    
    void release(std::unique_ptr<ARBD::METAL::Event> event) {
        std::lock_guard lock{mutex_};
        available_events_.push_back(std::move(event));
    }
};

// ============================================================================
// Modern Queue Manager with C++20 Features
// ============================================================================

class MetalQueueManager {
private:
    struct QueueInfo {
        size_t device_id;
        size_t queue_id;
        std::shared_ptr<std::atomic<size_t>> pending_work;
        
        QueueInfo(size_t dev_id, size_t q_id) 
            : device_id{dev_id}, queue_id{q_id}, 
              pending_work{std::make_shared<std::atomic<size_t>>(0)} {}
        
        // Add move constructor
        QueueInfo(QueueInfo&&) = default;
        QueueInfo& operator=(QueueInfo&&) = default;
    };
    
    std::vector<QueueInfo> queues_;
    
public:
    static MetalQueueManager& instance() {
        static MetalQueueManager manager;
        return manager;
    }
    
    MetalQueueManager() {
        if (METAL::METALManager::devices().empty()) {
            return;
        }
        
        constexpr size_t queues_per_device = 3;
        const auto& devices = METAL::METALManager::devices();
        
        queues_.reserve(devices.size() * queues_per_device);
        
        for (size_t dev_id = 0; dev_id < devices.size(); ++dev_id) {
            for (size_t q_id = 0; q_id < queues_per_device; ++q_id) {
                queues_.emplace_back(dev_id, q_id);
            }
        }
    }
    
    METAL::Queue& get_least_loaded_queue() {
        if (queues_.empty()) {
            return METAL::METALManager::get_current_device().get_next_queue();
        }
        
        // Use C++20 ranges to find minimum
        auto min_it = std::ranges::min_element(queues_, 
            [](const auto& a, const auto& b) {
                return a.pending_work->load() < b.pending_work->load();
            });
        
        min_it->pending_work->fetch_add(1);
        
        auto& device = METAL::METALManager::get_current_device();
        return device.get_next_queue();
    }
    
    void mark_complete() {
        for (auto& queue_info : queues_) {
            if (auto current = queue_info.pending_work->load(); current > 0) {
                queue_info.pending_work->compare_exchange_weak(current, current - 1);
                break;
            }
        }
    }
};

// ============================================================================
// Modern Kernel Launch with C++20 Features
// ============================================================================

template<typename... Args>
Event launch_metal_kernel(const Resource& resource,
                          const std::string& kernel_name,
                          size_t n,
                          const Kernels::KernelConfig& config,
                          const EventList& deps,
                          Args*... args) {
    
    auto& device = METAL::METALManager::get_current_device();
    auto& queue = config.async ? 
        MetalQueueManager::instance().get_least_loaded_queue() : 
        device.get_next_queue();
    
    // Wait for dependencies using C++20 ranges
    const auto& events = deps.get_events();
    std::ranges::for_each(events, [](const auto& event) {
        if (event.get_resource().type == ResourceType::METAL && event.is_valid()) {
            if (auto* metal_event = static_cast<ARBD::METAL::Event*>(event.get_event_impl());
                metal_event && !metal_event->is_complete()) {
                metal_event->wait();
            }
        }
    });
    
    // Get pipeline state
    MTL::ComputePipelineState* pipeline = METAL::METALManager::get_compute_pipeline_state(kernel_name);
    
    // Create command buffer and encoder
    void* cmd_buffer_ptr = queue.create_command_buffer();
    auto* cmd_buffer = static_cast<MTL::CommandBuffer*>(cmd_buffer_ptr);
    auto* encoder = cmd_buffer->computeCommandEncoder();
    
    encoder->setComputePipelineState(pipeline);
    
    // Set buffer arguments using fold expression
    int buffer_index = 0;
    ((encoder->setBuffer(get_metal_buffer_from_ptr(args), 0, buffer_index++)), ...);
    
    // Dispatch
    MTL::Size grid_size = MTL::Size::Make(n, 1, 1);
    NS::UInteger threadgroup_size = std::min(static_cast<NS::UInteger>(config.block_size), 
                                           pipeline->maxTotalThreadsPerThreadgroup());
    MTL::Size threadgroup = MTL::Size::Make(threadgroup_size, 1, 1);
    
    encoder->dispatchThreads(grid_size, threadgroup);
    encoder->endEncoding();
    
    // Create event and commit
    ARBD::METAL::Event metal_event(cmd_buffer_ptr);
    metal_event.commit();
    
    if (!config.async) {
        metal_event.wait();
    }
    
    return Event(metal_event, resource);
}

// ============================================================================
// C++20 Reduction Operations
// ============================================================================

template<typename T>
Event reduce_async_metal(const DeviceBuffer<T>& input,
                        DeviceBuffer<T>& output,
                        const Resource& resource,
                        const std::string& reduce_kernel_name,
                        T identity = T{}) {
    
    const size_t n = input.size();
    auto& device = METAL::METALManager::get_current_device();
    auto& queue = device.get_next_queue();
    
    EventList deps;
    const T* input_ptr = input.get_read_access(deps);
    T* output_ptr = output.get_write_access(deps);
    
    void* cmd_buffer_ptr = queue.create_command_buffer();
    auto* cmd_buffer = static_cast<MTL::CommandBuffer*>(cmd_buffer_ptr);
    auto* encoder = cmd_buffer->computeCommandEncoder();
    
    MTL::ComputePipelineState* pipeline = METAL::METALManager::get_compute_pipeline_state(reduce_kernel_name);
    encoder->setComputePipelineState(pipeline);
    
    // Set buffers
    MTL::Buffer* input_buffer = static_cast<MTL::Buffer*>(METAL::METALManager::get_metal_buffer_from_ptr(const_cast<T*>(input_ptr)));
    MTL::Buffer* output_buffer = static_cast<MTL::Buffer*>(METAL::METALManager::get_metal_buffer_from_ptr(output_ptr));
    
    encoder->setBuffer(input_buffer, 0, 0);
    encoder->setBuffer(output_buffer, 0, 1);
    encoder->setBytes(&identity, sizeof(T), 2);
    
    // Dispatch
    MTL::Size grid_size = MTL::Size::Make(n, 1, 1);
    constexpr NS::UInteger default_threadgroup_size = 256;
    NS::UInteger threadgroup_size = std::min(default_threadgroup_size, 
                                           pipeline->maxTotalThreadsPerThreadgroup());
    MTL::Size threadgroup = MTL::Size::Make(threadgroup_size, 1, 1);
    
    encoder->dispatchThreads(grid_size, threadgroup);
    encoder->endEncoding();
    
    ARBD::METAL::Event metal_event(cmd_buffer_ptr);
    metal_event.commit();
    
    return Event(metal_event, resource);
}

// ============================================================================
// C++20 Explicit Instantiations
// ============================================================================

#define INSTANTIATE_METAL_KERNELS(T) \
    template void metal_zero_typed_memory<T>(std::span<T>, const Resource&); \
    template Event reduce_async_metal<T>(const DeviceBuffer<T>&, DeviceBuffer<T>&, \
                                        const Resource&, const std::string&, T);

INSTANTIATE_METAL_KERNELS(float)
INSTANTIATE_METAL_KERNELS(double)
INSTANTIATE_METAL_KERNELS(int)
INSTANTIATE_METAL_KERNELS(unsigned int)

} // namespace Backend
} // namespace ARBD

#endif // USE_METAL