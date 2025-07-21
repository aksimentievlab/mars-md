#ifdef USE_METAL
#include "ARBDLogger.h"
#include "METALManager.h"
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <algorithm>
#include <iostream>
#include <unordered_map>

namespace ARBD {
namespace METAL {

// Static member initialization
std::vector<METALManager::Device> METALManager::all_devices_;
std::vector<METALManager::Device> METALManager::devices_;
int METALManager::current_device_{0};
bool METALManager::prefer_low_power_{false};

// DeviceMemory template implementation
template<typename T>
DeviceMemory<T>::DeviceMemory(void* device, size_t count) : device_(device), size_(count) {
    if (count > 0 && device) {
        id<MTLDevice> mtl_device = (__bridge id<MTLDevice>)device;
        id<MTLBuffer> mtl_buffer = [mtl_device newBufferWithLength:count * sizeof(T) 
                                                           options:MTLResourceStorageModeShared];
        if (!mtl_buffer) {
            ARBD_Exception(ExceptionType::MetalRuntimeError, 
                "Failed to allocate {} elements of type {}", count, typeid(T).name());
        }
        buffer_ = (__bridge_retained void*)mtl_buffer;
    }
}

template<typename T>
DeviceMemory<T>::~DeviceMemory() {
    if (buffer_) {
        id<MTLBuffer> mtl_buffer = (__bridge_transfer id<MTLBuffer>)buffer_;
        // ARC will handle cleanup
        mtl_buffer = nil;
    }
}

template<typename T>
DeviceMemory<T>::DeviceMemory(DeviceMemory&& other) noexcept
    : buffer_(std::exchange(other.buffer_, nullptr))
    , size_(std::exchange(other.size_, 0))
    , device_(std::exchange(other.device_, nullptr)) {}

template<typename T>
DeviceMemory<T>& DeviceMemory<T>::operator=(DeviceMemory&& other) noexcept {
    if (this != &other) {
        if (buffer_) {
            id<MTLBuffer> mtl_buffer = (__bridge_transfer id<MTLBuffer>)buffer_;
            mtl_buffer = nil;
        }
        buffer_ = std::exchange(other.buffer_, nullptr);
        size_ = std::exchange(other.size_, 0);
        device_ = std::exchange(other.device_, nullptr);
    }
    return *this;
}

template<typename T>
void DeviceMemory<T>::copyFromHost(std::span<const T> host_data) {
    if (host_data.size() > size_) {
        ARBD_Exception(ExceptionType::ValueError, 
            "Tried to copy {} elements but only {} allocated", 
            host_data.size(), size_);
    }
    if (!buffer_ || host_data.empty()) return;
    
    id<MTLBuffer> mtl_buffer = (__bridge id<MTLBuffer>)buffer_;
    void* buffer_contents = [mtl_buffer contents];
    std::memcpy(buffer_contents, host_data.data(), host_data.size() * sizeof(T));
}

template<typename T>
void DeviceMemory<T>::copyToHost(std::span<T> host_data) const {
    if (host_data.size() > size_) {
        ARBD_Exception(ExceptionType::ValueError,
            "Tried to copy {} elements but only {} allocated",
            host_data.size(), size_);
    }
    if (!buffer_ || host_data.empty()) return;
    
    id<MTLBuffer> mtl_buffer = (__bridge id<MTLBuffer>)buffer_;
    void* buffer_contents = [mtl_buffer contents];
    std::memcpy(host_data.data(), buffer_contents, host_data.size() * sizeof(T));
}

template<typename T>
T* DeviceMemory<T>::get() noexcept {
    if (!buffer_) return nullptr;
    id<MTLBuffer> mtl_buffer = (__bridge id<MTLBuffer>)buffer_;
    return static_cast<T*>([mtl_buffer contents]);
}

template<typename T>
const T* DeviceMemory<T>::get() const noexcept {
    if (!buffer_) return nullptr;
    id<MTLBuffer> mtl_buffer = (__bridge id<MTLBuffer>)buffer_;
    return static_cast<const T*>([mtl_buffer contents]);
}

// Explicit template instantiations for common types
template class DeviceMemory<float>;
template class DeviceMemory<double>;
template class DeviceMemory<int>;
template class DeviceMemory<unsigned int>;
template class DeviceMemory<char>;
template class DeviceMemory<unsigned char>;

// Queue implementation
Queue::Queue(void* device) : device_(device) {
    if (device) {
        id<MTLDevice> mtl_device = (__bridge id<MTLDevice>)device;
        id<MTLCommandQueue> mtl_queue = [mtl_device newCommandQueue];
        if (!mtl_queue) {
            ARBD_Exception(ExceptionType::MetalRuntimeError, "Failed to create Metal command queue");
        }
        queue_ = (__bridge_retained void*)mtl_queue;
    }
}

Queue::~Queue() {
    if (queue_) {
        id<MTLCommandQueue> mtl_queue = (__bridge_transfer id<MTLCommandQueue>)queue_;
        // ARC will handle cleanup
        mtl_queue = nil;
    }
}

Queue::Queue(Queue&& other) noexcept
    : queue_(std::exchange(other.queue_, nullptr))
    , device_(std::exchange(other.device_, nullptr)) {}

Queue& Queue::operator=(Queue&& other) noexcept {
    if (this != &other) {
        if (queue_) {
            id<MTLCommandQueue> mtl_queue = (__bridge_transfer id<MTLCommandQueue>)queue_;
            mtl_queue = nil;
        }
        queue_ = std::exchange(other.queue_, nullptr);
        device_ = std::exchange(other.device_, nullptr);
    }
    return *this;
}

void Queue::synchronize() {
    if (!queue_) return;
    
    id<MTLCommandQueue> mtl_queue = (__bridge id<MTLCommandQueue>)queue_;
    id<MTLCommandBuffer> sync_buffer = [mtl_queue commandBuffer];
    [sync_buffer commit];
    [sync_buffer waitUntilCompleted];
}

void* Queue::create_command_buffer() {
    if (!queue_) {
        ARBD_Exception(ExceptionType::MetalRuntimeError, "Queue not initialized");
    }
    
    id<MTLCommandQueue> mtl_queue = (__bridge id<MTLCommandQueue>)queue_;
    id<MTLCommandBuffer> cmd_buffer = [mtl_queue commandBuffer];
    if (!cmd_buffer) {
        ARBD_Exception(ExceptionType::MetalRuntimeError, "Failed to create command buffer");
    }
    
    return (__bridge_retained void*)cmd_buffer;
}

void Queue::commit_and_wait(void* command_buffer) {
    if (!command_buffer) return;
    
    id<MTLCommandBuffer> cmd_buffer = (__bridge id<MTLCommandBuffer>)command_buffer;
    [cmd_buffer commit];
    [cmd_buffer waitUntilCompleted];
}

bool Queue::is_available() const {
    return queue_ != nullptr;
}

// Event implementation
Event::Event(void* command_buffer) : command_buffer_(command_buffer) {
    if (command_buffer_) {
        start_time_ = std::chrono::high_resolution_clock::now();
        timing_available_ = true;
    }
}

Event::~Event() {
    if (command_buffer_) {
        id<MTLCommandBuffer> cmd_buffer = (__bridge_transfer id<MTLCommandBuffer>)command_buffer_;
        cmd_buffer = nil;
    }
}

Event::Event(Event&& other) noexcept
    : command_buffer_(std::exchange(other.command_buffer_, nullptr))
    , start_time_(other.start_time_)
    , end_time_(other.end_time_)
    , timing_available_(other.timing_available_) {}

Event& Event::operator=(Event&& other) noexcept {
    if (this != &other) {
        if (command_buffer_) {
            id<MTLCommandBuffer> cmd_buffer = (__bridge_transfer id<MTLCommandBuffer>)command_buffer_;
            cmd_buffer = nil;
        }
        command_buffer_ = std::exchange(other.command_buffer_, nullptr);
        start_time_ = other.start_time_;
        end_time_ = other.end_time_;
        timing_available_ = other.timing_available_;
    }
    return *this;
}

void Event::commit() {
    if (!command_buffer_) return;
    
    id<MTLCommandBuffer> cmd_buffer = (__bridge id<MTLCommandBuffer>)command_buffer_;
    [cmd_buffer commit];
}

void Event::wait() {
    if (!command_buffer_) return;
    
    id<MTLCommandBuffer> cmd_buffer = (__bridge id<MTLCommandBuffer>)command_buffer_;

    MTLCommandBufferStatus status = [cmd_buffer status];
    if (status == MTLCommandBufferStatusNotEnqueued) {
        // Command buffer was never committed, just return
        return;
    }
    
    [cmd_buffer waitUntilCompleted];
    
    if (timing_available_) {
        end_time_ = std::chrono::high_resolution_clock::now();
    }
}

bool Event::is_complete() const {
    if (!command_buffer_) return true;
    
    id<MTLCommandBuffer> cmd_buffer = (__bridge id<MTLCommandBuffer>)command_buffer_;
    MTLCommandBufferStatus status = [cmd_buffer status];
    
    // If the command buffer was never committed, consider it complete
    if (status == MTLCommandBufferStatusNotEnqueued) {
        return true;
    }
    
    return status == MTLCommandBufferStatusCompleted;
}

std::chrono::nanoseconds Event::get_execution_time() const {
    if (!timing_available_) {
        return std::chrono::nanoseconds{0};
    }
    
    return std::chrono::duration_cast<std::chrono::nanoseconds>(end_time_ - start_time_);
}

// Device class implementation
METALManager::Device::Device(void* device, unsigned int id) 
    : id_(id), device_(device) {
    
    query_device_properties();
    
    LOGINFO("Metal Device {} initialized: {}", id_, name_.c_str());
    LOGINFO("  Max threads per group: {}, Unified memory: {}, Low power: {}", 
         max_threads_per_group_, has_unified_memory_, is_low_power_);
    
    // Create queues
    try {
        for (size_t i = 0; i < queues_.size(); ++i) { 
            queues_[i] = Queue(device_); 
        }
    } catch (const ARBD::Exception& e) {
        LOGERROR("ARBD::Exception during Metal Queue construction for device {}: {}", id_, e.what());
        throw; 
    } catch (const std::exception& e) {
        LOGERROR("Unexpected std::exception during Metal Queue construction for device {}: {}", id_, e.what());
        throw; 
    }
}

void METALManager::Device::query_device_properties() {
    try {
        if (!device_) {
            name_ = "Unknown Device";
            max_threads_per_group_ = 1;
            recommended_max_working_set_size_ = 0;
            has_unified_memory_ = false;
            is_low_power_ = false;
            is_removable_ = false;
            supports_compute_ = false;
            return;
        }
        
        id<MTLDevice> mtl_device = (__bridge id<MTLDevice>)device_;
        
        name_ = std::string([[mtl_device name] UTF8String]);
        max_threads_per_group_ = [mtl_device maxThreadsPerThreadgroup].width;
        recommended_max_working_set_size_ = [mtl_device recommendedMaxWorkingSetSize];
        has_unified_memory_ = [mtl_device hasUnifiedMemory];
        is_low_power_ = [mtl_device isLowPower];
        is_removable_ = [mtl_device isRemovable];
        
        // Check if device supports compute shaders
        supports_compute_ = true; // All Metal devices support compute shaders
        
    } catch (const std::exception& e) {
        std::cerr << "!!! query_device_properties caught exception: " << e.what() 
                  << " for device (name might be uninit): " << name_ << std::endl;
        // Set defaults in case of error
        name_ = "Unknown Device";
        max_threads_per_group_ = 1;
        recommended_max_working_set_size_ = 0;
        has_unified_memory_ = false;
        is_low_power_ = false;
        is_removable_ = false;
        supports_compute_ = false;
    }
}

void METALManager::Device::synchronize_all_queues() const {
    for (const auto& queue : queues_) {
        // The queue object is const, but synchronize is not.
        // It's safe to const_cast here because synchronize doesn't change the logical state.
        const_cast<Queue&>(queue).synchronize();
    }
}

// METALManager static methods implementation
void METALManager::init() {
    LOGINFO("Initializing Metal Manager...");
    
    all_devices_.clear();
    devices_.clear();
    current_device_ = 0;
    
    discover_devices();
    
    if (all_devices_.empty()) {
        ARBD_Exception(ExceptionType::ValueError, "No Metal devices found");
    }
    
    LOGINFO("Found {} Metal device(s)", all_devices_.size());
}

void METALManager::discover_devices() {
    try {
        NSArray<id<MTLDevice>> *mtl_devices = MTLCopyAllDevices();
        
        if ([mtl_devices count] == 0) {
            LOGWARN("No Metal devices found on this system");
            return;
        }
        
        // Collect device information
        struct DeviceInfo {
            id<MTLDevice> device;
            unsigned int id;
            bool is_low_power;
        };
        
        std::vector<DeviceInfo> potential_device_infos;
        
        for (NSUInteger i = 0; i < [mtl_devices count]; ++i) {
            id<MTLDevice> device = mtl_devices[i];
            
            try {
                bool low_power = [device isLowPower];
                potential_device_infos.push_back({device, static_cast<unsigned int>(i), low_power});
                
                LOGDEBUG("Found Metal device: {} (Low power: {})",
                    [[device name] UTF8String], low_power);
                
            } catch (const std::exception& e) {
                LOGWARN("Exception during Metal device discovery for device {}: {}", (unsigned long)i, e.what());
            }
        }
        
        // Sort devices based on preference
        std::stable_sort(potential_device_infos.begin(), potential_device_infos.end(), 
            [](const DeviceInfo& a, const DeviceInfo& b) {
                if (METALManager::prefer_low_power_) {
                    if (a.is_low_power != b.is_low_power) {
                        return a.is_low_power;
                    }
                } else {
                    if (a.is_low_power != b.is_low_power) {
                        return !a.is_low_power; // Prefer high-performance devices
                    }
                }
                return a.id < b.id;
            });
        
        // Construct Device objects
        all_devices_.clear();
        all_devices_.reserve(potential_device_infos.size());
        
        for (size_t i = 0; i < potential_device_infos.size(); ++i) {
            const auto& device_info = potential_device_infos[i];
            void* device_ptr = (__bridge void*)device_info.device;
            all_devices_.emplace_back(device_ptr, static_cast<unsigned int>(i));
        }
        
    } catch (const std::exception& e) {
        LOGERROR("Exception during Metal device discovery: {}", e.what());
    }
}

void METALManager::load_info() {
    init();
    devices_ = std::move(all_devices_);
    init_devices();
}

void METALManager::init_devices() {
    LOGINFO("Initializing Metal devices...");
    std::string msg;
    
    for (size_t i = 0; i < devices_.size(); i++) {
        if (i > 0) {
            if (i == devices_.size() - 1) {
                msg += " and ";
            } else {
                msg += ", ";
            }
        }
        msg += std::to_string(devices_[i].get_id());
        
        LOGDEBUG("Metal Device {} ready: {}", devices_[i].get_id(), devices_[i].name().c_str());
    }
    
    LOGINFO("Initialized Metal devices: {}", msg.c_str());
    current_device_ = 0;
}

void METALManager::select_devices(std::span<const unsigned int> device_ids) {
    devices_.clear();
    devices_.reserve(device_ids.size());
    
    for (unsigned int id : device_ids) {
        if (id >= all_devices_.size()) {
            ARBD_Exception(ExceptionType::ValueError, 
                "Invalid device ID: {}", id);
        }
        devices_.emplace_back(all_devices_[id].metal_device(), id);
    }
    init_devices();
}

void METALManager::use(int device_id) {
    if (devices_.empty()) {
        ARBD_Exception(ExceptionType::ValueError, "No devices selected");
    }
    current_device_ = device_id % static_cast<int>(devices_.size());
}

void METALManager::sync(int device_id) {
    if (device_id >= static_cast<int>(devices_.size())) {
        ARBD_Exception(ExceptionType::ValueError, 
            "Invalid device ID: {}", device_id);
    }
    devices_[device_id].synchronize_all_queues();
}

void METALManager::sync() {
    for (auto& device : devices_) {
        device.synchronize_all_queues();
    }
}

void METALManager::finalize() {
    sync();
    devices_.clear();
    all_devices_.clear();
    current_device_ = 0;
    prefer_low_power_ = false;
}

int METALManager::current() {
    return current_device_;
}

void METALManager::prefer_low_power(bool prefer) {
    prefer_low_power_ = prefer;
    
    // Re-sort devices based on new preference
    if (!all_devices_.empty()) {
        std::sort(all_devices_.begin(), all_devices_.end(), 
            [prefer](const Device& a, const Device& b) {
                if (prefer) {
                    if (a.is_low_power() != b.is_low_power()) {
                        return a.is_low_power();
                    }
                } else {
                    if (a.is_low_power() != b.is_low_power()) {
                        return !a.is_low_power();
                    }
                }
                return a.get_id() < b.get_id();
            });
            
        // Reassign IDs after sorting
        for (size_t i = 0; i < all_devices_.size(); ++i) {
            all_devices_[i].set_id(static_cast<unsigned int>(i));
        }
    }
}

std::vector<unsigned int> METALManager::get_discrete_gpu_device_ids() {
    std::vector<unsigned int> discrete_ids;
    for (const auto& device : all_devices_) {
        if (!device.is_low_power() && !device.has_unified_memory()) {
            discrete_ids.push_back(device.get_id());
        }
    }
    return discrete_ids;
}

std::vector<unsigned int> METALManager::get_integrated_gpu_device_ids() {
    std::vector<unsigned int> integrated_ids;
    for (const auto& device : all_devices_) {
        if (device.has_unified_memory()) {
            integrated_ids.push_back(device.get_id());
        }
    }
    return integrated_ids;
}

std::vector<unsigned int> METALManager::get_low_power_device_ids() {
    std::vector<unsigned int> low_power_ids;
    for (const auto& device : all_devices_) {
        if (device.is_low_power()) {
            low_power_ids.push_back(device.get_id());
        }
    }
    return low_power_ids;
}

// Static buffer map to track MTLBuffer objects for raw pointer deallocation
static std::unordered_map<void*, void*> metal_buffer_map;

// C-style allocation functions for compatibility with UnifiedBuffer
void* METALManager::allocate_raw(size_t size) {
    if (devices_.empty()) {
        ARBD_Exception(ExceptionType::ValueError, "No Metal devices available for allocation");
    }
    
    auto& device = get_current_device();
    id<MTLDevice> mtl_device = (__bridge id<MTLDevice>)device.metal_device();
    id<MTLBuffer> mtl_buffer = [mtl_device newBufferWithLength:size 
                                                       options:MTLResourceStorageModeShared];
    if (!mtl_buffer) {
        ARBD_Exception(ExceptionType::MetalRuntimeError, "Metal buffer allocation failed");
    }
    
    // Store the buffer reference for later deallocation
    void* contents = [mtl_buffer contents];
    
    // We need to retain the buffer to prevent premature deallocation
    // Store it in the global map for later retrieval during deallocation
    metal_buffer_map[contents] = (__bridge_retained void*)mtl_buffer;
    
    return contents;
}

void METALManager::deallocate_raw(void* ptr) {
    if (!ptr) return;
    
    // Retrieve and release the stored buffer
    auto it = metal_buffer_map.find(ptr);
    if (it != metal_buffer_map.end()) {
        id<MTLBuffer> mtl_buffer = (__bridge_transfer id<MTLBuffer>)it->second;
        metal_buffer_map.erase(it);
        // ARC will handle cleanup when mtl_buffer goes out of scope
        mtl_buffer = nil;
    }
}

} // namespace METAL
} // namespace ARBD

#endif // USE_METAL