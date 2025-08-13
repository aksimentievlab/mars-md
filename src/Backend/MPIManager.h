// src/Backend/MPI/MPIManager.h
#pragma once
#include <mpi.h>
#include "Backend/Buffer.h"
#include "Backend/Resource.h"
#include "Backend/Events.h"

namespace ARBD::MPI {

class Manager {
private:
    MPI_Comm comm_;
    int rank_;
    int size_;
    bool initialized_ = false;
    
    // Pinned buffers cache
    mutable std::unordered_map<size_t, void*> staging_buffers_;
    
public:
    static Manager& instance() {
        static Manager instance;
        return instance;
    }
    
    void init() {
        if (!initialized_) {
            int provided;
            MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &provided);
            MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
            MPI_Comm_size(MPI_COMM_WORLD, &size_);
            comm_ = MPI_COMM_WORLD;
            initialized_ = true;
            LOGINFO("MPI Manager initialized: rank {}/{}", rank_, size_);
        }
    }
    
    void finalize() {
        if (initialized_) {
            for (auto& [size, ptr] : staging_buffers_) {
                free(ptr);
            }
            MPI_Finalize();
            initialized_ = false;
        }
    }

    template<typename T>
    Event allReduce(DeviceBuffer<T>& buffer, idx_t count, const Resource& resource) {
        T* host_buffer = get_staging_buffer<T>(count, resource);
        
        // Download
        buffer.copy_to_host(host_buffer, count);
        
        // MPI collective
        MPI_Allreduce(MPI_IN_PLACE, host_buffer, count, 
                      get_mpi_type<T>(), MPI_SUM, comm_);
        
        // Upload
        buffer.copy_from_host(host_buffer, count);
        
        // Return a completed event
        return Event(nullptr, resource);
    }
    
    template<typename T>
    Event broadcast(DeviceBuffer<T>& buffer, idx_t count, int root, const Resource& resource) {
        T* host_buffer = get_staging_buffer<T>(count, resource);
        
        if (rank_ == root) {
            buffer.copy_to_host(host_buffer, count);
        }
        
        MPI_Bcast(host_buffer, count, get_mpi_type<T>(), root, comm_);
        
        if (rank_ != root) {
            buffer.copy_from_host(host_buffer, count);
        }
        
        return Event(nullptr, resource);
    }
    
    int get_rank() const { return rank_; }
    int get_size() const { return size_; }
    
private:
    template<typename T>
    T* get_staging_buffer(idx_t count, const Resource& resource) {
        idx_t bytes = count * sizeof(T);
        auto it = staging_buffers_.find(bytes);
        
        if (it == staging_buffers_.end()) {
            void* ptr = nullptr;
            
            // Use pinned memory based on backend
            #ifdef USE_CUDA
            if (resource.type == ResourceType::CUDA) {
                cudaMallocHost(&ptr, bytes);
            } else
            #endif
            #ifdef USE_SYCL
            if (resource.type == ResourceType::SYCL) {
                // Use your SYCLManager to get pinned memory
                auto& device = SYCL::Manager::devices()[resource.id];
                auto& queue = device.get_queue(0);
                ptr = sycl::malloc_host(bytes, queue);
            } else
            #endif
            {
                ptr = aligned_alloc(256, bytes);
            }
            
            staging_buffers_[bytes] = ptr;
            return static_cast<T*>(ptr);
        }
        
        return static_cast<T*>(it->second);
    }
    
    template<typename T>
    MPI_Datatype get_mpi_type() {
        if constexpr (std::is_same_v<T, float>) return MPI_FLOAT;
        else if constexpr (std::is_same_v<T, double>) return MPI_DOUBLE;
        else if constexpr (std::is_same_v<T, int>) return MPI_INT;
        else return MPI_BYTE;
    }
};

}