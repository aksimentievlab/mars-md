#ifdef __CUDACC__ && USE_CUDA
#include "Random.h"
#include "Backend/CUDA/CUDAManager.h"
#include "Backend/Buffer.h"
#include "Backend/Events.h"
#include "Backend/Resource.h"
#include "openrand/philox.h"
#include "ARBDLogger.h"
#include "ARBDException.h"
#include "Random.h"
#include "openrand/philox.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>

namespace ARBD {

// CUDA Random Number Generation Kernels
namespace RandomCUDAKernels {

// Create Philox generator for thread
__device__ inline openrand::Philox create_philox(unsigned long seed, unsigned int thread_id, unsigned int offset) {
    return openrand::Philox(seed, thread_id + offset);
}

// Box-Muller transform for Gaussian from uniform
__device__ inline float2 box_muller(float u1, float u2) {
    float r = sqrtf(-2.0f * logf(u1));
    float theta = 2.0f * M_PI * u2;
    return make_float2(r * cosf(theta), r * sinf(theta));
}

// Box-Muller transform for double precision
__device__ inline double2 box_muller_double(double u1, double u2) {
    double r = sqrt(-2.0 * log(u1));
    double theta = 2.0 * M_PI * u2;
    return make_double2(r * cos(theta), r * sin(theta));
}

// Uniform float kernel
__global__ void generate_uniform_float_kernel(float* output, size_t count, 
                                            unsigned long seed, size_t offset, 
                                            float min, float max) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= count) return;
    
    auto rng = create_philox(seed, tid, offset);
    auto result = rng.draw_float4();
    
    output[tid] = min + result.x * (max - min);
}

// Gaussian float kernel  
__global__ void generate_gaussian_float_kernel(float* output, size_t count,
                                             unsigned long seed, size_t offset,
                                             float mean, float stddev) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;
    
    // Process pairs of outputs for Box-Muller
    for (size_t i = tid * 2; i < count; i += stride * 2) {
        auto rng = create_philox(seed, i, offset);
        auto uniform_vals = rng.draw_float4();
        
        float u1 = fmaxf(uniform_vals.x, 1e-7f);
        float u2 = uniform_vals.y;
        
        float2 gauss_pair = box_muller(u1, u2);
        
        if (i < count) {
            output[i] = mean + stddev * gauss_pair.x;
        }
        if (i + 1 < count) {
            output[i + 1] = mean + stddev * gauss_pair.y;
        }
    }
}

// Gaussian Vector3 kernel
__global__ void generate_gaussian_vector3_kernel(Vector3* output, size_t count,
                                                unsigned long seed, size_t offset,
                                                float mean, float stddev) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= count) return;
    
    auto rng = create_philox(seed, tid, offset);
    auto uniform_vals1 = rng.draw_float4();
    auto uniform_vals2 = rng.draw_float4();
    
    float u1 = fmaxf(uniform_vals1.x, 1e-7f);
    float u2 = uniform_vals1.y;
    float u3 = fmaxf(uniform_vals1.z, 1e-7f);
    float u4 = uniform_vals1.w;
    
    float2 gauss_pair1 = box_muller(u1, u2);
    float2 gauss_pair2 = box_muller(u3, u4);
    
    output[tid] = Vector3(
        mean + stddev * gauss_pair1.x,
        mean + stddev * gauss_pair1.y,
        mean + stddev * gauss_pair2.x
    );
}

// Uniform double kernel
__global__ void generate_uniform_double_kernel(double* output, size_t count,
                                             unsigned long seed, size_t offset,
                                             double min, double max) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= count) return;
    
    auto rng = create_philox(seed, tid, offset);
    uint64_t u64 = rng.template draw<uint64_t>();
    
    double u = u64 * 0x1.0p-64;  // Convert to [0,1)
    output[tid] = min + u * (max - min);
}

// Gaussian double kernel
__global__ void generate_gaussian_double_kernel(double* output, size_t count,
                                              unsigned long seed, size_t offset,
                                              double mean, double stddev) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;
    
    for (size_t i = tid * 2; i < count; i += stride * 2) {
        auto rng = create_philox(seed, i, offset);
        
        uint64_t u64_1 = rng.template draw<uint64_t>();
        uint64_t u64_2 = rng.template draw<uint64_t>();
        
        double u1 = fmax(u64_1 * 0x1.0p-64, 1e-15);
        double u2 = u64_2 * 0x1.0p-64;
        
        double2 gauss_pair = box_muller_double(u1, u2);
        
        if (i < count) {
            output[i] = mean + stddev * gauss_pair.x;
        }
        if (i + 1 < count) {
            output[i + 1] = mean + stddev * gauss_pair.y;
        }
    }
}

// Uniform int kernel
__global__ void generate_uniform_int_kernel(int* output, size_t count,
                                           unsigned long seed, size_t offset,
                                           int min, int max) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= count) return;
    
    auto rng = create_philox(seed, tid, offset);
    uint32_t u32 = rng.template draw<uint32_t>();
    
    unsigned int range = static_cast<unsigned int>(max - min + 1);
    if (range == 0) {
        output[tid] = min;
        return;
    }
    
    // Avoid modulo bias for more uniform distribution
    unsigned int limit = UINT_MAX - (UINT_MAX % range);
    while (u32 >= limit) {
        u32 = rng.template draw<uint32_t>();
    }
    
    output[tid] = min + static_cast<int>(u32 % range);
}

// Uniform unsigned int kernel
__global__ void generate_uniform_uint_kernel(unsigned int* output, size_t count,
                                            unsigned long seed, size_t offset,
                                            unsigned int min, unsigned int max) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= count) return;
    
    auto rng = create_philox(seed, tid, offset);
    uint32_t u32 = rng.template draw<uint32_t>();
    
    unsigned int range = max - min + 1;
    if (range == 0) {
        output[tid] = min;
        return;
    }
    
    unsigned int limit = UINT_MAX - (UINT_MAX % range);
    while (u32 >= limit) {
        u32 = rng.template draw<uint32_t>();
    }
    
    output[tid] = min + (u32 % range);
}

} // namespace RandomCUDAKernels

// RandomCUDA Implementation
template<size_t num_states>
RandomCUDA<num_states>::RandomCUDA(const Resource& resource) 
    : RandomDevice<num_states>(resource) {
    if (resource.type != ResourceType::CUDA) {
        throw_value_error("RandomCUDA requires CUDA resource");
    }
    LOGINFO("Created RandomCUDA with {} states", num_states);
}

template<size_t num_states>
Event RandomCUDA<num_states>::generate_uniform(DeviceBuffer<float>& output, float min, float max) {
    if (!this->initialized_) {
        throw_runtime_error("RandomCUDA not initialized - call init() first");
    }
    
    const size_t count = output.size();
    const int block_size = 256;
    const int grid_size = (count + block_size - 1) / block_size;
    
    // Get CUDA stream from manager
    auto& cuda_device = ARBD::CUDA::CUDAManager::get_current_device();
    cudaStream_t stream = cuda_device.get_next_stream();
    
    try {
        RandomCUDAKernels::generate_uniform_float_kernel<<<grid_size, block_size, 0, stream>>>(
            output.data(), count, this->seed_, this->offset_, min, max);
        
        cudaError_t cuda_error = cudaGetLastError();
        if (cuda_error != cudaSuccess) {
            throw_runtime_error("CUDA kernel launch failed: {}", cudaGetErrorString(cuda_error));
        }
        
        return Event(stream, this->resource_);
        
    } catch (const std::exception& e) {
        throw_runtime_error("CUDA error in uniform float generation: {}", e.what());
    }
}

template<size_t num_states>
Event RandomCUDA<num_states>::generate_gaussian(DeviceBuffer<float>& output, float mean, float stddev) {
    if (!this->initialized_) {
        throw_runtime_error("RandomCUDA not initialized - call init() first");
    }
    
    const size_t count = output.size();
    const int block_size = 256;
    const int grid_size = (count + block_size - 1) / block_size;
    
    auto& cuda_device = ARBD::CUDA::CUDAManager::get_current_device();
    cudaStream_t stream = cuda_device.get_next_stream();
    
    try {
        RandomCUDAKernels::generate_gaussian_float_kernel<<<grid_size, block_size, 0, stream>>>(
            output.data(), count, this->seed_, this->offset_, mean, stddev);
        
        cudaError_t cuda_error = cudaGetLastError();
        if (cuda_error != cudaSuccess) {
            throw_runtime_error("CUDA kernel launch failed: {}", cudaGetErrorString(cuda_error));
        }
        
        return Event(stream, this->resource_);
        
    } catch (const std::exception& e) {
        throw_runtime_error("CUDA error in Gaussian float generation: {}", e.what());
    }
}

template<size_t num_states>
Event RandomCUDA<num_states>::generate_gaussian_vector3(DeviceBuffer<Vector3>& output, float mean, float stddev) {
    if (!this->initialized_) {
        throw_runtime_error("RandomCUDA not initialized - call init() first");
    }
    
    const size_t count = output.size();
    const int block_size = 256;
    const int grid_size = (count + block_size - 1) / block_size;
    
    auto& cuda_device = ARBD::CUDA::CUDAManager::get_current_device();
    cudaStream_t stream = cuda_device.get_next_stream();
    
    try {
        RandomCUDAKernels::generate_gaussian_vector3_kernel<<<grid_size, block_size, 0, stream>>>(
            output.data(), count, this->seed_, this->offset_, mean, stddev);
        
        cudaError_t cuda_error = cudaGetLastError();
        if (cuda_error != cudaSuccess) {
            throw_runtime_error("CUDA kernel launch failed: {}", cudaGetErrorString(cuda_error));
        }
        
        return Event(stream, this->resource_);
        
    } catch (const std::exception& e) {
        throw_runtime_error("CUDA error in Gaussian Vector3 generation: {}", e.what());
    }
}

template<size_t num_states>
Event RandomCUDA<num_states>::generate_uniform(DeviceBuffer<double>& output, double min, double max) {
    if (!this->initialized_) {
        throw_runtime_error("RandomCUDA not initialized - call init() first");
    }
    
    const size_t count = output.size();
    const int block_size = 256;
    const int grid_size = (count + block_size - 1) / block_size;
    
    auto& cuda_device = ARBD::CUDA::CUDAManager::get_current_device();
    cudaStream_t stream = cuda_device.get_next_stream();
    
    try {
        RandomCUDAKernels::generate_uniform_double_kernel<<<grid_size, block_size, 0, stream>>>(
            output.data(), count, this->seed_, this->offset_, min, max);
        
        cudaError_t cuda_error = cudaGetLastError();
        if (cuda_error != cudaSuccess) {
            throw_runtime_error("CUDA kernel launch failed: {}", cudaGetErrorString(cuda_error));
        }
        
        return Event(stream, this->resource_);
        
    } catch (const std::exception& e) {
        throw_runtime_error("CUDA error in uniform double generation: {}", e.what());
    }
}

template<size_t num_states>
Event RandomCUDA<num_states>::generate_gaussian(DeviceBuffer<double>& output, double mean, double stddev) {
    if (!this->initialized_) {
        throw_runtime_error("RandomCUDA not initialized - call init() first");
    }
    
    const size_t count = output.size();
    const int block_size = 256;
    const int grid_size = (count + block_size - 1) / block_size;
    
    auto& cuda_device = ARBD::CUDA::CUDAManager::get_current_device();
    cudaStream_t stream = cuda_device.get_next_stream();
    
    try {
        RandomCUDAKernels::generate_gaussian_double_kernel<<<grid_size, block_size, 0, stream>>>(
            output.data(), count, this->seed_, this->offset_, mean, stddev);
        
        cudaError_t cuda_error = cudaGetLastError();
        if (cuda_error != cudaSuccess) {
            throw_runtime_error("CUDA kernel launch failed: {}", cudaGetErrorString(cuda_error));
        }
        
        return Event(stream, this->resource_);
        
    } catch (const std::exception& e) {
        throw_runtime_error("CUDA error in Gaussian double generation: {}", e.what());
    }
}

template<size_t num_states>
Event RandomCUDA<num_states>::generate_uniform_int(DeviceBuffer<int>& output, int min, int max) {
    if (!this->initialized_) {
        throw_runtime_error("RandomCUDA not initialized - call init() first");
    }
    
    const size_t count = output.size();
    const int block_size = 256;
    const int grid_size = (count + block_size - 1) / block_size;
    
    auto& cuda_device = ARBD::CUDA::CUDAManager::get_current_device();
    cudaStream_t stream = cuda_device.get_next_stream();
    
    try {
        RandomCUDAKernels::generate_uniform_int_kernel<<<grid_size, block_size, 0, stream>>>(
            output.data(), count, this->seed_, this->offset_, min, max);
        
        cudaError_t cuda_error = cudaGetLastError();
        if (cuda_error != cudaSuccess) {
            throw_runtime_error("CUDA kernel launch failed: {}", cudaGetErrorString(cuda_error));
        }
        
        return Event(stream, this->resource_);
        
    } catch (const std::exception& e) {
        throw_runtime_error("CUDA error in uniform int generation: {}", e.what());
    }
}

template<size_t num_states>
Event RandomCUDA<num_states>::generate_uniform_uint(DeviceBuffer<unsigned int>& output, unsigned int min, unsigned int max) {
    if (!this->initialized_) {
        throw_runtime_error("RandomCUDA not initialized - call init() first");
    }
    
    const size_t count = output.size();
    const int block_size = 256;
    const int grid_size = (count + block_size - 1) / block_size;
    
    auto& cuda_device = ARBD::CUDA::CUDAManager::get_current_device();
    cudaStream_t stream = cuda_device.get_next_stream();
    
    try {
        RandomCUDAKernels::generate_uniform_uint_kernel<<<grid_size, block_size, 0, stream>>>(
            output.data(), count, this->seed_, this->offset_, min, max);
        
        cudaError_t cuda_error = cudaGetLastError();
        if (cuda_error != cudaSuccess) {
            throw_runtime_error("CUDA kernel launch failed: {}", cudaGetErrorString(cuda_error));
        }
        
        return Event(stream, this->resource_);
        
    } catch (const std::exception& e) {
        throw_runtime_error("CUDA error in uniform uint generation: {}", e.what());
    }
}

// Explicit template instantiations for common sizes
template class RandomCUDA<32>;
template class RandomCUDA<64>;
template class RandomCUDA<128>;
template class RandomCUDA<256>;
template class RandomCUDA<512>;
template class RandomCUDA<1024>;

} // namespace ARBD

#endif