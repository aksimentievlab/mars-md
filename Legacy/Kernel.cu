#ifdef USE_CUDA

#include "Backend/Buffer.h"
#include "Backend/CUDA/CUDAManager.h"
#include "Backend/CUDA/CUDAProfiler.h"
#include "Backend/Events.h"
#include "Backend/Kernels.h"
#include "Backend/Resource.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>

namespace ARBD {
namespace Backend {

// ============================================================================
// CUDA-specific kernel templates for common operations
// ============================================================================

template<typename T>
__global__ void cuda_zero_memory_kernel(T* ptr, size_t count) {
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < count) {
		ptr[idx] = T{};
	}
}

template<typename T>
__global__ void cuda_fill_kernel(T* ptr, size_t count, T value) {
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < count) {
		ptr[idx] = value;
	}
}

template<typename T>
__global__ void cuda_copy_kernel(const T* src, T* dst, size_t count) {
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < count) {
		dst[idx] = src[idx];
	}
}

// ============================================================================
// CUDA Memory Operations
// ============================================================================

void cuda_zero_memory(void* ptr, size_t bytes) {
	cudaMemset(ptr, 0, bytes);
}

template<typename T>
void cuda_zero_typed_memory(T* ptr, size_t count) {
	dim3 block(256);
	dim3 grid((count + block.x - 1) / block.x);
	cuda_zero_memory_kernel<<<grid, block>>>(ptr, count);
	cudaDeviceSynchronize();
}

// ============================================================================
// CUDA Event Management
// ============================================================================

class CUDAEventPool {
  private:
	std::vector<cudaEvent_t> available_events_;
	std::mutex mutex_;

  public:
	static CUDAEventPool& instance() {
		static CUDAEventPool pool;
		return pool;
	}

	cudaEvent_t acquire() {
		std::lock_guard<std::mutex> lock(mutex_);
		if (available_events_.empty()) {
			cudaEvent_t event;
			cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
			return event;
		}

		cudaEvent_t event = available_events_.back();
		available_events_.pop_back();
		return event;
	}

	void release(cudaEvent_t event) {
		std::lock_guard<std::mutex> lock(mutex_);
		available_events_.push_back(event);
	}

	~CUDAEventPool() {
		for (auto event : available_events_) {
			cudaEventDestroy(event);
		}
	}
};

// ============================================================================
// CUDA Stream Management for Async Operations
// ============================================================================

class CUDAStreamManager {
  private:
	std::vector<cudaStream_t> streams_;
	std::atomic<size_t> current_stream_{0};
	const size_t num_streams_ = 4;

  public:
	static CUDAStreamManager& instance() {
		static CUDAStreamManager manager;
		return manager;
	}

	CUDAStreamManager() {
		streams_.resize(num_streams_);
		for (auto& stream : streams_) {
			cudaStreamCreate(&stream);
		}
	}

	cudaStream_t get_next_stream() {
		size_t idx = current_stream_.fetch_add(1) % num_streams_;
		return streams_[idx];
	}

	~CUDAStreamManager() {
		for (auto stream : streams_) {
			cudaStreamDestroy(stream);
		}
	}
};

// ============================================================================
// Advanced CUDA Kernel Launchers
// ============================================================================

// Kernel wrapper for functors that can't be directly passed to CUDA kernels
template<typename Func, typename... Args>
struct CUDAKernelWrapper {
	Func func;

	__device__ void operator()(size_t idx, Args... args) const {
		func(idx, args...);
	}
};

// Generic kernel that uses the wrapper
template<typename Wrapper, typename... Args>
__global__ void cuda_generic_kernel(size_t n, Wrapper wrapper, Args... args) {
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n) {
		wrapper(idx, args...);
	}
}

// Specialized launcher for lambda functions
template<typename Lambda, typename... Args>
Event launch_cuda_lambda(const Resource& resource,
						 size_t n,
						 Lambda&& lambda,
						 const Kernels::KernelConfig& config,
						 Args*... args) {
	// Calculate grid dimensions
	dim3 block(config.block_size);
	dim3 grid((n + block.x - 1) / block.x);

	// Get stream for async execution
	cudaStream_t stream = config.async ? CUDAStreamManager::instance().get_next_stream() : 0;

	// Create wrapper
	CUDAKernelWrapper<Lambda, Args*...> wrapper{lambda};

	// Launch kernel
	cuda_generic_kernel<<<grid, block, config.shared_memory, stream>>>(n, wrapper, args...);

	// Create and record event
	cudaEvent_t event = CUDAEventPool::instance().acquire();
	cudaEventRecord(event, stream);

	if (!config.async) {
		cudaEventSynchronize(event);
	}

	return Event(event, resource);
}

// ============================================================================
// Reduction Kernels
// ============================================================================

template<typename T, typename BinaryOp>
__global__ void cuda_reduce_kernel(const T* input, T* output, size_t n, BinaryOp op) {
	extern __shared__ T sdata[];

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	// Load data
	T val = (i < n) ? input[i] : T{};
	sdata[tid] = val;
	__syncthreads();

	// Reduction in shared memory
	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
		if (tid < s && i + s < n) {
			sdata[tid] = op(sdata[tid], sdata[tid + s]);
		}
		__syncthreads();
	}

	// Write result for this block
	if (tid == 0) {
		output[blockIdx.x] = sdata[0];
	}
}

template<typename T, typename BinaryOp>
Event reduce_async_cuda(const DeviceBuffer<T>& input,
						DeviceBuffer<T>& output,
						const Resource& resource,
						BinaryOp op) {
	size_t n = input.size();
	size_t block_size = 256;
	size_t grid_size = (n + block_size - 1) / block_size;

	// Temporary buffer for partial results
	DeviceBuffer<T> temp(grid_size);

	EventList deps;
	const T* input_ptr = input.get_read_access(deps);
	T* temp_ptr = temp.get_write_access(deps);
	T* output_ptr = output.get_write_access(deps);

	cudaStream_t stream = CUDAStreamManager::instance().get_next_stream();

	// First reduction pass
	cuda_reduce_kernel<<<grid_size, block_size, block_size * sizeof(T), stream>>>(input_ptr,
																				  temp_ptr,
																				  n,
																				  op);

	// Final reduction
	if (grid_size > 1) {
		cuda_reduce_kernel<<<1, grid_size, grid_size * sizeof(T), stream>>>(temp_ptr,
																			output_ptr,
																			grid_size,
																			op);
	} else {
		cudaMemcpyAsync(output_ptr, temp_ptr, sizeof(T), cudaMemcpyDeviceToDevice, stream);
	}

	cudaEvent_t event = CUDAEventPool::instance().acquire();
	cudaEventRecord(event, stream);

	return Event(event, resource);
}

// ============================================================================
// Matrix Operations (for demonstration)
// ============================================================================

template<typename T>
__global__ void cuda_matmul_kernel(const T* A, const T* B, T* C, size_t M, size_t N, size_t K) {
	size_t row = blockIdx.y * blockDim.y + threadIdx.y;
	size_t col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < M && col < N) {
		T sum = T{};
		for (size_t k = 0; k < K; ++k) {
			sum += A[row * K + k] * B[k * N + col];
		}
		C[row * N + col] = sum;
	}
}

template<typename T>
Event matmul_cuda(const DeviceBuffer<T>& A,
				  const DeviceBuffer<T>& B,
				  DeviceBuffer<T>& C,
				  size_t M,
				  size_t N,
				  size_t K,
				  const Resource& resource) {
	dim3 block(16, 16);
	dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

	EventList deps;
	const T* A_ptr = A.get_read_access(deps);
	const T* B_ptr = B.get_read_access(deps);
	T* C_ptr = C.get_write_access(deps);

	cudaStream_t stream = CUDAStreamManager::instance().get_next_stream();

	cuda_matmul_kernel<<<grid, block, 0, stream>>>(A_ptr, B_ptr, C_ptr, M, N, K);

	cudaEvent_t event = CUDAEventPool::instance().acquire();
	cudaEventRecord(event, stream);

	return Event(event, resource);
}

// ============================================================================
// Explicit instantiations for common types
// ============================================================================

#define INSTANTIATE_CUDA_KERNELS(T)                                              \
	template void cuda_zero_typed_memory<T>(T*, size_t);                         \
	template Event reduce_async_cuda<T, thrust::plus<T>>(const DeviceBuffer<T>&, \
														 DeviceBuffer<T>&,       \
														 const Resource&,        \
														 thrust::plus<T>);       \
	template Event matmul_cuda<T>(const DeviceBuffer<T>&,                        \
								  const DeviceBuffer<T>&,                        \
								  DeviceBuffer<T>&,                              \
								  size_t,                                        \
								  size_t,                                        \
								  size_t,                                        \
								  const Resource&);

INSTANTIATE_CUDA_KERNELS(float)
INSTANTIATE_CUDA_KERNELS(double)
INSTANTIATE_CUDA_KERNELS(int)
INSTANTIATE_CUDA_KERNELS(unsigned int)

} // namespace Backend
} // namespace ARBD

#endif // USE_CUDA
