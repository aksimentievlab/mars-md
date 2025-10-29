#pragma once
#include <cuda_runtime.h>

template<typename T>
__global__ void fill_kernel(T* __restrict__ ptr, T value, size_t num_elements) {
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_elements) {
		ptr[idx] = value;
	}
}
