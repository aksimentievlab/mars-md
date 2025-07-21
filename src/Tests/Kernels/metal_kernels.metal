#include <metal_stdlib>
using namespace metal;

// Simple vector addition kernel for testing
kernel void vector_add(device const float* a [[buffer(0)]],
                      device const float* b [[buffer(1)]],
                      device float* result [[buffer(2)]],
                      uint index [[thread_position_in_grid]]) {
    result[index] = a[index] + b[index];
}

// Simple scalar multiplication kernel
kernel void scalar_multiply(device const float* input [[buffer(0)]],
                           device float* output [[buffer(1)]],
                           constant float& scalar [[buffer(2)]],
                           uint index [[thread_position_in_grid]]) {
    output[index] = input[index] * scalar;
}

// Simple reduction sum kernel (partial reduction)
kernel void reduce_sum(device const float* input [[buffer(0)]],
                      device float* output [[buffer(1)]],
                      uint index [[thread_position_in_grid]],
                      uint threads_per_group [[threads_per_threadgroup]]) {
    threadgroup float shared_data[256]; // Shared memory for the threadgroup
    
    uint tid = index % threads_per_group;
    uint gid = index / threads_per_group;
    
    // Load data into shared memory
    if (index < 1024) { // Assuming max 1024 elements for simplicity
        shared_data[tid] = input[index];
    } else {
        shared_data[tid] = 0.0f;
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Perform reduction in shared memory
    for (uint stride = threads_per_group / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Write result for this threadgroup
    if (tid == 0) {
        output[gid] = shared_data[0];
    }
}

// Matrix multiplication kernel (simple version)
kernel void matrix_multiply(device const float* A [[buffer(0)]],
                           device const float* B [[buffer(1)]],
                           device float* C [[buffer(2)]],
                           constant uint& N [[buffer(3)]],
                           uint2 gid [[thread_position_in_grid]]) {
    uint row = gid.y;
    uint col = gid.x;
    
    if (row >= N || col >= N) return;
    
    float sum = 0.0f;
    for (uint k = 0; k < N; ++k) {
        sum += A[row * N + k] * B[k * N + col];
    }
    
    C[row * N + col] = sum;
}

// Simple fill kernel
kernel void fill_buffer(device float* output [[buffer(0)]],
                       constant float& value [[buffer(1)]],
                       uint index [[thread_position_in_grid]]) {
    output[index] = value;
} 