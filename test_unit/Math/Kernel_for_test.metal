#include <metal_stdlib>
#include "Math/Vector3.h"
using namespace metal;
using namespace ARBD;

// ============================================================================
// Metal Kernel Functions for Test Suite
// ============================================================================

// Simple transformation kernel: y = 2*x + 1
kernel void transform_kernel(const device float* input [[buffer(0)]],
                           device float* output [[buffer(1)]],
                           uint index [[thread_position_in_grid]]) {
    output[index] = 2.0f * input[index] + 1.0f;
}

// Combine two arrays: 70% uniform + 30% gaussian
kernel void combine_kernel(const device float* uniform [[buffer(0)]],
                          const device float* gaussian [[buffer(1)]],
                          device float* combined [[buffer(2)]],
                          uint index [[thread_position_in_grid]]) {
    combined[index] = 0.7f * uniform[index] + 0.3f * gaussian[index];
}

// Initialize walker positions to origin using ARBD::Vector3_t<float>
kernel void initialize_walkers_kernel(device Vector3_t<float>* positions [[buffer(0)]],
                                    uint index [[thread_position_in_grid]]) {
    positions[index] = Vector3_t<float>(0.0f, 0.0f, 0.0f);
}

// Random walk simulation kernel using ARBD::Vector3_t<float>
kernel void random_walk_kernel(const device Vector3_t<float>* steps [[buffer(0)]],
                              device Vector3_t<float>* positions [[buffer(1)]],
                              uint walker_id [[thread_position_in_grid]]) {
    
    // For simplicity, use hardcoded values or get them from the input size
    // In a real implementation, these would be passed as constants
    const uint num_steps = 100000;
    const uint num_walkers = 1000;
    
    if (walker_id >= num_walkers) return;
    
    Vector3_t<float> pos = positions[walker_id];
    
    // Take NUM_STEPS/NUM_WALKERS steps per walker
    uint steps_per_walker = num_steps / num_walkers;
    uint start_step = walker_id * steps_per_walker;
    
    for (uint step = 0; step < steps_per_walker && (start_step + step) < num_steps; ++step) {
        uint step_idx = start_step + step;
        Vector3_t<float> step_vec = steps[step_idx];
        
        // Normalize step to unit length
        float length = step_vec.length();
        if (length > 0.0f) {
            step_vec = step_vec / length;
        }
        
        // Take the step
        pos = pos + step_vec;
    }
    
    positions[walker_id] = pos;
}

// Calculate distances from origin using ARBD::Vector3_t<float>
kernel void calculate_distances_kernel(const device Vector3_t<float>* positions [[buffer(0)]],
                                     device float* distances [[buffer(1)]],
                                     uint index [[thread_position_in_grid]]) {
    Vector3_t<float> pos = positions[index];
    distances[index] = pos.length();
}

// Simple test kernel
kernel void simple_kernel(const device float* input [[buffer(0)]],
                         device float* output [[buffer(1)]],
                         uint index [[thread_position_in_grid]]) {
    output[index] = float(index);
}

// 3x3 smoothing filter kernel
kernel void smoothing_filter_kernel(const device float* input [[buffer(0)]],
                                   device float* output [[buffer(1)]],
                                   uint index [[thread_position_in_grid]]) {
    
    // For simplicity, use hardcoded grid size
    // In a real implementation, this would be passed as a constant
    const uint grid_size = 256;
    
    uint x = index % grid_size;
    uint y = index / grid_size;
    
    // Simple 3x3 averaging filter
    float sum = 0.0f;
    int count = 0;
    
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            int nx = int(x) + dx;
            int ny = int(y) + dy;
            
            if (nx >= 0 && nx < int(grid_size) && ny >= 0 && ny < int(grid_size)) {
                uint idx = uint(ny) * grid_size + uint(nx);
                sum += input[idx];
                count++;
            }
        }
    }
    
    output[index] = (count > 0) ? sum / float(count) : input[index];
}

// Gradient calculation kernel
kernel void gradient_calculation_kernel(const device float* input [[buffer(0)]],
                                       device float* output [[buffer(1)]],
                                       uint index [[thread_position_in_grid]]) {
    
    // For simplicity, use hardcoded grid size
    // In a real implementation, this would be passed as a constant
    const uint grid_size = 256;
    
    uint x = index % grid_size;
    uint y = index / grid_size;
    
    float grad_x = 0.0f, grad_y = 0.0f;
    
    // Calculate finite difference gradients
    if (x > 0 && x < grid_size - 1) {
        uint left_idx = y * grid_size + (x - 1);
        uint right_idx = y * grid_size + (x + 1);
        grad_x = (input[right_idx] - input[left_idx]) * 0.5f;
    }
    
    if (y > 0 && y < grid_size - 1) {
        uint top_idx = (y - 1) * grid_size + x;
        uint bottom_idx = (y + 1) * grid_size + x;
        grad_y = (input[bottom_idx] - input[top_idx]) * 0.5f;
    }
    
    output[index] = sqrt(grad_x * grad_x + grad_y * grad_y);
}