#include <metal_stdlib>
#include "Types/Vector3.h"
using namespace metal;
using namespace MARS;

// ============================================================================
// Metal Kernel Functions for Test Suite
// ============================================================================

// Simple transformation kernel: y = 2*x + 1
kernel void transform_kernel(const device float* input [[buffer(3)]],
                           device float* output [[buffer(4)]],
                           uint index [[thread_position_in_grid]]) {
    output[index] = 2.0f * input[index] + 1.0f;
}

// Combine two arrays: 70% uniform + 30% gaussian
kernel void combine_kernel(const device float* uniform [[buffer(3)]],
                          const device float* gaussian [[buffer(4)]],
                          device float* combined [[buffer(5)]],
                          uint index [[thread_position_in_grid]]) {
    combined[index] = 0.7f * uniform[index] + 0.3f * gaussian[index];
}

// Initialize walker positions to origin using MARS::Vector3_t<float>
kernel void initialize_walkers_kernel(device Vector3_t<float>* positions [[buffer(3)]],
                                    uint index [[thread_position_in_grid]]) {
    positions[index] = Vector3_t<float>(0.0f, 0.0f, 0.0f);
}

// Random walk simulation kernel using MARS::Vector3_t<float>
kernel void random_walk_kernel(const device Vector3_t<float>* steps [[buffer(3)]],
                              device Vector3_t<float>* positions [[buffer(4)]],
                              constant uint& grid_width [[buffer(0)]],
                              constant uint& grid_height [[buffer(1)]],
                              uint walker_id [[thread_position_in_grid]]) {
    
    // Use hardcoded values for now since we can't pass them as parameters
    uint num_walkers = 1000;  // NUM_WALKERS
    uint num_steps = 100000;  // NUM_STEPS
    
    if (walker_id >= num_walkers) return;
    

    
    Vector3_t<float> pos = positions[walker_id];
    
    // Take NUM_STEPS/NUM_WALKERS steps per walker
    uint steps_per_walker = num_steps / num_walkers;
    uint start_step = walker_id * steps_per_walker;
    
    for (uint step = 0; step < steps_per_walker && (start_step + step) < num_steps; ++step) {
        uint step_idx = start_step + step;
        Vector3_t<float> step_vec = steps[step_idx];
        
        // Take the step (no normalization - use the full Gaussian-distributed step)
        pos = pos + step_vec;
    }
    
    positions[walker_id] = pos;
}

// Calculate distances from origin using MARS::Vector3_t<float>
kernel void calculate_distances_kernel(const device Vector3_t<float>* positions [[buffer(3)]],
                                     device float* distances [[buffer(4)]],
                                     uint index [[thread_position_in_grid]]) {
    Vector3_t<float> pos = positions[index];
    distances[index] = pos.length();
}

// Simple test kernel
kernel void simple_kernel(const device float* input [[buffer(3)]],
                         device float* output [[buffer(4)]],
                         uint index [[thread_position_in_grid]]) {
    output[index] = float(index);
}

// 3x3 smoothing filter kernel
kernel void smoothing_filter_kernel(const device float* input [[buffer(3)]],
                                   device float* output [[buffer(4)]],
                                   uint index [[thread_position_in_grid]]) {
    
    // For now, use a reasonable default grid size
    // In the future, we could access grid dimensions from the automatically bound buffers
    uint grid_width = 256;
    uint grid_height = 256;
    
    uint x = index % grid_width;
    uint y = index / grid_width;
    
    // Simple 3x3 averaging filter
    float sum = 0.0f;
    int count = 0;
    
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            int nx = int(x) + dx;
            int ny = int(y) + dy;
            
            if (nx >= 0 && nx < int(grid_width) && ny >= 0 && ny < int(grid_height)) {
                uint idx = uint(ny) * grid_width + uint(nx);
                sum += input[idx];
                count++;
            }
        }
    }
    
    output[index] = (count > 0) ? sum / float(count) : input[index];
}

// Gradient calculation kernel
kernel void gradient_calculation_kernel(const device float* input [[buffer(3)]],
                                       device float* output [[buffer(4)]],
                                       constant uint& grid_width [[buffer(0)]],
                                       constant uint& grid_height [[buffer(1)]],
                                       uint index [[thread_position_in_grid]]) {
    
    // Use actual grid dimensions from kernel config
    uint x = index % grid_width;
    uint y = index / grid_height;
    
    float grad_x = 0.0f, grad_y = 0.0f;
    
    // Calculate finite difference gradients
    if (x > 0 && x < grid_width - 1) {
        uint left_idx = y * grid_width + (x - 1);
        uint right_idx = y * grid_width + (x + 1);
        grad_x = (input[right_idx] - input[left_idx]) * 0.5f;
    }
    
    if (y > 0 && y < grid_height - 1) {
        uint top_idx = (y - 1) * grid_width + x;
        uint bottom_idx = (y + 1) * grid_width + x;
        grad_y = (input[bottom_idx] - input[top_idx]) * 0.5f;
    }
    
    output[index] = sqrt(grad_x * grad_x + grad_y * grad_y);
}

// Circle test kernel for Monte Carlo π estimation
kernel void circle_test_kernel(const device float* x_coords [[buffer(3)]],
                               const device float* y_coords [[buffer(4)]],
                               device int* inside_circle [[buffer(5)]],
                               uint index [[thread_position_in_grid]]) {
    
    float x = x_coords[index];
    float y = y_coords[index];
    
    // Check if point is inside unit circle
    float dist_sq = x * x + y * y;
    inside_circle[index] = (dist_sq <= 1.0f) ? 1 : 0;
}