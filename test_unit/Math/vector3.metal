/*********************************************************************
 * @file  vector3_metal_kernels.metal
 *
 * @brief Metal compute shaders demonstrating Vector3_t usage
 *********************************************************************/
#include <metal_stdlib>
using namespace metal;

#include "Types/Vector3.h"

using namespace ARBD;

// Simple debug kernel to test basic execution
kernel void debug_kernel(
    device float* output     [[buffer(0)]],
    uint index                      [[thread_position_in_grid]]
) {
    // Write a simple constant to test if kernel executes
    output[index] = 42.0f;
}

// Basic vector operations kernel - matches test expectations
kernel void vector_operations_kernel(
    device MetalVector3* input_a    [[buffer(3)]],
    device MetalVector3* input_b    [[buffer(4)]],
    device float* input_c         [[buffer(5)]],
    device MetalVector3* output     [[buffer(6)]],
    uint index                      [[thread_position_in_grid]]
) {
    Vector3_t<float> a = Vector3_t<float>(input_a[index].x, input_a[index].y, input_a[index].z);
    Vector3_t<float> b = Vector3_t<float>(input_b[index].x, input_b[index].y, input_b[index].z);
    
    // Perform vector addition and multiply by 2.0f
    Vector3_t<float> result = (a + b) * input_c[0];
    
    // Calculate dot product for w component
    float dot_product = a.dot(b);
    
    output[index] = MetalVector3(result.x, result.y, result.z, dot_product);
}

// Cross product kernel - matches test expectations
kernel void cross_product_kernel(
    device MetalVector3* input_a    [[buffer(3)]],
    device MetalVector3* input_b    [[buffer(4)]],
    device MetalVector3* output     [[buffer(5)]],
    uint index                      [[thread_position_in_grid]]
) {
    Vector3_t<float> a = Vector3_t<float>(input_a[index].x, input_a[index].y, input_a[index].z);
    Vector3_t<float> b = Vector3_t<float>(input_b[index].x, input_b[index].y, input_b[index].z);
    
    Vector3_t<float> cross_result = a.cross(b);
    
    output[index] = MetalVector3(cross_result);
}

// Length and normalization kernel - matches test expectations
kernel void length_operations_kernel(
    device MetalVector3* input      [[buffer(3)]],
    device MetalVector3* output     [[buffer(4)]],
    device float* lengths           [[buffer(5)]],
    uint index                      [[thread_position_in_grid]]
) {
    Vector3_t<float> v = Vector3_t<float>(input[index].x, input[index].y, input[index].z);
    
    // Calculate length
    float len = v.length();
    lengths[index] = len;
    
    // Normalize vector (store normalized version)
    if (len > 0.0f) {
        float inv_len = 1.0f / len;  // Use simple division instead of rLength()
        output[index] = MetalVector3(v * inv_len);
    } else {
        output[index] = MetalVector3(0.0f, 0.0f, 0.0f);
    }
}

// Element-wise operations kernel
kernel void element_operations_kernel(
    device MetalVector3* input_a    [[buffer(3)]],
    device MetalVector3* input_b    [[buffer(4)]],
    device MetalVector3* output     [[buffer(5)]],
    uint index                      [[thread_position_in_grid]]
) {
    Vector3_t<float> a = Vector3_t<float>(input_a[index].x, input_a[index].y, input_a[index].z);
    Vector3_t<float> b = Vector3_t<float>(input_b[index].x, input_b[index].y, input_b[index].z);
    
    // Element-wise multiplication
    Vector3_t<float> result = a.element_mult(b);
    
    // Test element_floor on result
    result = result.element_floor();
    
    output[index] = MetalVector3(result);
}

// Physics simulation kernel example
kernel void particle_update_kernel(
    device MetalVector3* positions  [[buffer(3)]],
    device MetalVector3* velocities [[buffer(4)]],
    device MetalVector3* forces     [[buffer(5)]],
    constant float& dt              [[buffer(6)]],
    constant float& mass            [[buffer(7)]],
    uint index                      [[thread_position_in_grid]]
) {
    Vector3_t<float> pos = Vector3_t<float>(positions[index].x, positions[index].y, positions[index].z);
    Vector3_t<float> vel = Vector3_t<float>(velocities[index].x, velocities[index].y, velocities[index].z);
    Vector3_t<float> force = Vector3_t<float>(forces[index].x, forces[index].y, forces[index].z);
    
    // Simple Euler integration
    Vector3_t<float> acceleration = force / mass;
    vel = vel + acceleration * dt;
    pos = pos + vel * dt;
    
    // Update buffers
    positions[index] = MetalVector3(pos);
    velocities[index] = MetalVector3(vel);
}

// Field gradient kernel example
kernel void field_gradient_kernel(
    device MetalVector3* field      [[buffer(3)]],
    device MetalVector3* gradient   [[buffer(4)]],
    constant float& dx              [[buffer(5)]],
    constant float& dy              [[buffer(6)]],
    constant float& dz              [[buffer(7)]],
    uint index                      [[thread_position_in_grid]]
) {
    Vector3_t<float> center = Vector3_t<float>(field[index].x, field[index].y, field[index].z);
    
    // Simple finite difference gradient (simplified)
    // In a real implementation, you'd need to access neighboring elements
    Vector3_t<float> grad(0.0f, 0.0f, 0.0f);
    
    // For demonstration, just use the field value as gradient
    grad = center * 0.1f; // Simplified gradient calculation
    
    gradient[index] = MetalVector3(grad);
}