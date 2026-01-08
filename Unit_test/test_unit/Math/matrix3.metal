#include <metal_stdlib>
using namespace metal;

#include "Types/Matrix3.h"

using namespace ARBD;

// Element-wise matrix multiplication kernel - matches test expectations
kernel void matrix3_mult_kernel(
    device MetalMatrix3* input_a    [[buffer(3)]],
    device MetalMatrix3* input_b    [[buffer(4)]],
    device MetalMatrix3* output     [[buffer(5)]],
    uint index                      [[thread_position_in_grid]]
) {
    Matrix3_t<float> a = Matrix3_t<float>(input_a[index]);
    Matrix3_t<float> b = Matrix3_t<float>(input_b[index]);
    
    // Element-wise multiplication of matrix components
    Matrix3_t<float> result;
    
    // Multiply each column vector element-wise
    Vector3_t<float> ex_result = Vector3_t<float>(
        a.ex().x * b.ex().x,
        a.ex().y * b.ex().y,
        a.ex().z * b.ex().z
    );
    result.set_ex(ex_result);
    
    Vector3_t<float> ey_result = Vector3_t<float>(
        a.ey().x * b.ey().x,
        a.ey().y * b.ey().y,
        a.ey().z * b.ey().z
    );
    result.set_ey(ey_result);
    
    Vector3_t<float> ez_result = Vector3_t<float>(
        a.ez().x * b.ez().x,
        a.ez().y * b.ez().y,
        a.ez().z * b.ez().z
    );
    result.set_ez(ez_result);
    
    output[index] = MetalMatrix3(result);
}

// Test kernel to verify Matrix3_t Metal compatibility
kernel void test_matrix3_kernel(
    device float* output [[buffer(3)]],
    constant MetalMatrix3& input_matrix [[buffer(4)]],
    constant MetalVector3& input_vector [[buffer(5)]],
    uint tid [[thread_position_in_grid]]) {
    
    // Test basic matrix operations
    if (tid == 0) {
        // Create identity matrix
        Matrix3_t<float> identity;
        
        // Test vector multiplication
        Vector3_t<float> test_vec = Vector3_t<float>(input_vector);
        Vector3_t<float> result = identity * test_vec;
        
        // Test matrix multiplication
        Matrix3_t<float> input_mat = Matrix3_t<float>(input_matrix);
        Matrix3_t<float> scaled = input_mat * 2.0f;
        Matrix3_t<float> combined = scaled * identity;
        
        // Output determinant
        output[0] = combined.det();
        
        // Output transformed vector components
        output[1] = result.x;
        output[2] = result.y; 
        output[3] = result.z;
    }
}