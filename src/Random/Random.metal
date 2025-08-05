#include <metal_stdlib>
#include "../Math/METAL/Vector3.h"
#include "philox.h"
#include "util.h"

using namespace metal;
using namespace ARBD;
using namespace openrand;

// Metal-specific Philox implementation for kernels
struct MetalPhilox {
    uint64_t seed;
    uint32_t counter;
    uint32_t global_seed;
    uint32_t internal_counter;
    
    MetalPhilox(uint64_t s, uint32_t c, uint32_t gs = DEFAULT_GLOBAL_SEED) 
        : seed(s), counter(c), global_seed(gs), internal_counter(0) {}
    
    uint32_t next_uint() {
        Philox rng(seed, counter + internal_counter, global_seed);
        internal_counter++;
        return rng.draw<uint32_t>();
    }
    
    float next_float() {
        return u01<float, uint32_t>(next_uint());
    }
    
    float2 next_float2() {
        uint32_t u1 = next_uint();
        uint32_t u2 = next_uint();
        return float2(u01<float, uint32_t>(u1), u01<float, uint32_t>(u2));
    }
    
    double next_double() {
        uint64_t u = (static_cast<uint64_t>(next_uint()) << 32) | next_uint();
        return u01<double, uint64_t>(u);
    }
};

// Metal-compatible utility functions
inline float metal_int2float(uint32_t i) {
    constexpr float factor = 1.0f / (4294967295.0f + 1.0f);
    constexpr float halffactor = 0.5f * factor;
    return static_cast<float>(i) * factor + halffactor;
}

// Box-Muller transform for Gaussian distribution
float2 box_muller(float u1, float u2) {
    float r = sqrt(-2.0f * log(max(u1, 1e-7f)));
    float theta = 2.0f * M_PI_F * u2;
    return float2(r * cos(theta), r * sin(theta));
}

// Metal kernel functors matching RandomKernels.h structure
struct UniformFunctor {
    float min_val;
    float max_val;
    uint64_t base_seed;
    uint32_t base_ctr;
    uint32_t global_seed;

    void operator()(uint thread_id, device float* output) const {
        MetalPhilox rng(base_seed, base_ctr + thread_id, global_seed);
        uint32_t random_int = rng.next_uint();
        float random_float_01 = metal_int2float(random_int);
        output[thread_id] = min_val + random_float_01 * (max_val - min_val);
    }
};

struct GaussianFunctor {
    float mean;
    float stddev;
    uint output_size;
    uint64_t base_seed;
    uint32_t base_ctr;
    uint32_t global_seed;

    void operator()(uint thread_id, device float* output) const {
        if (thread_id >= output_size) return;

        MetalPhilox rng(base_seed, base_ctr + thread_id, global_seed);
        uint32_t i1 = rng.next_uint();
        uint32_t i2 = rng.next_uint();

        float u1 = (metal_int2float(i1) < 1e-7f) ? 1e-7f : metal_int2float(i1);
        float u2 = (metal_int2float(i2) < 1e-7f) ? 1e-7f : metal_int2float(i2);

        float r = sqrt(-2.0f * log(u1));
        float theta = 2.0f * M_PI_F * u2;
        float gaussian_val = r * cos(theta);

        output[thread_id] = mean + stddev * gaussian_val;
    }
};

struct GaussianVector3Functor {
    Vector3_t<float> mean;
    Vector3_t<float> stddev;
    uint output_size;
    uint64_t base_seed;
    uint32_t base_ctr;
    uint32_t global_seed;

    Vector3_t<float> box_muller_vec(float u1, float u2) const {
        float r = sqrt(-2.0f * log(u1));
        float theta = 2.0f * M_PI_F * u2;
        return Vector3_t<float>(r * cos(theta), r * sin(theta), 0.0f);
    }

    void operator()(uint thread_id, device Vector3_t<float>* output) const {
        if (thread_id >= output_size) return;

        MetalPhilox rng(base_seed, base_ctr + thread_id, global_seed);

        uint32_t i1 = rng.next_uint();
        uint32_t i2 = rng.next_uint();
        uint32_t i3 = rng.next_uint();
        uint32_t i4 = rng.next_uint();

        float u1_x = (metal_int2float(i1) < 1e-7f) ? 1e-7f : metal_int2float(i1);
        float u2_x = (metal_int2float(i2) < 1e-7f) ? 1e-7f : metal_int2float(i2);
        float u1_y = (metal_int2float(i3) < 1e-7f) ? 1e-7f : metal_int2float(i3);
        float u2_y = (metal_int2float(i4) < 1e-7f) ? 1e-7f : metal_int2float(i4);

        Vector3_t<float> gauss_pair1 = box_muller_vec(u1_x, u2_x);
        Vector3_t<float> gauss_pair2 = box_muller_vec(u1_y, u2_y);

        output[thread_id] = Vector3_t<float>(
            mean.x + stddev.x * gauss_pair1.x,
            mean.y + stddev.y * gauss_pair2.y,
            mean.z + stddev.z * gauss_pair2.x
        );
    }
};

// Uniform float generation
kernel void generate_uniform_float(device float* output [[buffer(0)]],
                                 constant uint& count [[buffer(1)]],
                                 constant uint64_t& seed [[buffer(2)]],
                                 constant uint64_t& offset [[buffer(3)]],
                                 constant float& min_val [[buffer(4)]],
                                 constant float& max_val [[buffer(5)]],
                                 uint thread_id [[thread_position_in_grid]]) {
    if (thread_id >= count) return;
    
    MetalPhilox rng(seed, uint32_t(thread_id + offset), 0x12345);
    float u = rng.next_float();
    output[thread_id] = min_val + u * (max_val - min_val);
}

// Gaussian float generation
kernel void generate_gaussian_float(device float* output [[buffer(0)]],
                                  constant uint& count [[buffer(1)]],
                                  constant uint64_t& seed [[buffer(2)]],
                                  constant uint64_t& offset [[buffer(3)]],
                                  constant float& mean [[buffer(4)]],
                                  constant float& stddev [[buffer(5)]],
                                  uint thread_id [[thread_position_in_grid]]) {
    uint base_id = thread_id * 2;
    if (base_id >= count) return;
    
    MetalPhilox rng(seed, uint32_t(thread_id + offset), 0x12345);
    float2 uniform_vals = rng.next_float2();
    float2 gaussian_vals = box_muller(uniform_vals.x, uniform_vals.y);
    
    output[base_id] = mean + stddev * gaussian_vals.x;
    if (base_id + 1 < count) {
        output[base_id + 1] = mean + stddev * gaussian_vals.y;
    }
}


// Gaussian Vector3 generation
kernel void generate_gaussian_vector3(device Vector3_t<float>* output [[buffer(0)]],
                                    constant uint& count [[buffer(1)]],
                                    constant uint64_t& seed [[buffer(2)]],
                                    constant uint64_t& offset [[buffer(3)]],
                                    device const Vector3_t<float>& mean [[buffer(4)]],
                                    device const Vector3_t<float>& stddev [[buffer(5)]],
                                    uint thread_id [[thread_position_in_grid]]) {
    if (thread_id >= count) return;
    
    MetalPhilox rng(seed, uint32_t(thread_id + offset), 0x12345);
    
    // Generate 3 Gaussian values using 2 Box-Muller transforms
    float2 uniform_vals1 = rng.next_float2();
    float2 uniform_vals2 = rng.next_float2();
    
    float2 gaussian_pair1 = box_muller(uniform_vals1.x, uniform_vals1.y);
    float2 gaussian_pair2 = box_muller(uniform_vals2.x, uniform_vals2.y);
    
    output[thread_id] = Vector3_t<float>(
        mean.x + stddev.x * gaussian_pair1.x,
        mean.y + stddev.y * gaussian_pair1.y,
        mean.z + stddev.z * gaussian_pair2.x
    );
}

// Uniform double generation
kernel void generate_uniform_double(device double* output [[buffer(0)]],
                                  constant uint& count [[buffer(1)]],
                                  constant uint64_t& seed [[buffer(2)]],
                                  constant uint64_t& offset [[buffer(3)]],
                                  constant double& min_val [[buffer(4)]],
                                  constant double& max_val [[buffer(5)]],
                                  uint thread_id [[thread_position_in_grid]]) {
    if (thread_id >= count) return;
    
    MetalPhilox rng(seed, uint32_t(thread_id + offset), 0x12345);
    double u = rng.next_double();
    output[thread_id] = min_val + u * (max_val - min_val);
}

// Gaussian double generation
kernel void generate_gaussian_double(device double* output [[buffer(0)]],
                                   constant uint& count [[buffer(1)]],
                                   constant uint64_t& seed [[buffer(2)]],
                                   constant uint64_t& offset [[buffer(3)]],
                                   constant double& mean [[buffer(4)]],
                                   constant double& stddev [[buffer(5)]],
                                   uint thread_id [[thread_position_in_grid]]) {
    uint base_id = thread_id * 2;
    if (base_id >= count) return;
    
    MetalPhilox rng(seed, uint32_t(thread_id + offset), 0x12345);
    float2 uniform_vals = rng.next_float2();
    float2 gaussian_vals = box_muller(uniform_vals.x, uniform_vals.y);
    
    output[base_id] = double(mean) + double(stddev) * double(gaussian_vals.x);
    if (base_id + 1 < count) {
        output[base_id + 1] = double(mean) + double(stddev) * double(gaussian_vals.y);
    }
}

// Uniform int generation
kernel void generate_uniform_int(device int* output [[buffer(0)]],
                               constant uint& count [[buffer(1)]],
                               constant uint64_t& seed [[buffer(2)]],
                               constant uint64_t& offset [[buffer(3)]],
                               constant int& min_val [[buffer(4)]],
                               constant int& max_val [[buffer(5)]],
                               uint thread_id [[thread_position_in_grid]]) {
    if (thread_id >= count) return;
    
    MetalPhilox rng(seed, uint32_t(thread_id + offset), 0x12345);
    uint range = uint(max_val - min_val + 1);
    
    if (range == 0) {
        output[thread_id] = min_val;
        return;
    }
    
    // Avoid modulo bias
    uint limit = UINT_MAX - (UINT_MAX % range);
    uint u32 = rng.next_uint();
    while (u32 >= limit) {
        u32 = rng.next_uint();
    }
    
    output[thread_id] = min_val + int(u32 % range);
}

// Uniform uint generation
kernel void generate_uniform_uint(device uint* output [[buffer(0)]],
                                constant uint& count [[buffer(1)]],
                                constant uint64_t& seed [[buffer(2)]],
                                constant uint64_t& offset [[buffer(3)]],
                                constant uint& min_val [[buffer(4)]],
                                constant uint& max_val [[buffer(5)]],
                                uint thread_id [[thread_position_in_grid]]) {
    if (thread_id >= count) return;
    
    MetalPhilox rng(seed, uint32_t(thread_id + offset), 0x12345);
    uint range = max_val - min_val + 1;
    
    if (range == 0) {
        output[thread_id] = min_val;
        return;
    }
    
    // Avoid modulo bias
    uint limit = UINT_MAX - (UINT_MAX % range);
    uint u32 = rng.next_uint();
    while (u32 >= limit) {
        u32 = rng.next_uint();
    }
    
    output[thread_id] = min_val + (u32 % range);
}