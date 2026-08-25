//
//  arbdtest.metal
//  arbdtest
//
//  Created by PinYi on 8/3/25.
//

#include <metal_stdlib>
#include "philox.h"
using namespace metal;


struct UniformFunctor {
    float min_val;
    float max_val;
    uint64_t base_seed;
    uint32_t base_ctr;
    uint32_t global_seed;

    void operator()(uint thread_id, device float4* output) const {
        openrand::Philox rng(base_seed, base_ctr + thread_id, global_seed);
        output[thread_id] = rng.draw_float4();
        
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

        openrand::Philox rng(base_seed, base_ctr + thread_id, global_seed);
        float4 u1 = rng.draw_float4();

        float r = metal::sqrt(-2.0f * log(u1[0]));
        float theta = 2.0f * M_PI_F * u1[1];
        float gaussian_val = r * cos(theta);

        output[thread_id] = mean + stddev * gaussian_val;
    }
};
