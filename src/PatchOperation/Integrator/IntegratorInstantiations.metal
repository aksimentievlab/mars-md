#include <metal_stdlib>
#include BAOAB.h
#include <metal_stdlib>
using namespace metal;

// 1. Set up macros for Metal
//#define KERNEL_FUNC inline
//#define DEVICE_PTR(type) device type*

#include "BAOAB.h"

namespace MARS{
// 3. Wrapper for BAOABIntegrate<float>
kernel void baoab_integrate_kernel(
    // Auto-bound grid dimensions (Indices 0, 1, 2)
    constant uint32_t& grid_width  [[buffer(0)]],
    constant uint32_t& grid_height [[buffer(1)]],
    constant uint32_t& grid_depth  [[buffer(2)]],

    // User arguments (Starts at Index 3)
    // Passed by value/reference via setBytes() in C++
    constant ParticleView& pv                [[buffer(3)]],
    constant ParticleTypeView& pt            [[buffer(4)]],
    constant PeriodicBox& sim_box            [[buffer(5)]],
    constant float& timestep                 [[buffer(6)]],
    constant size_t& current_step            [[buffer(7)]],
    constant float& temp                     [[buffer(8)]],
    constant idx_t& num_particles            [[buffer(9)]],
    constant uint64_t& base_seed             [[buffer(10)]],
    constant uint32_t& base_ctr              [[buffer(11)]],

    // Metal Thread ID
    uint thread_id [[thread_position_in_grid]]
) {
    if (thread_id >= num_particles) return;

    // Instantiate the functor (forces TemperatureType = float)
    BAOABIntegrate<float> integrator(
        pv, pt, sim_box, timestep, current_step,
        temp, num_particles, base_seed, base_ctr
    );

    // Run the integration step
    integrator(thread_id);
}

// 4. Wrapper for BAOAB_LastUpdate<float>
kernel void baoab_last_update_kernel(
    constant uint32_t& grid_width  [[buffer(0)]],
    constant uint32_t& grid_height [[buffer(1)]],
    constant uint32_t& grid_depth  [[buffer(2)]],

    constant ParticleView& pv                [[buffer(3)]],
    constant ParticleTypeView& pt            [[buffer(4)]],
    constant Vector3& box_size               [[buffer(5)]],
    constant float& timestep                       [[buffer(6)]],
    constant size_t& current_step                  [[buffer(7)]],
    constant float& temp                           [[buffer(8)]],
    constant idx_t& num_particles            [[buffer(9)]],
    constant uint64_t& base_seed                   [[buffer(10)]],
    constant uint32_t& base_ctr                    [[buffer(11)]],

    uint thread_id [[thread_position_in_grid]]){
    if (thread_id >= num_particles) return;

    // Instantiate the functor
    BAOAB_LastUpdate<float> updater(
        pv, pt, box_size, timestep, current_step,
        temp, num_particles, base_seed, base_ctr
    );

    // Run the update step
    updater(thread_id);
}
}
// You would add similar wrappers for BDIntegrate, RBLangevinForceKernel, etc.
