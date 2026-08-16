All Device Kernels here.

Calling Metal kernel example from HOST from Gemini:
// C++ Side Launch
Event event = launch_metal_kernel(
    metal_res,
    num_particles,
    config,
    "baoab_integrate_kernel", // The exact name of the metal kernel above

    // Arguments in exact order of constructor
    particle_view,
    particle_types,
    sim_box,
    timestep,
    current_step,
    kT,
    num_particles,
    base_seed,
    base_ctr
);
