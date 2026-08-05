#pragma once
#include "../Random/philox.h"
#include "Constants.h"
#include "Header.h"
#include "Objects/DeviceParticle.h"
#include "System/PeriodicBox.h"
#include "Types/BaseGrid.h"
#include "Types/IndexList.h"
#include "Types/Types.h"
#include "Types/Vector3.h"

namespace ARBD {
template<typename TemperatureType = float>
struct BAOABIntegrate {
	ParticleView particle_view;
	const ParticleTypeView particle_types;
	PeriodicBox sim_box;
	float timestep;
	TemperatureType kT;
	idx_t num_particles;
	uint64_t base_seed;
	uint32_t base_ctr;
	size_t current_step;

	BAOABIntegrate(ParticleView pv,
				   const ParticleTypeView pt,
				   const PeriodicBox& box,
				   float dt,
				   size_t current_step,
				   TemperatureType temp,
				   idx_t n,
				   uint64_t seed,
				   uint32_t ctr)
		: particle_view(pv), particle_types(pt), sim_box(box), timestep(dt),
		  current_step(current_step), kT(temp), num_particles(n), base_seed(seed), base_ctr(ctr) {}

	KERNEL_FUNC void operator()(idx_t idx) const {
		if (idx >= num_particles)
			return;
		// Rigid-body attached particles are positioned by their parent body,
		// not integrated - see ParticleFlags::FLAG_RB_ATTACHED.
		if (particle_view.flags[idx] & ParticleFlags::FLAG_RB_ATTACHED)
			return;

		// 1. Load Particle State
		Vector3 pos = particle_view.pos[idx];
		Vector3 mom = particle_view.mom[idx];
		Vector3 force = particle_view.ForceEnergy[idx];
		int type = particle_view.type_id[idx];

		// 2. Physical Constants & Properties
		float mass = particle_types.mass[type];

		// gamma is transDamping directly (legacy: Vector3 gamma = pt.transDamping).
		// This is distinct from `diffusion`, which only feeds the plain
		// Brownian integrator (BD.h) - the two are independent inputs here,
		// matching legacy BrownianParticleType's separate fields. Legacy only
		// derives gamma from diffusion via the Einstein relation when a
		// per-particle diffusionGrid is set (position-dependent D); that
		// override, and the analogous position-dependent kT (kTGrid) case,
		// are unimplemented here - TODO: a separate BAOAB_TempGrid kernel
		// variant, not a branch in this hot path.
		Vector3 gamma = particle_types.trans_damping[type];

		// --- B: Momentum Update (Half Step) ---
		// p = p + 0.5 * dt * F * Unit1
		mom += 0.5f * timestep * force * constants::FORCE_CONVERSION_FACTOR;

		// --- A: Position Update (Half Step) ---
		// r = r + 0.5 * dt * (p/m) * 1e4
		// 1e4 accounts for the ns -> internal velocity scaling
		pos += 0.5f * timestep * mom / mass * 10000.0f;

		// --- O: Ornstein-Uhlenbeck Process (Vectorized) ---
		// Calculate decay factors (c) and noise scales component-wise
		// c = exp(-gamma * dt)
		Vector3 c(expf(-gamma.x * timestep), expf(-gamma.y * timestep), expf(-gamma.z * timestep));

		// noise_scale = sqrt(kT * m * (1 - c^2)) * Unit2
		Vector3 noise_scale(sqrtf(kT * mass * (1.0f - c.x * c.x)) * constants::SQRT_CAL_TO_JOULE,
							sqrtf(kT * mass * (1.0f - c.y * c.y)) * constants::SQRT_CAL_TO_JOULE,
							sqrtf(kT * mass * (1.0f - c.z * c.z)) * constants::SQRT_CAL_TO_JOULE);

		// Generate Random Numbers (Box-Muller)
		openrand::Philox rng(base_seed, base_ctr + static_cast<uint32_t>(idx));
		openrand::float4 uniform = rng.draw_float4();

		float r1 = sqrtf(-2.0f * logf(uniform.x));
		float theta1 = 2.0f * 3.1415926535f * uniform.y;
		float r2 = sqrtf(-2.0f * logf(uniform.z));
		float theta2 = 2.0f * 3.1415926535f * uniform.w;

		Vector3 random_force(r1 * cosf(theta1), r1 * sinf(theta1), r2 * cosf(theta2));

		// Update Momentum
		mom.x = c.x * mom.x + noise_scale.x * random_force.x;
		mom.y = c.y * mom.y + noise_scale.y * random_force.y;
		mom.z = c.z * mom.z + noise_scale.z * random_force.z;

		// --- A: Position Update (Second Half Step) ---
		// r = r + 0.5 * dt * (p/m) * 1e4
		// Added 1e4 to match Old Kernel line: r0 = r0 + 0.5f * timestep * p0 * 1e4 / mass;
		pos += 0.5f * timestep * mom / mass * 10000.0f;

		// Apply Periodic Boundary Conditions.
		// Matches legacy "r0 = sys->wrap(r0)": wrapping is relative to the box
		// origin, so a box centered on zero stays centered instead of being
		// shifted into [0, L).
		pos = sim_box.wrap(pos);

		// Write Back
		particle_view.pos[idx] = pos;
		particle_view.mom[idx] = mom;
	}
};
template<typename TemperatureType = float>
struct BAOAB_LastUpdate {
	ParticleView particle_view;
	const ParticleTypeView particle_types;
	Vector3 box_size;
	float timestep;
	TemperatureType kT;
	idx_t num_particles;
	uint64_t base_seed;
	uint32_t base_ctr;
	size_t current_step;

	BAOAB_LastUpdate(ParticleView pv,
					 const ParticleTypeView pt,
					 const Vector3& box,
					 float dt,
					 size_t current_step,
					 TemperatureType temp,
					 idx_t n,
					 uint64_t seed,
					 uint32_t ctr)
		: particle_view(pv), particle_types(pt), box_size(box), timestep(dt),
		  current_step(current_step), kT(temp), num_particles(n), base_seed(seed), base_ctr(ctr) {}

	KERNEL_FUNC void operator()(idx_t idx) const {
		if (idx >= num_particles)
			return;
		// Rigid-body attached particles are positioned by their parent body,
		// not integrated - see ParticleFlags::FLAG_RB_ATTACHED.
		if (particle_view.flags[idx] & ParticleFlags::FLAG_RB_ATTACHED)
			return;

		Vector3 pos = particle_view.pos[idx];
		Vector3 mom = particle_view.mom[idx];
		Vector3 force = particle_view.ForceEnergy[idx];
		int type = particle_view.type_id[idx];

		mom += 0.5f * timestep * force * constants::FORCE_CONVERSION_FACTOR;
		particle_view.mom[idx] = mom;
	}
};
} // namespace ARBD

#ifdef USE_CUDA
namespace ARBD {
extern template Event launch_cuda_kernel(const Resource& resource,
										 const KernelConfig& config,
										 BAOABIntegrate<float> kernel_func);
extern template Event launch_cuda_kernel(const Resource& resource,
										 const KernelConfig& config,
										 BAOAB_LastUpdate<float> kernel_func);
} // namespace ARBD
#endif
// SYCL device copyable trait
#ifdef USE_SYCL
#include <sycl/sycl.hpp>
template<typename TemperatureType>
struct sycl::is_device_copyable<ARBD::BAOABIntegrate<TemperatureType>> : std::true_type {};

template<typename TemperatureType>
struct sycl::is_device_copyable<ARBD::BAOAB_LastUpdate<TemperatureType>> : std::true_type {};
#endif
