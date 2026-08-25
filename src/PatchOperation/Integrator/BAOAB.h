#pragma once
#include "../Random/philox.h"
#include "Constants.h"
#include "Header.h"
#include "Interactions/Nonbonded/Pmf.h"
#include "Objects/DeviceParticle.h"
#include "System/PeriodicBox.h"
#include "Types/BaseGrid.h"
#include "Types/IndexList.h"
#include "Types/Types.h"
#include "Types/Vector3.h"

namespace MARS {
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
	const BaseGridView<mars_real>* grid_configs; ///< PMF/force grids (nullptr = none); fused per v1
	Vector3 electric_field;						 ///< Uniform global E field applied here
	int interpolation_scheme;					 ///< 0=linear, 1=cubic
	constexpr static uint32_t rng_stream = 0x1356914u; // Arbitrary stream ID for Philox RNG

	BAOABIntegrate(ParticleView pv,
				   const ParticleTypeView pt,
				   const PeriodicBox& box,
				   float dt,
				   size_t current_step,
				   TemperatureType temp,
				   idx_t n,
				   uint64_t seed,
				   uint32_t ctr,
				   const BaseGridView<mars_real>* grids,
				   const Vector3& efield,
				   int scheme)
		: particle_view(pv), particle_types(pt), sim_box(box), timestep(dt),
		  current_step(current_step), kT(temp), num_particles(n), base_seed(seed), base_ctr(ctr),
		  grid_configs(grids), electric_field(efield), interpolation_scheme(scheme) {}

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

		/**  Position-dependent force (PMF/force grid + uniform E) fused here per v1
		 (evaluated at pos_N; closes step N-1 via BAOAB_LastUpdate and opens step N).
		force += compute_position_dependent_force(pos,
												  type,
												  particle_types,
												  grid_configs,
												  electric_field,
												  interpolation_scheme);
		*/
		// 2. Physical Constants & Properties
		float mass = particle_types.mass[type];

		Vector3 gamma = particle_types.trans_damping[type];

		// --- B: Momentum Update (Half Step) ---
		// p = p + 0.5 * dt * F * Unit1
		mom += mars_real(0.5) * timestep * force * constants::FORCE_CONVERSION_FACTOR;

		// --- A: Position Update (Half Step) ---
		// r = r + 0.5 * dt * (p/m) * 1e4
		// 1e4 accounts for the ns -> internal velocity scaling
		pos += mars_real(0.5) * timestep * mom / mass * mars_real(10000.0);

		// --- O: Ornstein-Uhlenbeck Process (Vectorized) ---
		// Calculate decay factors (c) and noise scales component-wise
		// c = exp(-gamma * dt)
		Vector3 c(expf(-gamma.x * timestep), expf(-gamma.y * timestep), expf(-gamma.z * timestep));

		// noise_scale = sqrt(kT * m * (1 - c^2)) * Unit2
		Vector3 noise_scale(sqrtf(kT * mass * (1.0f - c.x * c.x)) * constants::SQRT_CAL_TO_JOULE,
							sqrtf(kT * mass * (1.0f - c.y * c.y)) * constants::SQRT_CAL_TO_JOULE,
							sqrtf(kT * mass * (1.0f - c.z * c.z)) * constants::SQRT_CAL_TO_JOULE);

		openrand::Philox rng(base_seed,
							 base_ctr + static_cast<uint32_t>(idx),
							 static_cast<uint32_t>(current_step),
							 rng_stream);
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
		pos += mars_real(0.5) * timestep * mom / mass * mars_real(10000.0);

		pos = sim_box.wrap(pos);

		particle_view.pos[idx] = pos;
		particle_view.mom[idx] = mom;
	}
};
template<typename TemperatureType = float>
struct BAOAB_LastUpdate {
	ParticleView particle_view;
	const ParticleTypeView particle_types;
	float timestep;
	TemperatureType kT;
	idx_t num_particles;
	uint64_t base_seed;
	uint32_t base_ctr;
	size_t current_step;
	const BaseGridView<mars_real>* grid_configs; ///< PMF/force grids (nullptr = none); fused per v1
	Vector3 electric_field;						 ///< Uniform global E field applied here
	int interpolation_scheme;					 ///< 0=linear, 1=cubic

	BAOAB_LastUpdate(ParticleView pv,
					 const ParticleTypeView pt,
					 float dt,
					 size_t current_step,
					 TemperatureType temp,
					 idx_t n,
					 uint64_t seed,
					 uint32_t ctr,
					 const BaseGridView<mars_real>* grids,
					 const Vector3& efield,
					 int scheme)
		: particle_view(pv), particle_types(pt), timestep(dt),
		  current_step(current_step), kT(temp), num_particles(n), base_seed(seed), base_ctr(ctr),
		  grid_configs(grids), electric_field(efield), interpolation_scheme(scheme) {}

	KERNEL_FUNC void operator()(idx_t idx) const {
		if (idx >= num_particles)
			return;
		if (particle_view.flags[idx] & ParticleFlags::FLAG_RB_ATTACHED)
			return;

		Vector3 pos = particle_view.pos[idx];
		Vector3 mom = particle_view.mom[idx];
		Vector3 force = particle_view.ForceEnergy[idx];
		int type = particle_view.type_id[idx];

		/**  Closing half-kick: force at pos_N must include the same position-dependent
		// term the opening kernel added (v1 evaluates it in LastUpdateKernelBAOAB too).
		force += compute_position_dependent_force(pos,
												  type,
												  particle_types,
												  grid_configs,
												  electric_field,
												  interpolation_scheme);
		*/
		mom += mars_real(0.5) * timestep * force * constants::FORCE_CONVERSION_FACTOR;
		particle_view.mom[idx] = mom;
	}
};
} // namespace MARS

#ifdef USE_CUDA
namespace MARS {
extern template Event launch_cuda_kernel(const Resource& resource,
										 const KernelConfig& config,
										 BAOABIntegrate<float> kernel_func);
extern template Event launch_cuda_kernel(const Resource& resource,
										 const KernelConfig& config,
										 BAOAB_LastUpdate<float> kernel_func);
} // namespace MARS
#endif
// SYCL device copyable trait
#ifdef USE_SYCL
#include <sycl/sycl.hpp>
template<typename TemperatureType>
struct sycl::is_device_copyable<MARS::BAOABIntegrate<TemperatureType>> : std::true_type {};

template<typename TemperatureType>
struct sycl::is_device_copyable<MARS::BAOAB_LastUpdate<TemperatureType>> : std::true_type {};
#endif
