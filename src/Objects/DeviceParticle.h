#pragma once
#include "Header.h"
#include "Types/Types.h"
#include "Types/Vector3.h"
namespace ARBD {
struct ParticleView {
	DEVICE_PTR(int) __restrict__ id;
	DEVICE_PTR(int) __restrict__ type_id;
	DEVICE_PTR(Vector3) __restrict__ pos;
	DEVICE_PTR(Vector3) __restrict__ mom;
	DEVICE_PTR(Vector3) __restrict__ ForceEnergy;
	DEVICE_PTR(Vector3) __restrict__ orient;
	DEVICE_PTR(uint32_t) __restrict__ flags; // Combined flags
};

struct ConstParticleView {
	CONSTANT_PTR(int) __restrict__ id;
	CONSTANT_PTR(int) __restrict__ type_id;
	CONSTANT_PTR(Vector3) __restrict__ pos;
	CONSTANT_PTR(Vector3) __restrict__ mom;
	CONSTANT_PTR(Vector3) __restrict__ ForceEnergy;
	CONSTANT_PTR(Vector3) __restrict__ orient;
	CONSTANT_PTR(uint32_t) __restrict__ flags;
};

struct alignas(16) ParticleTypeView {
	CONSTANT_PTR(float) __restrict__ mass;
	CONSTANT_PTR(float) __restrict__ charge;
	CONSTANT_PTR(float) __restrict__ radius;
	CONSTANT_PTR(float) __restrict__ eps;
	CONSTANT_PTR(Vector3) __restrict__ diffusion;
	CONSTANT_PTR(Vector3) __restrict__ trans_damping;
	CONSTANT_PTR(float) __restrict__ mu;
	CONSTANT_PTR(float) __restrict__ pmf_scale;
	CONSTANT_PTR(float) __restrict__ pmf_scale_slope;
	CONSTANT_PTR(float) __restrict__ pmf_smd_freq;
	CONSTANT_PTR(int) __restrict__ pmf_grid_id;
	CONSTANT_PTR(int) __restrict__ diffusion_grid_id;
	CONSTANT_PTR(int3) __restrict__ force_grid_id;
};
} // namespace ARBD

// SYCL device copyable trait
#ifdef USE_SYCL
#include <sycl/sycl.hpp>
template<>
struct sycl::is_device_copyable<ARBD::ParticleTypeView> : std::true_type {};
template<>
struct sycl::is_device_copyable<ARBD::ParticleView> : std::true_type {};
template<>
struct sycl::is_device_copyable<ARBD::ConstParticleView> : std::true_type {};
#endif
