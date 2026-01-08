#pragma once
#include "Header.h"
#include "Types/Types.h"
#include "Types/Vector3.h"

namespace ARBD {
struct ParticleView {
	int* id;
	int* type_id;
	Vector3* pos;
	Vector3* mom;
	Vector3* ForceEnergy;
	Vector3* orient;
	uint32_t* flags; // Combined flags
};

struct ConstParticleView {
	const int* id;
	const int* type_id;
	const Vector3* pos;
	const Vector3* mom;
	const Vector3* ForceEnergy;
	const Vector3* orient;
	const uint32_t* flags;
};

struct alignas(16) ParticleTypeView {
	float* mass;
	float* charge;
	float* radius;
	float* eps;
	Vector3* transDamping;
	float* mu;
	float* pmf_scale;
	float* pmf_scale_slope;
	float* pmf_smd_freq;
	int* pmf_grid_id;
	int* diffusion_grid_id;
	int3* force_grid_id;
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
