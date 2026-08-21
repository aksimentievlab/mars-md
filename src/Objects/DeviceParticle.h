#pragma once
#include "Header.h"
#include "Types/GridTerm.h"
#include "Types/Types.h"
#include "Types/Vector3.h"
namespace ARBD {

/**
 * @brief Bit flags stored in ParticleView::flags.
 */
enum ParticleFlags : uint32_t {
	FLAG_NONE = 0,
	FLAG_DUMMY = 1 << 0,
	FLAG_ACTIVE = 1 << 1,
	FLAG_RB_ATTACHED = 1 << 2,
};

struct ParticleView {
	DEVICE_PTR(int) __restrict__ id;
	DEVICE_PTR(int) __restrict__ type_id;
	DEVICE_PTR(Vector3) __restrict__ pos;
	DEVICE_PTR(Vector3) __restrict__ mom;
	DEVICE_PTR(Vector3) __restrict__ ForceEnergy;
	DEVICE_PTR(Vector3) __restrict__ orient;
	DEVICE_PTR(uint32_t) __restrict__ flags; // Combined flags
	DEVICE_PTR(Vector3) __restrict__ external_force; // Per-particle diffusion vector (x/y/z
};

struct ConstParticleView {
	CONSTANT_PTR(int) __restrict__ id;
	CONSTANT_PTR(int) __restrict__ type_id;
	CONSTANT_PTR(Vector3) __restrict__ pos;
	CONSTANT_PTR(Vector3) __restrict__ mom;
	CONSTANT_PTR(Vector3) __restrict__ ForceEnergy;
	CONSTANT_PTR(Vector3) __restrict__ orient;
	CONSTANT_PTR(uint32_t) __restrict__ flags;
	CONSTANT_PTR(Vector3) __restrict__ external_force;
};
/**
 * @param pmf_grid_offset Index of first PMF grid for this type in pmf_grid_terms
 * @param pmf_grid_count Number of PMF grids for this type
 * @param pmf_grid_terms Flat array of all PMF grids for all types, with per-type ranges defined by
 * pmf_grid_offset/count
 * @param diffusion_grid_id Index of the diffusion grid for this type in the global grid list (or -1
 * if none)
 */
struct alignas(16) ParticleTypeView {
	CONSTANT_PTR(float) __restrict__ mass;
	CONSTANT_PTR(float) __restrict__ charge;
	CONSTANT_PTR(float) __restrict__ radius;
	CONSTANT_PTR(float) __restrict__ eps;
	CONSTANT_PTR(Vector3) __restrict__ diffusion;
	CONSTANT_PTR(Vector3) __restrict__ trans_damping;
	CONSTANT_PTR(float) __restrict__ mu;
	CONSTANT_PTR(uint32_t) __restrict__ pmf_smd_freq;
	CONSTANT_PTR(int) __restrict__ pmf_grid_offset;
	CONSTANT_PTR(int) __restrict__ pmf_grid_count;
	CONSTANT_PTR(GridTerm) __restrict__ pmf_grid_terms;
	CONSTANT_PTR(int) __restrict__ diffusion_grid_id;
	CONSTANT_PTR(int3) __restrict__ force_grid_id;
	// Per-type x/y/z scale for force_grid_id's grids
	CONSTANT_PTR(Vector3) __restrict__ force_grid_scale;
};
} // namespace ARBD

#ifdef USE_SYCL
#include <sycl/sycl.hpp>
template<>
struct sycl::is_device_copyable<ARBD::ParticleTypeView> : std::true_type {};
template<>
struct sycl::is_device_copyable<ARBD::ParticleView> : std::true_type {};
template<>
struct sycl::is_device_copyable<ARBD::ConstParticleView> : std::true_type {};
#endif
