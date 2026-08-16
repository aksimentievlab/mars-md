#pragma once
#include "Header.h"
#include "Types/GridTerm.h"
#include "Types/Types.h"
#include "Types/Vector3.h"
namespace ARBD {

/**
 * @brief Bit flags stored in ParticleView::flags.
 *
 * Defined here rather than in ParticleProperties.h so device-side kernels
 * (integrators in particular) can test them without pulling in the host-side
 * particle/reservoir headers.
 */
enum ParticleFlags : uint32_t {
	FLAG_NONE = 0,
	FLAG_DUMMY = 1 << 0,
	FLAG_ACTIVE = 1 << 1,
	// Rigidly slaved to a parent rigid body: the integrators skip it, and
	// RigidBodyManager rewrites its position from the body's transform every
	// step instead. It still takes part in every force path (pairlist,
	// nonbonded, bonded) like any other particle - the force it accumulates is
	// reduced into the parent body's net force/torque rather than moving it.
	// Set from ParticleIO::attached_rigid_body_id in pack_flags().
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
	CONSTANT_PTR(uint32_t) __restrict__ pmf_smd_freq;
	// A type can reference any number of PMF grids (legacy's `gridFile` takes a
	// list), so the per-type data is an offset+count range into the flat
	// pmf_grid_terms array below - same layout as RigidBodyTypeView's
	// potential/density/pmf grid ranges. Per-grid scale and boundary condition
	// live in the term, not here: legacy keys both per (type, grid) pair, so two
	// types sharing one deduped grid file can weight it differently.
	CONSTANT_PTR(int) __restrict__ pmf_grid_offset;
	CONSTANT_PTR(int) __restrict__ pmf_grid_count;
	CONSTANT_PTR(GridTerm) __restrict__ pmf_grid_terms;
	CONSTANT_PTR(int) __restrict__ diffusion_grid_id;
	CONSTANT_PTR(int3) __restrict__ force_grid_id;
	// Per-type x/y/z scale for force_grid_id's grids (see
	// ParticleType::force_grid_scale for why this is a runtime factor rather
	// than baked into the grid data at load time).
	CONSTANT_PTR(Vector3) __restrict__ force_grid_scale;
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
