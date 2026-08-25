#include "Backend/CUDA/KernelHelper.cuh"
#include "Header.h"
#include "Interactions/Nonbonded/AnalyticalPairKernels.h"
#include "Interactions/Nonbonded/GridGridKernels.h"
#include "Interactions/Nonbonded/Pairwise.h"
#include "Interactions/Nonbonded/Pmf.h"
#include "Interactions/Nonbonded/RigidBodyAttachedParticles.h"
#include "Interactions/Nonbonded/RigidBodyGridBatch.h"
#include "Interactions/Nonbonded/RigidBodyParticleGridBatch.h"
#include "System/PeriodicBox.h"
#include "Types/Types.h"
#include <cuda_runtime.h>

// Explicit template instantiations for Nonbonded interaction kernels.
// These are needed so templates can be instantiated in CUDA compilation units
// instead of C++ files where they would get stub implementations.
namespace MARS {

template Event launch_cuda_kernel(const Resource& resource,
								  const KernelConfig& config,
								  TabulatedNonBondedComputer kernel_func);

// Per-pair table-index resolver, run once per pairlist rebuild (Pairwise.h).
template Event launch_cuda_kernel(const Resource& resource,
								  const KernelConfig& config,
								  ResolvePairTableKernel kernel_func);

// Every analytical nonbonded term shares one kernel, so this is the only
// instantiation needed no matter how many terms are enabled
// (AnalyticalPairKernels.h).
template Event launch_cuda_kernel(const Resource& resource,
								  const KernelConfig& config,
								  AnalyticalPairKernel kernel_func);

// Launched with a trailing argument pack from Patch::calculate_nonbonded_forces
// (only when force/PMF grids are present), so the pack is spelled out here.
template Event launch_cuda_kernel(const Resource& resource,
								  const KernelConfig& config,
								  ComputePMFKernel kernel_func,
								  ParticleView particles,
								  ParticleTypeView types,
								  const BaseGridView<mars_real>* grid_configs,
								  idx_t num_particles);

// Grid-grid RB force kernel is a block-reduction kernel (see GridGridKernels.h),
// so it goes through launch_cuda_kernel_with_workitem instead of launch_cuda_kernel.
template Event launch_cuda_kernel_with_workitem(const Resource& resource,
												const KernelConfig& config,
												ComputeGridGridForceKernel kernel_func,
												const BaseGridView<mars_real> rho,
												const BaseGridView<mars_real> u,
												Vector3* ret_force_energy,
												Vector3* ret_torque);

// Elementwise grid mutation (GridGridKernels.h).
template Event launch_cuda_kernel(const Resource& resource,
								  const KernelConfig& config,
								  ZeroGridKernel<mars_real> kernel_func);

template Event launch_cuda_kernel(const Resource& resource,
								  const KernelConfig& config,
								  ScaleGridKernel<mars_real> kernel_func);

template Event launch_cuda_kernel(const Resource& resource,
								  const KernelConfig& config,
								  ShiftGridKernel<mars_real> kernel_func);

template Event launch_cuda_kernel(const Resource& resource,
								  const KernelConfig& config,
								  MultiplyGridKernel<mars_real> kernel_func);

template Event launch_cuda_kernel(const Resource& resource,
								  const KernelConfig& config,
								  ConvolveGridKernel<mars_real> kernel_func);

} // namespace MARS
