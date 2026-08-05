#include "Backend/CUDA/KernelHelper.cuh"
#include "Header.h"
#include "Interactions/Nonbonded/GridGridKernels.h"
#include "Interactions/Nonbonded/Pairwise.h"
#include "Interactions/Nonbonded/PmfKernels.h"
#include "Interactions/Nonbonded/RigidBodyAttachedParticles.h"
#include "Interactions/Nonbonded/RigidBodyGridBatch.h"
#include "Interactions/Nonbonded/RigidBodyParticleGridBatch.h"
#include "System/PeriodicBox.h"
#include "Types/Types.h"
#include <cuda_runtime.h>

// Explicit template instantiations for Nonbonded interaction kernels.
// These are needed so templates can be instantiated in CUDA compilation units
// instead of C++ files where they would get stub implementations.
namespace ARBD {

template Event launch_cuda_kernel(const Resource& resource,
								  const KernelConfig& config,
								  TabulatedNonBondedComputer kernel_func);

// Launched with a trailing argument pack from Patch::calculate_nonbonded_forces
// (only when force/PMF grids are present), so the pack is spelled out here.
template Event launch_cuda_kernel(const Resource& resource,
								  const KernelConfig& config,
								  ComputePMFKernel kernel_func,
								  Vector3* positions,
								  int* type_ids,
								  Vector3* forces,
								  ParticleTypeView types,
								  const BaseGridView<float>* grid_configs,
								  idx_t num_particles);

// Grid-grid RB force kernel is a block-reduction kernel (see GridGridKernels.h),
// so it goes through launch_cuda_kernel_with_workitem instead of launch_cuda_kernel.
template Event launch_cuda_kernel_with_workitem(const Resource& resource,
												const KernelConfig& config,
												ComputeGridGridForceKernel kernel_func,
												const BaseGridView<float> rho,
												const BaseGridView<float> u,
												Vector3* ret_force_energy,
												Vector3* ret_torque);

// Phase 4.1 batched RB grid-grid dispatch (RigidBodyGridBatch.h): cull +
// worklist build, prefix sum, and the batched block-reduction force kernel.
template Event launch_cuda_kernel(const Resource& resource,
								  const KernelConfig& config,
								  RBGridCullKernel kernel_func);

template Event launch_cuda_kernel(const Resource& resource,
								  const KernelConfig& config,
								  RBGridPrefixSumKernel kernel_func);

template Event launch_cuda_kernel_with_workitem(const Resource& resource,
												const KernelConfig& config,
												RBGridBatchedForceKernel kernel_func);

// Phase 4.3 batched particle-RB grid dispatch (RigidBodyParticleGridBatch.h):
// transform build + the batched block-reduction force kernel.
template Event launch_cuda_kernel(const Resource& resource,
								  const KernelConfig& config,
								  RBParticleGridBuildKernel kernel_func);

template Event launch_cuda_kernel_with_workitem(const Resource& resource,
												const KernelConfig& config,
												RBParticleGridForceKernel kernel_func);

// Attached particles (RigidBodyAttachedParticles.h): per-step position slaving
// and the block-reduction of their forces back onto the parent body.
template Event launch_cuda_kernel(const Resource& resource,
								  const KernelConfig& config,
								  RBSyncAttachedPositionsKernel kernel_func);

template Event launch_cuda_kernel_with_workitem(const Resource& resource,
												const KernelConfig& config,
												RBReduceAttachedForcesKernel kernel_func);

} // namespace ARBD
