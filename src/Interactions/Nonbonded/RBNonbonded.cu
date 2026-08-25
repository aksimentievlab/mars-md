#include "Backend/CUDA/KernelHelper.cuh"
#include "Header.h"
#include "Interactions/Nonbonded/GridGridKernels.h"
#include "Interactions/Nonbonded/Pairwise.h"
#include "Interactions/Nonbonded/Pmf.h"
#include "Interactions/Nonbonded/RigidBodyAttachedParticles.h"
#include "Interactions/Nonbonded/RigidBodyGridBatch.h"
#include "Interactions/Nonbonded/RigidBodyParticleGridBatch.h"
#include "System/PeriodicBox.h"
#include "Types/Types.h"
#include <cuda_runtime.h>

namespace MARS {
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

template Event launch_cuda_kernel(const Resource& resource,
								  const KernelConfig& config,
								  RBSyncAttachedPositionsKernel kernel_func);

template Event launch_cuda_kernel_with_workitem(const Resource& resource,
												const KernelConfig& config,
												RBReduceAttachedForcesKernel kernel_func);

} // namespace MARS
