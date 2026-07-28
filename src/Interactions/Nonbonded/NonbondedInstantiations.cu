#include "Backend/CUDA/KernelHelper.cuh"
#include "Header.h"
#include "Interactions/Nonbonded/Pairwise.h"
#include "Interactions/Nonbonded/PmfKernels.h"
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

} // namespace ARBD
