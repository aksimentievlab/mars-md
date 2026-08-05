#include "Backend/CUDA/KernelHelper.cuh"
#include "Objects/DeviceRigidBody.h"
#include "Objects/RigidBodyCosmeticsKernel.h"

namespace ARBD {

template Event launch_cuda_kernel(const Resource& resource,
								  const KernelConfig& config,
								  RBCosmeticPositionsKernel kernel_func);

} // namespace ARBD
