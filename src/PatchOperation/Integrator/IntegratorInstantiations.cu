#include "Backend/CUDA/KernelHelper.cuh"
#include "PatchOperation/Integrator/BD.h"
#include "Types/Types.h"
#include <cuda_runtime.h>

// Explicit template instantiations for Integrator kernels
// These are needed so templates can be instantiated in CUDA compilation units
// instead of C++ files where they would get stub implementations
namespace ARBD {

// BDIntegrate instantiation
template struct BDIntegrate<float>;

// launch_cuda_kernel instantiations
template Event launch_cuda_kernel(const Resource& resource,
								  const KernelConfig& config,
								  BDIntegrate<float> kernel_func);

} // namespace ARBD
