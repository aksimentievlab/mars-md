#include "Backend/CUDA/KernelHelper.cuh"
#include "PatchOperation/Integrator/BAOAB.h"
#include "PatchOperation/Integrator/BD.h"
#include "PatchOperation/Integrator/RBBD.h"
#include "PatchOperation/Integrator/RBDLM.h"
#include "Types/Types.h"
#include <cuda_runtime.h>

// Explicit template instantiations for Integrator kernels
// These are needed so templates can be instantiated in CUDA compilation units
// instead of C++ files where they would get stub implementations
namespace ARBD {

// BDIntegrate instantiation
template struct BDIntegrate<float>;
template struct BAOABIntegrate<float>;
template struct BAOAB_LastUpdate<float>;
template struct RBLangevinForceKernel<float>;
template struct RBIntegrateBDKernel<float>;
// launch_cuda_kernel instantiations
template Event launch_cuda_kernel(const Resource& resource,
								  const KernelConfig& config,
								  BDIntegrate<float> kernel_func);
template Event launch_cuda_kernel(const Resource& resource,
								  const KernelConfig& config,
								  BAOABIntegrate<float> kernel_func);

template Event launch_cuda_kernel(const Resource& resource,
								  const KernelConfig& config,
								  BAOAB_LastUpdate<float> kernel_func);

template Event launch_cuda_kernel(const Resource& resource,
								  const KernelConfig& config,
								  RBLangevinForceKernel<float> kernel_func);

template Event launch_cuda_kernel(const Resource& resource,
								  const KernelConfig& config,
								  RBIntegrateDLMKernel kernel_func);

template Event launch_cuda_kernel(const Resource& resource,
								  const KernelConfig& config,
								  RBIntegrateBDKernel<float> kernel_func);

} // namespace ARBD
