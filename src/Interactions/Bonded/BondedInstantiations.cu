#include "Backend/CUDA/KernelHelper.cuh"
#include "Header.h"
#include "Interactions/Bonded/BondComputer.h"
#include "System/PeriodicBox.h"
#include "Types/Types.h"
#include <cuda_runtime.h>

// Only include device-safe parts - avoid host-only code from Interactions.h
// The template structs we're instantiating are device-safe

// Explicit template instantiations for Bonded interaction kernels
// These are needed so templates can be instantiated in CUDA compilation units
// instead of C++ files where they would get stub implementations
namespace ARBD {

// AnalyticalBondComputer instantiations
template struct AnalyticalBondComputer<0>;
template struct AnalyticalBondComputer<1>;

// launch_cuda_kernel instantiations
template Event launch_cuda_kernel(const Resource& resource,
								  const KernelConfig& config,
								  AnalyticalBondComputer<0> kernel_func);

template Event launch_cuda_kernel(const Resource& resource,
								  const KernelConfig& config,
								  AnalyticalBondComputer<1> kernel_func);

} // namespace ARBD
