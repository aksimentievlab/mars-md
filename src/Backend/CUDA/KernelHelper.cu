#include "Header.h"
#include "KernelHelper.cuh"
#include "MemoryKernels.cuh"
#include "Types/Types.h"
#include <cuda_runtime.h>

// Explicit template instantiations for fill_impl to support CUDA separable compilation
// These are needed because CUDA uses separable compilation (-rdc=true)
namespace ARBD {
namespace CUDA {
template void fill_impl<float>(void* dst, float value, size_t num_elements, void* queue, bool sync);
template void
fill_impl<uint32_t>(void* dst, uint32_t value, size_t num_elements, void* queue, bool sync);
template void fill_impl<int>(void* dst, int value, size_t num_elements, void* queue, bool sync);
template void
fill_impl<double>(void* dst, double value, size_t num_elements, void* queue, bool sync);
template void fill_impl<Vector3_t<float>>(void* dst,
										  Vector3_t<float> value,
										  size_t num_elements,
										  void* queue,
										  bool sync);
} // namespace CUDA
} // namespace ARBD
