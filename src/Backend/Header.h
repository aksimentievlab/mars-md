// Define HOST and DEVICE macros
#include <cstddef>
#ifdef __CUDACC__
#define HOST __host__
#define DEVICE __device__
#else
#define HOST
#define DEVICE
#endif

#if !defined(__CUDA_ARCH__) && !defined(__SYCL_DEVICE_ONLY__) && !defined(__METAL_VERSION__)
#define HOST_GUARD
#endif

using idx_t = size_t;
using device_id_t = size_t;
