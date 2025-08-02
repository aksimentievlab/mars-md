// Define HOST and DEVICE macros
#ifdef __CUDACC__
#define HOST __host__
#define DEVICE __device__
#else
#define HOST
#define DEVICE
#endif

#define DEVICE_TYPE !defined(__METAL_VERSION__) && !defined(__SYCL_DEVICE_ONLY__) && !defined(__CUDA_ARCH__)