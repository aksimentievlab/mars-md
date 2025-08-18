// Define HOST and DEVICE macros
#include <cstddef>
#ifdef __CUDACC__
#define HOST __host__
#define DEVICE __device__
#else
#define HOST
#define DEVICE
#endif
using idx_t = size_t;