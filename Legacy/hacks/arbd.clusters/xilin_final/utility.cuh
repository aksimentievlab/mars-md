#ifndef UTIL
#define UTIL

#include "Cluster.cuh"

// precision of forces
#if (FLOAT == 1)
typedef float force_t;
#endif
#if (DOUBLE == 1)
typedef double force_t;
#endif

// assert error
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        std::cout<<"CUDA error at "<<file<<":"<<line<<" :"<<cudaGetErrorString(code)<<std::endl;
    }
} 

// check kernel launch success
#define cudaSafeCall(call) \
do \
{ \
    call; \
    cudaError_t err = cudaGetLastError(); \
    if (cudaSuccess != err) \
        std::cout<<"Kernel launch failed: "<<cudaGetErrorString(err)<<std::endl; \
    err = cudaDeviceSynchronize(); \
    if (cudaSuccess != err) \
        std::cout<<"cudaDeviceSynchronize failed: "<<cudaGetErrorString(err)<<std::endl; \
} while (0)

// print running program information
#define INFO(USE_GPU, BOUNDBOX, BOUNDSPHERE, FLOAT, DOUBLE) \
std::cout<<"______________________________________________________"<<std::endl;\
std::cout<<"Number of particles: "<<len<<std::endl;\
std::cout<<"Size of clusters: "<<CLUSTER_SIZE<<std::endl;\
if(USE_GPU)\
    std::cout<<"Run GPU and CPU"<<std::endl;\
else\
    std::cout<<"Run CPU only"<<std::endl;\
if(USE_OMP)\
    std::cout<<"Run openMP for CPU"<<std::endl;\
else\
    std::cout<<"Do not use openMP"<<std::endl;\
if(BOUNDBOX)\
    std::cout<<"Use bounding box to measure cluster distance"<<std::endl;\
if(BOUNDSPHERE)\
    std::cout<<"Use bounding sphere to measure cluster distance"<<std::endl;\
if(FLOAT)\
    std::cout<<"Use single precision floating points for forces"<<std::endl;\
if(DOUBLE)\
    std::cout<<"Use double precision floating points for forces"<<std::endl;\
std::cout<<"______________________________________________________"<<std::endl;\


// timer for benchmark
#define TIMER_START(program) \
gpuErrchk(cudaEventCreate(&start));\
gpuErrchk(cudaEventCreate(&stop));\
gpuErrchk(cudaEventRecord(start, 0));\
std::cout<<"Timer starts: "<<program<<std::endl;\

#define TIMER_END \
gpuErrchk(cudaEventRecord(stop, 0));\
gpuErrchk(cudaEventSynchronize(stop));\
gpuErrchk(cudaEventElapsedTime(&time, start, stop));\
printf("Timer stops:  %3.2f ms\n", time);\

// calculate Lennard-Jones potential 
__host__ __device__
force_t LJP(float r2);

// read position from .restart file
int read_position(const char *file, Point *pos_arr, int len);

// return simulation range as a BoundBox instance
BoundBox simulation_range(Point *pos_arr, int len);



#endif