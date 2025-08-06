#pragma once
#ifdef USE_CUDA
#include "Backend/CUDA/CUDAManager.h"
#include "Types/Types.h"
#include "../ParticlePatch.h"

namespace ARBD {

class PatchCUDA : public Patch {
public:
    
    void compute();
    // void assign_gpu(GPU);
    
private:

    GPU my_gpu;
    
    // Device arrays
    size_t* groupSiteData_d;
    //Vector3* pos_force_d;
    //Vector3* momentum_d;
    ARBD::CUDA::DeviceMemory<Vector3> pos_force_d;
    ARBD::CUDA::DeviceMemory<Vector3> momentum_d;
    ARBD::CUDA::DeviceMemory<Vector3> rb_pos_d;
    ARBD::CUDA::DeviceMemory<Matrix3> rb_orient_d;
    ARBD::CUDA::DeviceMemory<Vector3> rb_mom_d;
    ARBD::CUDA::DeviceMemory<Vector3> rb_ang_mom_d;
    size_t* type_d;
    size_t* rb_type_d;

};

} // namespace ARBD
#endif