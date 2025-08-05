#pragma once
#include "../Integrator.h"

#ifdef USE_CUDA

namespace ARBD {

class BDIntegrateCUDA : public Integrator {
public:
    void compute(Patch* patch);
    int num_patches() const { return 1; };
};

} // namespace ARBD

#endif
