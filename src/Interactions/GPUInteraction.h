#pragma once
#include "Interaction.h"

#ifdef USE_CUDA
class LocalBondedCUDA : public LocalInteraction {
public:
    void compute(Patch* patch);
    int num_patches() const { return 1; };
};
#endif
