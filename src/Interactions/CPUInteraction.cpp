#include "CPU.h"

void LocalBonded::compute(Patch* p) {
    std::cout << "LocalBonded::compute()" << std::endl;
    InteractionKernels::HarmonicBonds();
};
