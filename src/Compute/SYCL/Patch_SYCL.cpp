#include "CPU.h"

void BDIntegrate::compute(Patch* p) {
    std::cout << "BDIntegrate::compute()" << std::endl;
    IntegratorKernels::BDIntegrate();
};
