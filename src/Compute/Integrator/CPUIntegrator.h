#pragma once
#include "../Integrator.h"

class BDIntegrate : public Integrator {
public:
    void compute(Patch* patch);
    int num_patches() const { return 1; };
};

class LangevinIntegrate : public Integrator {
public:
    void compute(Patch* patch);
    int num_patches() const { return 1; };
};
