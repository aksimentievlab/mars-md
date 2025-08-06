#pragma once

class ComputeForce;
class NonbondedCompute {
public:
    virtual void decompose(const ComputeForce &compute) = 0;
    virtual float computeTabulated(bool get_energy, const ComputeForce &compute) = 0;
protected:
    static GPUManager gpuman;
};

#include "VerletCompute.h"
#include "ClusterCompute.h"
