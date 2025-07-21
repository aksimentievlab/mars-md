#pragma once
#include "Interaction.h"

class LocalBonded : public LocalInteraction {
public:
    void compute(Patch* patch);
    int num_patches() const { return 1; };
};
