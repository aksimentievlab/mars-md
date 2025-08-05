#pragma once
#include "Objects/ParticlePatch.h"

#ifdef USE_CUDA
class PatchCPU : public Patch {
public:
    void compute();

        void add_compute(std::unique_ptr<BasePatchOp>&& p) {
	local_computes.emplace_back(std::move(p));
    };

    void add_point_particles(size_t num_added);
    void add_point_particles(size_t num_added, Vector3* positions, Vector3* momenta = nullptr);

};
#endif
