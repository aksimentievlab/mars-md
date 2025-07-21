#pragma once
#include "PatchOp.h"

struct ParticleType {
    float diffusion = 1;
};
struct SimParameters {
    float timestep = 1;
    float kT = 1;
};

struct Random {
    __host__ __device__ Vector3 gaussian_vector() {
	return Vector3(0.0f);
    }
    int seed = 1;
};


namespace LocalOneParticleLoop {

    namespace BD {
	template<typename ...Op_t>
	__global__ void op_kernel(size_t num, Vector3* __restrict__ pos_force, size_t* type_ids, ParticleType* types, SimParameters& sim, Random& random) {
	    for (size_t i = threadIdx.x + blockIdx.x*blockDim.x; i < num; i+= gridDim.x*blockDim.x) {
		Vector3& pos = pos_force[i];
		Vector3& force = pos_force[num+i];
		ParticleType& t = types[type_ids[i]];
		
		using expander = int[];
		(void)expander{0, (void(Op_t::op(pos, force, t, sim, random)))...};
	    }
	}
    }
    
    namespace MD {
	template<typename ...Op_t>
	__global__ void op_kernel(size_t num, Vector3* __restrict__ pos_force, size_t* type_ids, ParticleType* types) {
	    for (size_t i = threadIdx.x + blockIdx.x*blockDim.x; i < num; i+= gridDim.x*blockDim.x) {
		Vector3& pos = pos_force[i];
		Vector3& mom = pos_force[num+i];
		Vector3& force = pos_force[2*num+i];
		ParticleType& t = types[type_ids[i]];
	    
		using expander = int[];
		(void)expander{0, (void(Op_t::op(&pos, &force, t, &mom)))...};	
	    }
	}
    }

    template<size_t block_size=32, bool is_BD=true, typename ...Op_t>
    class LocalOneParticleLoop : public PatchOp {
	void compute(Patch* patch) {
	    size_t num_blocks = (patch->num+1)/block_size;
	    if (is_BD) {
		BD::op_kernel<Op_t...><<<block_size, num_blocks>>>(patch->num, patch->pos, patch->type_ids, patch->types);
	    } else {
		MD::op_kernel<Op_t...><<<block_size, num_blocks>>>(patch->num, patch->pos, patch->type_ids, patch->types);
	    }
	}
    };

    struct OpIntegrateBD {
	HOST DEVICE static void op(Vector3& __restrict__ pos,
				   Vector3& __restrict__ force,
				   ParticleType& type,
				   SimParameters& sim,
				   Random& random,
				   Vector3* __restrict__ mom = nullptr) {

	    const float Dt = type.diffusion*sim.timestep;
	    Vector3 R = random.gaussian_vector(); // TODO who owns "random" object; how is it's state stored and recalled (especially if hardware changes); how is the implementation efficient on HOST/DEVICE?

	    const Vector3 new_pos = pos + (Dt / sim.kT) * force + sqrtf(2.0f * Dt) * R;
	    // pos = sim.wrap(new_pos); // TODO decide how wrapping will be handled
	}
    };

    struct OpComputeForceBD {
	HOST DEVICE static void op(Vector3& __restrict__ pos,
				   Vector3& __restrict__ force,
				   ParticleType& type,
				   SimParameters& sim,
				   Random& random,
				   Vector3* __restrict__ mom = nullptr) {

	    const float Dt = type.diffusion*sim.timestep;
	    Vector3 R = random.gaussian_vector(); // TODO who owns "random" object; how is it's state stored and recalled (especially if hardware changes); how is the implementation efficient on HOST/DEVICE?

	    const Vector3 new_pos = pos + (Dt / sim.kT) * force + sqrtf(2.0f * Dt) * R;
	    // pos = sim.wrap(new_pos); // TODO decide how wrapping will be handled
	}
    };

}
