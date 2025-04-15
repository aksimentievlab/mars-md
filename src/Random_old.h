#pragma once

#include "GPUManager.h"
#include "Types.h"
#include "Proxy.h"
#include <map>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#ifdef USE_CUDA
#include <curand_kernel.h>
#include <curand.h>
#include <cassert>
#include <mutex>
#include <iostream>

#define cuRandchk(ans) { cuRandAssert((ans), __FILE__, __LINE__); }
inline void cuRandAssert(curandStatus code, const char *file, int line, bool abort=true) {
    if (code != CURAND_STATUS_SUCCESS) {
        fprintf(stderr, "CURAND Error: %d (%s:%d)\n", code, file, line);
        if (abort) exit(code);
    }
}
#endif

/// Oct 2024, reworking Random classes a bit
// Q: What does the "front end" host interface to Random need to have?
// A:   A way to fetch and restore the state of the backend. Could also have functions for retrieving values, but these would not be used in most of our code, because we need something that works regardless of whether it's living on host or device

class RandomBackend;

template<size_t num_states> class RandomCPU;
#ifdef USE_CUDA
template<size_t num_states> class RandomGPU;
#endif

// class RandomBackend {
//     // TBD
// };

// class RandomBackendGPU : RandomBackend {};

// END Oct2024

namespace Random {

    // Interface
    template<typename RNG>
    HOST DEVICE inline typename RNG::state_t* get_gaussian_state(RNG* random);

    template<typename RNG>
    HOST DEVICE inline void set_gaussian_state(RNG* random, typename RNG::state_t* state);

    template<typename RNG>
    HOST DEVICE inline float gaussian(RNG* random, typename RNG::state_t* state = nullptr);
    
    // Host-only implementations for RandomCPU
    template<size_t num_states>
    HOST inline typename RandomCPU<num_states>::state_t* get_gaussian_state(RandomCPU<num_states>* random) {
        return random->get_gaussian_state();
    }

    template<size_t num_states>
    HOST inline void set_gaussian_state(RandomCPU<num_states>* random, typename RandomCPU<num_states>::state_t* state) {
        random->set_gaussian_state(state);
    }

    template<size_t num_states>
    HOST inline float gaussian(RandomCPU<num_states>* random, typename RandomCPU<num_states>::state_t* state) {
        return random->gaussian(state);
    }

    // Device implementations for RandomGPU
    #ifdef USE_CUDA
    template<size_t num_states>
    HOST DEVICE inline typename RandomGPU<num_states>::state_t* get_gaussian_state(RandomGPU<num_states>* random) {
        return random->get_gaussian_state();
    }

    template<size_t num_states>
    HOST DEVICE inline void set_gaussian_state(RandomGPU<num_states>* random, typename RandomGPU<num_states>::state_t* state) {
        random->set_gaussian_state(state);
    }

    template<size_t num_states>
    HOST DEVICE inline float gaussian(RandomGPU<num_states>* random, typename RandomGPU<num_states>::state_t* state) {
        return random->gaussian(state);
    }
    #endif

} // namespace Random

template<size_t num_states>
class RandomCPU {
    static_assert(num_states > 0);
public:
    using state_t = curandGenerator_t;

    HOST RandomCPU() : generator{nullptr}, buffer_index{2} {
        // Create a pseudo-random number generator
        cuRandchk(curandCreateGeneratorHost(&generator, CURAND_RNG_PSEUDO_DEFAULT));
    }

    HOST ~RandomCPU() {
        // Destroy the generator
        if (generator) {
            cuRandchk(curandDestroyGenerator(generator));
            generator = nullptr;
        }
    }

    HOST void init(unsigned long seed, size_t offset = 0) {
        cuRandchk(curandSetPseudoRandomGeneratorSeed(generator, seed));
        // Set the offset if needed
        if (offset > 0) {
            cuRandchk(curandSetGeneratorOffset(generator, offset));
        }
    }

    // Generate a single Gaussian random number
    HOST float gaussian(state_t* state = nullptr) {
        if (buffer_index >= 2) {
            // Buffer is empty, generate new random numbers
            cuRandchk(curandGenerateNormal(generator, buffer, 2, 0.0f, 1.0f));
            buffer_index = 0;
        }
        // Return the next number from the buffer
        return buffer[buffer_index++];
    }

    // Get the current generator state
    HOST state_t* get_gaussian_state() {
        return &generator;
    }

    // Set the generator state (not commonly used with cuRAND Host API)
    HOST void set_gaussian_state(state_t* state) {
        if (state != nullptr) {
            generator = *state;
        }
    }

    // Generate a 3D vector of Gaussian random numbers
    HOST Vector3 gaussian_vector(state_t* state = nullptr) {
        float vals[4]; // Request 4 numbers to satisfy the multiple of 2 requirement
        cuRandchk(curandGenerateNormal(generator, vals, 4, 0.0f, 1.0f));
        return Vector3(vals[0], vals[1], vals[2]);
    }

private:
    state_t generator;
    float buffer[2]; // Buffer to store extra random numbers
    int buffer_index; // Index to track usage
};

class RandomCPUOld {
public:
    using state_t = void;

    // Methods for maninpulating the state
    void init(size_t num, unsigned long seed, size_t offset) {
	// 
    };

    // Methods for generating random values
    HOST DEVICE inline float gaussian(state_t* state) {
	LOGWARN("RandomCPU::gaussian(): not implemented");
	return 0;
    };

    HOST DEVICE inline state_t* get_gaussian_state() { return nullptr; };

    HOST DEVICE inline void set_gaussian_state(state_t* state) {};

    // // HOST DEVICE inline float gaussian(RandomState* state) {};
    // HOST inline Vector3 gaussian_vector(size_t idx, size_t thread) {
    // 	return Vector3();
    // };
    // unsigned int integer() {
    // 	return 0;
    // };
    // unsigned int poisson(float lambda) {
    // 	return 0;
    // };
    // float uniform() {
    // 	return 0;
    // };
    // void reorder(int *a, int n);
};

#ifdef USE_CUDA

// Forward declarations for RandomGPU<>
template<size_t num_states> class RandomGPU_template; // This templated version wouldn't allow us to use std::variant to provide a nice uniform host/Python interface regardless of RNG backend. We leave it here anyway to test performance
// template<size_t num_states>
// __global__ void RandomGPU_template__test_kernel(RandomGPU_template<num_states>* rng, size_t num_vals, float* buf, unsigned long seed, size_t offset);

//class RandomGPU;
template<typename RNG>
__global__ void RandomGPU__test_kernel(RNG* rng, size_t num_vals, float* buf, unsigned long seed, size_t offset=0);


// Forward declarations
//
template<size_t num_states>
__global__ void RandomGPU__test_kernel(RandomGPU<num_states>* rng, size_t num_vals, float* buf, unsigned long seed, size_t offset);

template<size_t num_states>
class RandomGPU {
    static_assert(num_states > 0);
public:
    using state_t = curandStateXORWOW_t;

    // DEVICE RandomGPU_template(Random::Conf c, Resource& location) : num_states{c.num_threads}, states{nullptr} {

    HOST DEVICE RandomGPU_template() : states{nullptr} {
	// Not sure how this
	assert( threadIdx.x + blockIdx.x == 0 );
	
	LOGINFO("Creating RandomGPU_template");
	// assert( location.type == Resource::GPU );
	states = new state_t[num_states];
    }
    HOST DEVICE ~RandomGPU_template() {
	LOGINFO("Destroying RandomGPU_template");
	assert( threadIdx.x + blockIdx.x == 0 );
	if (states != nullptr) delete [] states;
	states = nullptr;
    }

    HOST static RandomGPU_template<num_states>* instantiate_on_GPU(RandomGPU_template<num_states> *dest = nullptr) {
	using RNG = RandomGPU_template<num_states>;
	RNG* rng = reinterpret_cast<RNG*>(new char[sizeof(RNG)]); // avoid calling constructor
	if (dest == nullptr) {
	    gpuErrchk(cudaMalloc((void**)&dest, sizeof(RNG)));
	}
	gpuErrchk(cudaMalloc((void**)&(rng->states), sizeof(state_t)*num_states));
	gpuErrchk(cudaMemcpy(dest, rng, sizeof(RNG),cudaMemcpyHostToDevice));
	rng->states = nullptr;
	// We avoided constructer, so don't call `delete rng;`
	return dest;
   }
	
    HOST DEVICE inline float gaussian(state_t* state) {
#ifdef __CUDA_ARCH__    
	return curand_normal( state );
#else
	LOGWARN("RandomGPU_template::gaussian(): not implemented on host");	
	return 0.0f;
#endif
    };
    
    // Useful for getting the RNG state for the thread from global
    // memory, putting it in local memory for use
    HOST DEVICE inline state_t* get_gaussian_state() {
#ifdef __CUDA_ARCH__    
	const size_t& i = threadIdx.x + blockDim.x*blockIdx.x;
	return (i < num_states && states != nullptr) ? &(states[i]) : nullptr;
#else
	return (state_t*) nullptr;
#endif
    };

    HOST DEVICE inline void set_gaussian_state(state_t* state) {
#ifdef __CUDA_ARCH__    
	const size_t& i = threadIdx.x + blockDim.x*blockIdx.x;
	if (i < num_states && state != nullptr) states[i] = *state;
#endif
    };

    HOST DEVICE inline Vector3 gaussian_vector( state_t* state) {
	return Vector3( gaussian(state), gaussian(state), gaussian(state) );
    };

    template<size_t kernel_threads = 64> 
    static void launch_test_kernel(unsigned long seed, size_t offset = 0, size_t num_vals = 4*num_states, RandomGPU_template<num_states>* rng = nullptr) {
	using RNG = RandomGPU_template<num_states>;
	static_assert( (num_states >= kernel_threads) );
	static_assert( (num_states % kernel_threads) == 0 );
    
	bool rng_was_null = (rng == nullptr);
	if (rng_was_null) rng = RNG::instantiate_on_GPU();

	float *buf_d, *buf;

	LOGTRACE("cudaMalloc {}", num_vals);
	gpuErrchk(cudaMalloc((void**)&buf_d, num_vals*sizeof(float)));
	LOGTRACE("...launching kernel");
	RandomGPU__test_kernel<<<1+(num_states-1)/kernel_threads, kernel_threads>>>(rng, num_vals, buf_d, seed, offset);
	buf = new float[num_vals];
	gpuErrchk(cudaDeviceSynchronize());
	LOGTRACE("...copying data back");
    
	gpuErrchk(cudaMemcpy(buf, buf_d, num_vals*sizeof(float),cudaMemcpyDeviceToHost));
	float sum = 0;
	float sum2 = 0;
	for (size_t i = 0; i < num_vals; ++i) {
	    sum += buf[i];
	    sum2 += buf[i]*buf[i];
	}
	LOGINFO("RNG gaussian mean, std, count: {}, {}, {}", sum/num_vals, sqrt((sum2/num_vals) - sum*sum/(num_vals*num_vals)), num_vals);

	if (rng_was_null) cudaFree( rng );
	cudaFree( buf_d );
	delete[] buf;
    }

    HOST DEVICE constexpr size_t get_num_states() { return num_states; };
    
private:
    state_t* states;	// RNG states stored in global memory 
};

// template<size_t num_states = 128>
// __global__ void RandomGPU_template__test_kernel(RandomGPU_template<num_states>* rng, size_t num_vals, float* buf, unsigned long seed, size_t offset = 0) {
//     using RNG = RandomGPU_template<num_states>;
//     // printf("THREAD %d\n", threadIdx.x);
//     auto local_state = Random::get_gaussian_state<RNG>(rng);
//     for (size_t idx = threadIdx.x + blockIdx.x*blockDim.x; idx < num_states; idx += blockDim.x * gridDim.x) {
// 	assert(blockIdx.x == 0);
// 	// curand_init(unsigned long long seed,
// 	// 	    unsigned long long sequence,
// 	// 	    unsigned long long offset,
// 	// 	    curandStateXORWOW_t *state)
// 	curand_init(seed, idx, offset, local_state);
//     }
	
//     for (size_t idx = threadIdx.x + blockIdx.x*blockDim.x; idx < num_vals; idx += blockDim.x * gridDim.x) {
// 	buf[idx] = Random::gaussian<RNG>(rng, local_state);
//     }
//     Random::set_gaussian_state<RNG>(rng, local_state);
// };
class RandomGPU {
public:
    using state_t = curandStateXORWOW_t;

    // DEVICE RandomGPU_template(Random::Conf c, Resource& location) : num_states{c.num_threads}, states{nullptr} {

    HOST DEVICE RandomGPU(size_t num_states) : states{nullptr} {
	assert( num_states > 0 );
	// Not sure how this
	assert( threadIdx.x + blockIdx.x == 0 );
	
	LOGINFO("Creating RandomGPU");
	// assert( location.type == Resource::GPU );
	states = new state_t[num_states];
    }
    HOST DEVICE ~RandomGPU() {
	LOGINFO("Destroying RandomGPU");
	assert( threadIdx.x + blockIdx.x == 0 );
	if (states != nullptr) delete [] states;
	states = nullptr;
    }

    HOST static RandomGPU* instantiate_on_GPU(size_t num_states, RandomGPU *dest = nullptr) {
	using RNG = RandomGPU;
	RNG* rng = reinterpret_cast<RNG*>(new char[sizeof(RNG)]); // avoid calling constructor
	if (dest == nullptr) {
	    gpuErrchk(cudaMalloc((void**)&dest, sizeof(RNG)));
	}
	rng->num_states = num_states;
	gpuErrchk(cudaMalloc((void**)&(rng->states), sizeof(state_t)*num_states));
	gpuErrchk(cudaMemcpy(dest, rng, sizeof(RNG),cudaMemcpyHostToDevice));
	rng->states = nullptr;
	// We avoided constructer, so don't call `delete rng;`
	return dest;
    }
	
    HOST DEVICE inline float gaussian(state_t* state) {
#ifdef __CUDA_ARCH__    
	return curand_normal( state );
#else
	LOGWARN("RandomGPU::gaussian(): not implemented on host");	
	return 0.0f;
#endif
    };
    
    // Useful for getting the RNG state for the thread from global
    // memory, putting it in local memory for use
    HOST DEVICE inline state_t* get_gaussian_state() {
#ifdef __CUDA_ARCH__    
	const size_t& i = threadIdx.x + blockDim.x*blockIdx.x;
	return (i < num_states && states != nullptr) ? &(states[i]) : nullptr;
#else
	return (state_t*) nullptr;
#endif
    };

    HOST DEVICE inline void set_gaussian_state(state_t* state) {
#ifdef __CUDA_ARCH__    
	const size_t& i = threadIdx.x + blockDim.x*blockIdx.x;
	if (i < num_states && state != nullptr) states[i] = *state;
#endif
    };

    HOST DEVICE inline Vector3 gaussian_vector( state_t* state) {
	return Vector3( gaussian(state), gaussian(state), gaussian(state) );
    };

    template<size_t num_states = 128, size_t kernel_threads = 64> 
    static void launch_test_kernel(unsigned long seed, size_t offset = 0, size_t num_vals = 4*num_states, RandomGPU* rng = nullptr) {
	using RNG = RandomGPU;
	static_assert( (num_states >= kernel_threads) );
	static_assert( (num_states % kernel_threads) == 0 );
    
	bool rng_was_null = (rng == nullptr);
	if (rng_was_null) rng = RNG::instantiate_on_GPU(num_states);

	float *buf_d, *buf;

	LOGTRACE("cudaMalloc {}", num_vals);
	gpuErrchk(cudaMalloc((void**)&buf_d, num_vals*sizeof(float)));
	LOGTRACE("...launching kernel");
	RandomGPU__test_kernel<<<1+(num_states-1)/kernel_threads, kernel_threads>>>(rng, num_vals, buf_d, seed, offset);
	buf = new float[num_vals];
	gpuErrchk(cudaDeviceSynchronize());
	LOGTRACE("...copying data back");
    
	gpuErrchk(cudaMemcpy(buf, buf_d, num_vals*sizeof(float),cudaMemcpyDeviceToHost));
	float sum = 0;
	float sum2 = 0;
	for (size_t i = 0; i < num_vals; ++i) {
	    sum += buf[i];
	    sum2 += buf[i]*buf[i];
	}
	LOGINFO("RNG gaussian mean, std, count: {}, {}, {}", sum/num_vals, sqrt((sum2/num_vals) - sum*sum/(num_vals*num_vals)), num_vals);

	if (rng_was_null) cudaFree( rng );
	cudaFree( buf_d );
	delete[] buf;
    };

    HOST DEVICE size_t get_num_states() const { return num_states; };
    
private:
    size_t num_states;
    state_t* states;	// RNG states stored in global memory 
};

template<typename RNG>
__global__ void RandomGPU__test_kernel(RNG* rng, size_t num_vals, float* buf, unsigned long seed, size_t offset) {
    // printf("THREAD %d\n", threadIdx.x);
    const size_t num_states = rng->get_num_states();
    auto local_state = Random::get_gaussian_state<RNG>(rng);
    for (size_t idx = threadIdx.x + blockIdx.x*blockDim.x; idx < num_states; idx += blockDim.x * gridDim.x) {
	assert(blockIdx.x == 0);
	// curand_init(unsigned long long seed,
	// 	    unsigned long long sequence,
	// 	    unsigned long long offset,
	// 	    curandStateXORWOW_t *state)
	curand_init(seed, idx, offset, local_state);
    }
    for (size_t idx = threadIdx.x + blockIdx.x*blockDim.x; idx < num_vals; idx += blockDim.x * gridDim.x) {
	buf[idx] = Random::gaussian<RNG>(rng, local_state);
    }
    Random::set_gaussian_state<RNG>(rng, local_state);
};


class RandomAcc {
public:
    using state_t = void;
   
    // Methods for maninpulating the state
    void init(size_t num, unsigned long seed, size_t offset) {
	//
	
    };

    // Methods for generating random values
    // template<typename S>
    HOST DEVICE inline float gaussian(state_t* state) {
	LOGWARN("RandomAcc::gaussian(): not implemented");
	return 0;
    };

    // template<typename S>
    HOST DEVICE inline state_t* get_gaussian_state() { return nullptr; };

    HOST DEVICE inline void set_gaussian_state(state_t* state) {};


    
    
    // // HOST DEVICE inline float gaussian(RandomState* state) {};
    // HOST inline Vector3 gaussian_vector(size_t idx, size_t thread) {
    // 	return Vector3();
    // };
    // unsigned int integer() {
    // 	return 0;
    // };
    // unsigned int poisson(float lambda) {
    // 	return 0;
    // };
    // float uniform() {
    // 	return 0;
    // };
    // void reorder(int *a, int n);
};

// class RandomTest {
//     static constexpr bool no_send = true;
//     using RNGp = std::variant<RandomAcc*, RandomCPU*>;
//     RNGp rng;		// could be host or device

//     template<typename T>
//     auto set( T ptr ) {
// 	rng = ptr;
//     };

//         // Example function to get Gaussian state (optional)
//     template <typename T>
//     auto get_gaussian_state() {
//         // if (auto ptr = std::get_if<T*>(rng)) {
//         //     return (*ptr)->get_gaussian_state();
//         // }
//         // throw std::runtime_error("Requested type not currently held in variant.");
//         // if (auto ptr = std::get_if<std::shared_ptr<T>>(&rng)) {
// 	return std::get<T>(&rng)->get_gaussian_state();
// 	    //}
// 	    // throw std::runtime_error("Requested type not currently held in variant.");
//     }    
// };

class RandomI {

public:
    static constexpr bool no_send = true;
    float gaussian() { return rng.gaussian(); }

    std::any get_gaussian_state() {
	return std::visit([this](auto&& ptr) -> std::any {
        using PtrType = std::decay_t<decltype(ptr)>;
        if constexpr (std::is_same_v<PtrType, RandomAcc*> || std::is_same_v<PtrType, RandomCPU*>) {
            if (ptr != nullptr) {
                return ptr->get_gaussian_state();
            }
        }
        throw std::runtime_error("Invalid or nullptr variant state.");
    }, rng);
    }

    float gaussian() {
	return std::visit([this](auto&& ptr) -> float {
        using PtrType = std::decay_t<decltype(ptr)>;
        if constexpr (std::is_same_v<PtrType, RandomAcc*> || std::is_same_v<PtrType, RandomCPU*>) {
            if (ptr != nullptr) {
                return ptr->gaussian(ptr->get_gaussian_state());
            }
        }
        throw std::runtime_error("Invalid or nullptr variant state.");
    }, rng);
    }
    
    float gaussian(std::any state) {
	if (std::holds_alternative<RandomAcc*>(rng)) {
	    return _gaussian_helper<RandomAcc>(state);
	} else if (std::holds_alternative<RandomCPU*>(rng)) {
	    return _gaussian_helper<RandomCPU>(state);
	} else {
	    throw std::runtime_error("Requested type not currently held in variant.");
	}
    }

private:
    template<typename T>
    std::any _get_gaussian_state_helper() {
	auto ptr = std::get<T*>(rng);
        if (ptr != nullptr) {
            auto state = ptr->get_gaussian_state();
	    return std::any(state);
        } else {
	    throw std::runtime_error("Attempted to obtain RNG state from uninitialized Random interface.");
	}
    }
    template<typename T>
    float _gaussian_helper(std::any state) {
	auto ptr = std::get<T*>(rng);
        if (ptr != nullptr) {
            return ptr->gaussian(std::any_cast<typename T::state_t*>(state));
        } else {
	    throw std::runtime_error("Attempted to obtain gaussian from uninitialized Random interface.");
	}
    }

    RNG* rng;		// could be stored on host or device
};


class RandomTest1 {
    static constexpr bool no_send = true;
    using RNGp = std::variant<RandomAcc*, RandomCPU*>;

public:
    template<typename T>
    auto set( T* ptr ) { rng = ptr; };

    std::any get_gaussian_state() {
	// if (std::holds_alternative<RandomAcc*>(rng)) {
	//     return _get_gaussian_state_helper<RandomAcc>();
	// } else if (std::holds_alternative<RandomCPU*>(rng)) {
	//     return _get_gaussian_state_helper<RandomCPU>();
	// } else {
	//     throw std::runtime_error("Requested type not currently held in variant.");
	// }
	return std::visit([this](auto&& ptr) -> std::any {
        using PtrType = std::decay_t<decltype(ptr)>;
        if constexpr (std::is_same_v<PtrType, RandomAcc*> || std::is_same_v<PtrType, RandomCPU*>) {
            if (ptr != nullptr) {
                return ptr->get_gaussian_state();
            }
        }
        throw std::runtime_error("Invalid or nullptr variant state.");
    }, rng);
    }
    float gaussian(std::any state) {
	if (std::holds_alternative<RandomAcc*>(rng)) {
	    return _gaussian_helper<RandomAcc>(state);
	} else if (std::holds_alternative<RandomCPU*>(rng)) {
	    return _gaussian_helper<RandomCPU>(state);
	} else {
	    throw std::runtime_error("Requested type not currently held in variant.");
	}
    }
private:
    template<typename T>
    std::any _get_gaussian_state_helper() {
	auto ptr = std::get<T*>(rng);
        if (ptr != nullptr) {
            auto state = ptr->get_gaussian_state();
	    return std::any(state);
        } else {
	    throw std::runtime_error("Attempted to obtain RNG state from uninitialized Random interface.");
	}
    }
    template<typename T>
    float _gaussian_helper(std::any state) {
	auto ptr = std::get<T*>(rng);
        if (ptr != nullptr) {
            return ptr->gaussian(std::any_cast<typename T::state_t*>(state));
        } else {
	    throw std::runtime_error("Attempted to obtain gaussian from uninitialized Random interface.");
	}
    }

    RNGp rng;		// could be stored on host or device
};
// class RandomTest {
//     static constexpr bool no_send = true;
//     using RNGp = std::variant<std::weak_ptr<RandomAcc>, std::weak_ptr<RandomCPU>>;
//     RNGp rng;		// could be host or device

//     template<typename T>
//     auto set( T* ptr ) { rng = std::weak_ptr<T>(ptr); };
//         // Example function to get Gaussian state (optional)
//     template <typename T>
//     auto get_gaussian_state() {
//         if (auto ptr = std::get_if<std::weak_ptr<T>>(&rng)) {
//             return (*ptr)->get_gaussian_state();
//         }
//         throw std::runtime_error("Requested type not currently held in variant.");
//     }

//     template <typename T>
//     auto gaussian() {
//         if (auto ptr = std::get_if<std::weak_ptr<T>>(&rng)) {
//             return (*ptr)->gaussian();
//         }
//         throw std::runtime_error("Requested type not currently held in variant.");
//     }
    
// };

    // template<typename T>
    // inline typename T::state_t* get_gaussian_state() {
    // 	return std::get<std::shared_ptr<T>> rng->get_gaussian_state();
    // }

   // auto get_gaussian_state() {
   //      return visit_rng([](auto&& ptr) {
   //          using T = std::decay_t<decltype(ptr)>;
   //          return ptr->get_gaussian_state(); // Access the method based on the type
   //      });
   //  }

//    static constexpr bool no_send = true;
 // Should cause compilation error or runtime issues! We won't know the type of hte state in the backend. What are some ways around this?
	// one approach might be to make the result depend on


    // inline auto set_gaussian_state() {
    // 	return rng->set_gaussian_state();
    // }
    // location? other stuff?


#endif
