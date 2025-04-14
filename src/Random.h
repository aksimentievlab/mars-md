// Random.h
//Clean version for compile purposes

#pragma once

#include "GPUManager.h"
#include "Types.h"
#include "Proxy.h"
#include <map>
#include <vector>
#include <memory>
#include <random>

// CPU Implementation
template<size_t num_states>
class RandomCPU {
    static_assert(num_states > 0, "Number of states must be positive");

private:
    // CPU-specific RNG state
    struct State {
        std::mt19937 rng;
        std::normal_distribution<float> dist;
        State() : rng(std::random_device{}()), dist(0.0f, 1.0f) {}
    };

public:
    using state_type = State;

    RandomCPU() : state{std::make_unique<State>()} {}

    void init(unsigned long seed, size_t offset = 0) {
        state->rng.seed(seed);
        if (offset > 0) {
            for (size_t i = 0; i < offset; ++i) {
                state->dist(state->rng);
            }
        }
    }

    float gaussian(state_type* external_state = nullptr) {
        auto& s = external_state ? *external_state : *state;
        return s.dist(s.rng);
    }

    state_type* get_gaussian_state() {
        return state.get();
    }

    void set_gaussian_state(state_type* new_state) {
        if (new_state) {
            *state = *new_state;
        }
    }

    Vector3 gaussian_vector(state_type* external_state = nullptr) {
        return Vector3(gaussian(external_state), 
                      gaussian(external_state), 
                      gaussian(external_state));
    }

private:
    std::unique_ptr<State> state;
};

#ifdef __CUDACC__
// GPU Implementation
template<size_t num_states>
class RandomGPU {
    static_assert(num_states > 0, "Number of states must be positive");

public:
    using state_type = curandStateXORWOW_t;

    HOST DEVICE RandomGPU() : states{nullptr} {
        assert(threadIdx.x + blockIdx.x == 0);
        states = new state_type[num_states];
    }

    HOST DEVICE ~RandomGPU() {
        assert(threadIdx.x + blockIdx.x == 0);
        if (states) {
            delete[] states;
        }
    }

    DEVICE void init(unsigned long seed, size_t offset = 0) {
        auto state = get_gaussian_state();
        if (state) {
            size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
            curand_init(seed, idx, offset, state);
        }
    }

    DEVICE float gaussian(state_type* external_state = nullptr) {
        return curand_normal(external_state ? external_state : get_gaussian_state());
    }

    DEVICE state_type* get_gaussian_state() {
        size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
        return (idx < num_states && states) ? &states[idx] : nullptr;
    }

    DEVICE void set_gaussian_state(state_type* new_state) {
        size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
        if (idx < num_states && new_state && states) {
            states[idx] = *new_state;
        }
    }

    DEVICE Vector3 gaussian_vector(state_type* external_state = nullptr) {
        return Vector3(gaussian(external_state), 
                      gaussian(external_state), 
                      gaussian(external_state));
    }

private:
    state_type* states;
};
#endif

