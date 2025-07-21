// SimManager.cu, from GrandBorwnTown.cu/cuh...etc
#include "SimManager.h"
#include "Random.h"
#include "ARBDLogger.h"
#include "ARBDException.h"
#include <memory>

namespace ARBD {

void SimManager::run() {
    LOGINFO("Starting simulation...");

    Patch p;
    LOGINFO("Initialized patch");

    RandomCPU<128> cpu_rng;
    LOGINFO("Initialized CPU RNG");
    
    cpu_rng.init(42);
    
    std::cout << "CPU RNG test:\n";
    for (int i = 0; i < 5; ++i) {
        float rand_num = cpu_rng.gaussian();
        std::cout << "  Gaussian(" << i << "): " << rand_num << "\n";
    }

    Vector3 rand_vec = cpu_rng.gaussian_vector();
    std::cout << "Random vector: ("
              << rand_vec.x << ", "
              << rand_vec.y << ", "
              << rand_vec.z << ")\n";

    for (size_t step = 0; step < 10; ++step) {
        LOGINFO("Processing step {}", step);
        p.compute();
        
        #ifdef __CUDACC__
        cudaDeviceSynchronize();
        #endif
    }

    LOGINFO("Simulation complete");
}

} // namespace ARBD
