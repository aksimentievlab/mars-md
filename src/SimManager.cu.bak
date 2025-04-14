#include "SimManager.h"
#include "Random.h"
#include <memory>

void SimManager::run() {
    LOGINFO("Running!");
    // TODO: process a command queue?

    // SimSystem sys = SimSystem();
    // Patch p(10,0,0,sys);

    Patch p;
    LOGINFO("init patcj");
//*
    RandomCPU<> rng{};
    LOGINFO("init RandomCPU");
    rng.init(124);
    for (int i = 0; i < 10; ++i) {
        float randNum = rng.gaussian();
        std::cout << "Random Gaussian Number: " << randNum << std::endl;
    }

    // Generate a random vector
    Vector3 randVec = rng.gaussian_vector();
    std::cout << "Random Gaussian Vector: (" << randVec.x << ", " << randVec.y << ", " << randVec.z << ")" << std::endl;

    // RandomAcc rng{};
    //ProxyPatch p2(10,0,0);
// ||||||| Stash base
//     RandomCPU rng{};
//     LOGINFO("there2!");
//     // RandomAcc rng{};
//     //ProxyPatch p2(10,0,0);
// =======
//     // RandomCPU rng{};
//     // LOGINFO("there2!");
//     RandomAcc rng1{};
//     RandomTest rng{};
//     // rng.set<decltype(rng1)>(&rng1);
//     rng.set(&rng1);
//     // ProxyPatch p2(10,0,0);
// >>>>>>> Stashed changes

    // p.add_compute( std::make_unique<LocalPairForce>() );
    // p.add_compute( std::make_unique<NeighborPairForce>() );

// #ifdef USE_CUDA
//     p.add_compute( std::make_unique<BDIntegrateCUDA>() );
//     p.add_compute( std::make_unique<LocalBondedCUDA>() );
// #else
//     p.add_compute( std::make_unique<BDIntegrate>() );
//     p.add_compute( std::make_unique<LocalBonded>() );
// #endif

    
    LOGINFO("here!");
    // auto state = rng.get_gaussian_state<decltype(rng1)>();
    auto state = rng.get_gaussian_state();
    LOGINFO("there!");
    for (size_t step = 0; step < 10; ++step) {
	LOGINFO("where!");
	// LOGINFO("Step {:d}: random", step); // {:0.2f}", step, Random::gaussian(&rng,(RandomCPU::state_t*) nullptr));
	LOGINFO("Step {:d}: random {:0.2f}", step, rng.gaussian(state));
	p.compute();
#ifdef USE_CUDA
	cudaDeviceSynchronize();
#endif
    }
    // */
#ifdef USE_CUDA
    RandomGPU_template<128>::launch_test_kernel<64>((size_t) 1);
    cudaDeviceSynchronize();

    RandomGPU::launch_test_kernel<128,64>((size_t) 1);
    cudaDeviceSynchronize();
#endif

};
