// Conditional includes for MPI and CUDA
#ifdef USE_MPI
#include <mpi.h>
#endif

#ifdef USE_CUDA
#include <cuda.h> // Or <cuda.h> depending on CUDAManager's needs
#include <cuda_runtime.h>
#endif

#include <cstdio>    // For printf
#include <cstring>   // For strcmp
#include <string>    // For std::string (modern C++)
#include <iostream>  // For std::cout, std::endl (modern C++)
#include "SignalManager.h"

// Define this if not provided by CMake/build system for version info
#ifndef VERSION
#define VERSION "Development Build - June 2025"
#endif

// Consider moving constants to a dedicated configuration header or class
const unsigned int kDefaultIMDPort = 71992;

struct ProgramOptions {
    std::string configFile;
    std::string outputFile;
};


bool parse_basic_args(int argc, char* argv[], ProgramOptions& opts) {
    if (argc == 2 && (strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0)) {
        printf("Usage: %s [OPTIONS] CONFIGFILE OUTPUT [SEED]\n", argv[0]);
        printf("\n");
        printf("  -h, --help         Display this help and exit\n");
        printf("  --info             Output basic CPU and CUDA information (stubbed) and exit\n");
        printf("  --version          Output version information and exit\n");
        printf("\n  (More options will be enabled as components are modernized)\n");
        // printf("  -r, --replicas=    Number of replicas to run\n");
        // printf("  -g, --gpu=         Index of gpu to use (defaults to 0)\n");
        // printf("  -i, --imd=         IMD port (defaults to %u)\n", kDefaultIMDPort);
        // printf("  -d, --debug        Debug mode\n");
        // printf("  --safe             Do not use CUDAs that may timeout\n");
        // printf("  --unsafe           Use CUDAs that may timeout (default)\n");
        return false; // Indicates help was shown, program should exit
    } else if (argc == 2 && (strcmp(argv[1], "--version") == 0)) {
        printf("%s %s\n", argv[0], VERSION);
        return false; // Indicates version was shown, program should exit
    } else if (argc == 2 && (strcmp(argv[1], "--info") == 0)) {
#ifdef USE_CUDA

        printf("CUDA is enabled.\n");
        // Add a very simple device query here for Week 1 if possible
        int deviceCount = 0;
        cudaError_t err = cudaGetDeviceCount(&deviceCount);
        if (err == cudaSuccess) {
            printf("Number of CUDA devices found: %d\n", deviceCount);
            for (int i = 0; i < deviceCount; ++i) {
                cudaDeviceProp deviceProp;
                cudaGetDeviceProperties(&deviceProp, i);
                printf("  Device %d: %s\n", i, deviceProp.name);
            }
        } else {
            printf("cudaGetDeviceCount failed: %s\n", cudaGetErrorString(err));
        }
#else
        printf("ARBD not compiled with CUDA support.\n");
#endif
        printf("Basic system info (more details to come).\n");
        return false; // Indicates info was shown, program should exit
    } else if (argc < 3) { // Expecting at least program_name, config, output
        printf("%s: missing arguments (expected CONFIGFILE OUTPUT)\n", argv[0]);
        printf("Try '%s --help' for more information.\n", argv[0]);
        return false; // Indicates error, program should exit
    }

    if (argc >= 3) { // Simplistic check
        opts.configFile = argv[argc - 2];
        opts.outputFile = argv[argc - 1];
        // if (argc >= 4) { // Optional seed
        //     try {
        //         opts.seed = std::stoi(argv[argc - 1]);
        //         opts.outputFile = argv[argc - 2]; // Adjust if seed is last
        //         opts.configFile = argv[argc - 3];
        //     } catch (const std::exception& e) {
        //         printf("Warning: Could not parse seed, using default.\n");
        //     }
        // }
        return true; // Arguments parsed (minimally)
    }
    return false; // Should not reach here if logic above is correct
}

int main(int argc, char* argv[]) {
    // MPI Initialization (kept as is, conditional)
#ifdef USE_MPI
    MPI_Init(NULL, NULL);
    int world_rank = 0; // Default for non-MPI runs
    // int world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    // MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    // char processor_name[MPI_MAX_PROCESSOR_NAME];
    // int name_len;
    // MPI_Get_processor_name(processor_name, &name_len);
    // if (world_rank == 0) { // Print once
    //     printf("MPI Initialized. Hello from rank %d on %s (%d total ranks)\n",
    //            world_rank, processor_name, world_size);
    // }
#endif

    ARBD::SignalManager::manage_segfault();

    ProgramOptions options;
    if (!parse_basic_args(argc, argv, options)) {
#ifdef USE_MPI
        MPI_Finalize();
#endif
        return (argc < 3 && !(argc == 2 && (strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--version") == 0 || strcmp(argv[1], "--info") == 0))) ? 1 : 0;
    }

    // Print a startup message
    // Use std::cout for modern C++
    std::cout << "--- Atomic Resolution Brownian Dynamics (ARBD) ---" << std::endl;
    std::cout << "Version: " << VERSION << std::endl;
    std::cout << "Config File: " << options.configFile << std::endl;
    std::cout << "Output Target: " << options.outputFile << std::endl;

#ifdef USE_CUDA
    std::cout << "Initializing CUDA Manager..." << std::endl;
    // CUDAManager::init(); // Call the static init method
    // CUDAManager::select_gpus({0}); // Example: Select CUDA 0 for now
    // size_t n_gpus = CUDAManager::allGpuSize();
    // std::cout << "Number of available CUDAs: " << n_gpus << std::endl;
    // if (n_gpus > 0) {
    //     std::cout << "Selected CUDA(s) for simulation." << std::endl;
    // } else {
    //     std::cout << "No CUDAs available or selected for simulation." << std::endl;
    // }
#else
    std::cout << "CUDA support is disabled." << std::endl;
#endif

    std::cout << "Initializing Simulation Manager..." << std::endl;
    // SimManager sim; // Instantiate SimManager
                    // The constructor might take configuration options later.
                    // For Week 1, it might be a default constructor.

    // The main simulation run is commented out for Week 1
    // std::cout << "Starting simulation (stubbed for Week 1)..." << std::endl;
    // sim.run();
    // std::cout << "Simulation finished (stubbed)." << std::endl;

    // Commented out original complex argument parsing and simulation setup
    /*
    bool debug = false, safe = false;
    int replicas = 1;
    unsigned int imd_port = 0;
    bool imd_on = false;
    // ... (rest of the original complex parsing loop) ...

    char* configFile = options.configFile.data(); // Unsafe if string is empty
    char* outArg = options.outputFile.data();   // Unsafe

    // ... (original CUDA selection logic) ...

    // Configuration config(configFile, replicas, debug);
    // config.copyToCUDA();
    // GrandBrownTown brown(config, outArg,
    //      debug, imd_on, imd_port, replicas);
    // brown.run();
    */

#ifdef USE_MPI
    MPI_Finalize();
#endif

    return 0;
}
