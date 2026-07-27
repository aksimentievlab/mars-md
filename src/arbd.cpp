
#ifndef HOST_GUARD
#define HOST_GUARD
#include "IO/ConfigParser.h"
#endif

#ifdef USE_MPI
#include <mpi.h>
#endif

#include <cstdio>	// For printf
#include <cstring>	// For strcmp
#include <iostream> // For std::cout, std::endl (modern C++)
#include <string>	// For std::string (modern C++)
#include <vector>	// For std::vector

#include "ARBDException.h"
#include "Backend/Resource.h"
#include "IO/ConfigParser.h"
#include "SignalManager.h"
#include "SimManager.h"
#include "System/SimSystem.h"
// Define this if not provided by CMake/build system for version info
#ifndef VERSION
#define VERSION "Development Build - July 2026"
#endif

// Consider moving constants to a dedicated configuration header or class
const unsigned int kDefaultIMDPort = 71992;
const unsigned int kDefaultNodes = 1;
const unsigned int kDefaultGpus = 0;
unsigned int gpus[] = {kDefaultGpus};

struct ProgramOptions {
	std::string configFile;
	std::string outputFile;
	std::vector<short> gpuIds;
	int numGpus = 0;
	int numNodes = 1;
	int stride = 1;
};

bool is_unsigned_integer(const char* s) {
	if (!s || *s == '\0') {
		return false;
	}
	for (const char* p = s; *p; ++p) {
		if (!isdigit(static_cast<unsigned char>(*p))) {
			return false;
		}
	}
	return true;
}

bool parse_basic_args(int argc, char* argv[], ProgramOptions& opts) {
	if (argc == 2 && (strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0)) {
		printf("Usage: %s [OPTIONS] CONFIGFILE OUTPUT [SEED]\n", argv[0]);
		printf("\n");
		printf("  -h, --help         Display this help and exit\n");
		printf("  --info             Output basic CPU and CUDA information (stubbed) and exit\n");
		printf("  --version          Output version information and exit\n");
		printf("  -i, --imd=         IMD port (defaults to %u)\n", kDefaultIMDPort);
		printf("  -g, --gpus=        Number of GPUs to use (defaults to %u)\n", kDefaultGpus);
		printf("  -gid, --gpu_ids=   List of GPU IDs to use (e.g., --gid 0 1 2 3)\n");
		printf("  -s, --stride=      Stride for DCD processing (defaults to 1)\n");
		// printf("  -n, --nodes=       Number of nodes to use (defaults to %u)\n", kDefaultNodes);
		return false; // Indicates help was shown, program should exit
	} else if (argc == 2 && (strcmp(argv[1], "--version") == 0)) {
		printf("%s %s\n", argv[0], VERSION);
		return false; // Indicates version was shown, program should exit
	} else if (argc == 2 && (strcmp(argv[1], "--info") == 0)) {
		printf("Use the main program to see detailed resource information.\n");
		printf("Example: %s --help\n", argv[0]);
		return false;	   // Indicates info was shown, program should exit
	} else if (argc < 3) { // Expecting at least program_name, config, output
		printf("%s: missing arguments (expected CONFIGFILE OUTPUT)\n", argv[0]);
		printf("Try '%s --help' for more information.\n", argv[0]);
		return false; // Indicates error, program should exit
	}

	// Parse command line arguments
	for (int i = 1; i < argc; ++i) {
		if (strcmp(argv[i], "-g") == 0 || strcmp(argv[i], "--gpus") == 0) {
			if (i + 1 < argc) {
				opts.numGpus = atoi(argv[i + 1]);
				++i; // Skip next argument
			}
		} else if (strcmp(argv[i], "-gid") == 0 || strcmp(argv[i], "--gpu_ids") == 0) {
			// Parse GPU IDs until we hit a non-numeric token (a flag, or the
			// CONFIGFILE/OUTPUT positional args), not just any '-'-prefixed token -
			// otherwise the positional args get silently swallowed as bogus IDs.
			++i;
			while (i < argc && is_unsigned_integer(argv[i])) {
				opts.gpuIds.push_back(atoi(argv[i]));
				++i;
			}
			--i; // Back up one since we'll increment in the loop
		} else if (strcmp(argv[i], "-n") == 0 || strcmp(argv[i], "--nodes") == 0) {
			if (i + 1 < argc) {
				opts.numNodes = atoi(argv[i + 1]);
				++i; // Skip next argument
			}
		} else if (strcmp(argv[i], "-s") == 0 || strcmp(argv[i], "--stride") == 0) {
			if (i + 1 < argc) {
				opts.stride = atoi(argv[i + 1]);
				if (opts.stride <= 0) {
					printf("ERROR: Invalid stride value: %d (must be > 0)\n", opts.stride);
					return false;
				}
				++i; // Skip next argument
			}
		}
	}

	// Find config and output files (last two non-flag arguments)
	int fileArgs = 0;
	for (int i = argc - 1; i >= 1 && fileArgs < 2; --i) {
		if (argv[i][0] != '-') {
			if (fileArgs == 0) {
				opts.outputFile = argv[i];
			} else if (fileArgs == 1) {
				opts.configFile = argv[i];
			}
			++fileArgs;
		}
	}

	if (opts.configFile.empty() || opts.outputFile.empty()) {
		printf("%s: missing arguments (expected CONFIGFILE OUTPUT)\n", argv[0]);
		printf("Try '%s --help' for more information.\n", argv[0]);
		return false;
	}

	return true;
}

int main(int argc, char* argv[]) {
	// MPI Initialization (kept as is, conditional)

	ARBD::SignalManager::manage_segfault();

	ProgramOptions options;
	if (!parse_basic_args(argc, argv, options)) {
		return (argc < 3 &&
				!(argc == 2 &&
				  (strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0 ||
				   strcmp(argv[1], "--version") == 0 || strcmp(argv[1], "--info") == 0)))
				   ? 1
				   : 0;
	}

	// Print a startup message
	// Use std::cout for modern C++
	std::cout << "--- Atomic Resolution Brownian Dynamics (ARBD) ---" << std::endl;
	std::cout << "Version: " << VERSION << std::endl;
	std::cout << "Config File: " << options.configFile << std::endl;
	std::cout << "Output Target: " << options.outputFile << std::endl;

	std::cout << "Initializing Simulation Manager..." << std::endl;

	std::vector<ARBD::Resource> resource_collection;

#ifdef USE_CUDA
	std::cout << "ARBD compiled with CUDA support." << std::endl;
	int deviceCount = 0;
	if (cudaGetDeviceCount(&deviceCount) == cudaSuccess && deviceCount > 0) {
		// If user specified GPU IDs, use those
		if (!options.gpuIds.empty()) {
			for (int gpuId : options.gpuIds) {
				if (gpuId >= 0 && gpuId < deviceCount) {
					resource_collection.push_back(
						ARBD::Resource::create_cuda_device(static_cast<short>(gpuId)));
				} else {
					std::cout << "Warning: GPU ID " << gpuId << " is invalid (available: 0-"
							  << (deviceCount - 1) << ")" << std::endl;
				}
			}
		}
		// If user specified number of GPUs, use first N devices
		else if (options.numGpus > 0) {
			int gpusToUse = std::min(options.numGpus, deviceCount);
			for (int i = 0; i < gpusToUse; ++i) {
				resource_collection.push_back(
					ARBD::Resource::create_cuda_device(static_cast<short>(i)));
			}
		}
		// Default: use all available GPUs
		else {
			for (int i = 0; i < deviceCount; ++i) {
				resource_collection.push_back(
					ARBD::Resource::create_cuda_device(static_cast<short>(i)));
			}
		}
	}

	// Fallback to CPU if no GPUs available or selected
	if (resource_collection.empty()) {
		std::cout << "No GPUs available. Falling back to CPU." << std::endl;
		resource_collection.push_back(ARBD::Resource(ARBD::ResourceType::CPU));
	}

#elif defined(USE_SYCL)
	std::cout << "ARBD compiled with SYCL support." << std::endl;

	// Discover SYCL devices before building/validating any SYCL Resource -
	// Resource::validate() checks id_ against SYCL::Manager::device_count(),
	// which always returns 0 until Manager::init() has run.
	size_t deviceCount = 0;
	try {
		ARBD::SYCL::Manager::init();
		deviceCount = ARBD::SYCL::Manager::device_count();
	} catch (const ARBD::Exception& e) {
		std::cout << "SYCL device discovery failed: " << e.what() << std::endl;
	}

	if (deviceCount > 0) {
		if (!options.gpuIds.empty()) {
			for (int gpuId : options.gpuIds) {
				if (gpuId >= 0 && static_cast<size_t>(gpuId) < deviceCount) {
					resource_collection.push_back(
						ARBD::Resource::create_sycl_device(static_cast<short>(gpuId)));
				} else {
					std::cout << "Warning: GPU ID " << gpuId << " is invalid (available: 0-"
							  << (deviceCount - 1) << ")" << std::endl;
				}
			}
		}
		// If user specified number of GPUs, try first N devices
		else if (options.numGpus > 0) {
			size_t gpusToUse = std::min(static_cast<size_t>(options.numGpus), deviceCount);
			for (size_t i = 0; i < gpusToUse; ++i) {
				resource_collection.push_back(
					ARBD::Resource::create_sycl_device(static_cast<short>(i)));
			}
		}
		// Default: try device 0
		else {
			resource_collection.push_back(ARBD::Resource::create_sycl_device(0));
		}
	}

	// Fallback to CPU if no SYCL devices available or selected
	if (resource_collection.empty()) {
		std::cout << "No SYCL devices available. Falling back to CPU." << std::endl;
		resource_collection.push_back(ARBD::Resource(ARBD::ResourceType::CPU));
	}

#elif defined(USE_METAL)
	std::cout << "ARBD compiled with METAL support." << std::endl;
	// For SYCL/Metal/OpenMP, force GPU ID to 0 and ignore user GPU specifications
	options.gpuIds.clear();
	options.gpuIds.push_back(0);
	options.numGpus = 1;

	resource_collection.push_back(ARBD::Resource(0));
#endif

	// Validate all selected resources and remove invalid ones
	std::cout << "Validating " << resource_collection.size() << " compute resource(s)..."
			  << std::endl;
	std::vector<ARBD::Resource> validResources;

	for (const auto& res : resource_collection) {
		try {
			// Create a copy to validate (validate() is const)
			ARBD::Resource resCopy = res;
			resCopy.validate();
			validResources.push_back(res);
			std::cout << "✓ " << res.toString() << " validated successfully" << std::endl;
		} catch (const ARBD::Exception& e) {
			std::cout << "✗ " << res.toString() << " validation failed: " << e.what() << std::endl;
		}
	}

	// Update resources with only valid ones
	resource_collection = validResources;

	// If no valid resources, fallback to CPU
	if (resource_collection.empty()) {
		std::cout << "No valid compute resources found. Falling back to CPU." << std::endl;
	}

	std::cout << "Selected " << resource_collection.size() << " compute resource(s): ";
	for (const auto& res : resource_collection) {
		std::cout << res.toString() << " ";
	}
	std::cout << std::endl;

	ARBD::SimSystem sys(resource_collection);
	ARBD::ConfigParser parser(sys, options.configFile);
	// OUTPUT is a required CLI positional (see usage), so it always takes
	// precedence over any outputName/output_name set in the config file.
	sys.set_output_name(options.outputFile);
	sys.validate_physical_parameters();
	sys.validate_method_parameters();
	sys.validate_output_parameters();
	if (!sys.is_valid()) {
		throw ARBD::Exception(ARBD::ExceptionType::ValueError,
							  ARBD::SourceLocation(),
							  "Invalid system configuration");
	}
	ARBD::SimManager manager(sys);
	manager.set_initial_particles(parser.get_init_particles());
	manager.set_bonded_interactions(parser.get_init_bonded_interactions());
	manager.init();
	manager.run();

	// Build system and manager
	// ARBD::SimSystem sys(conf, resource_collection);
	// ARBD::SimManager manager(sys, resource_collection);

	// Single initialization-time domain decomposition
	// sys.decompose_system();

	// Main simulation loop orchestration lives in SimManager::run()
	// Pseudocode inside run(): build neighbor lists, schedule patch ops,
	// halo exchange (if multi-resource), integrate, write outputs
	// manager.run();

	/*
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
