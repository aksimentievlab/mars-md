#include "../extern/Catch2/extras/catch_amalgamated.hpp"

#include <cstdio>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>

// Include backend-specific headers
#ifdef USE_CUDA
#include "Backend/CUDA/CUDAManager.h"
#include "SignalManager.h"
#include <cuda.h>
#include <nvfunctional>
#endif

#ifdef USE_SYCL
#include "Backend/SYCL/SYCLManager.h"
#include <sycl/sycl.hpp>
#endif

#ifdef USE_METAL
#include "Backend/METAL/METALManager.h"
#endif

// Common includes
#include "ARBDLogger.h"
#include "Backend/Buffer.h"
#include "Backend/Events.h"
#include "Backend/Resource.h"
#ifdef USE_MPI
#include "Backend/MPIManager.h"
#endif
#include "Types/TypeName.h"
#include "Types/Types.h"

// Use Catch2 v3 amalgamated header (self-contained)

// Macro for run_trial function - defines run_trial as an alias to run_trial function
#define DEF_RUN_TRIAL using Tests::run_trial;

// Macro for cleanup function - defines cleanup as an alias to TestBackendManager::cleanup
#define DEF_CLEANUP using Tests::cleanup;

namespace Tests {

// =============================================================================
// Backend-specific kernel implementations
// =============================================================================

#if defined(USE_CUDA) && defined(__CUDACC__)
template<typename Op_t, typename R, typename... T>
__global__ void cuda_op_kernel(R* result, T... args) {
	if (blockIdx.x == 0 && threadIdx.x == 0) {
		*result = Op_t::op(args...);
	}
}
#endif

// =============================================================================
// Unified Backend Manager
// =============================================================================

/**
 * @brief Unified backend manager for test execution across different compute backends
 */
class TestBackendManager {
  private:
	static TestBackendManager* instance_;
	static std::mutex mutex_;
	bool initialized_ = false;

	// Unified resource owning its own streams/queues
	ARBD::Resource device_resource_{};
	std::vector<ARBD::Resource> resources_{};

	// Private constructor for singleton pattern
	TestBackendManager() {
		initialize();
	}

  public:
	// Delete copy constructor and assignment operator
	TestBackendManager(const TestBackendManager&) = delete;
	TestBackendManager& operator=(const TestBackendManager&) = delete;

	// Expose the active resource for buffer operations
	const ARBD::Resource& get_resource() const {
		return device_resource_;
	}

	const std::vector<ARBD::Resource>& get_resources() const {
		return resources_;
	}

	// Preferred buffer-based allocation helpers
	template<typename R>
	ARBD::DeviceBuffer<R> allocate_buffer(size_t count) {
		if (!initialized_) {
			std::cerr << "Warning: Backend not initialized, skipping buffer allocation"
					  << std::endl;
			return ARBD::DeviceBuffer<R>{};
		}
		return ARBD::DeviceBuffer<R>(count, device_resource_);
	}

	template<typename R>
	void free_buffer(ARBD::DeviceBuffer<R>& buffer) {
		if (!initialized_)
			return;
		buffer.clear();
	}

	// Back-compat allocation API backed by Buffer policies
	template<typename R>
	R* allocate_device_memory(size_t count) {
		if (!initialized_) {
			std::cerr << "Warning: Backend not initialized, skipping memory allocation"
					  << std::endl;
			return nullptr;
		}
		void* queue = device_resource_.get_stream(ARBD::StreamType::Memory);
		return static_cast<R*>(
			ARBD::BackendPolicy::allocate(device_resource_, count * sizeof(R), queue, true));
	}

	template<typename R>
	void free_device_memory(R* ptr) {
		if (!ptr)
			return;
		if (!initialized_) {
			std::cerr << "Warning: Backend not initialized, skipping memory deallocation"
					  << std::endl;
			return;
		}
		void* queue = device_resource_.get_stream(ARBD::StreamType::Memory);
		ARBD::BackendPolicy::deallocate(static_cast<void*>(ptr), queue, true);
	}

	// Get singleton instance
	static TestBackendManager& getInstance() {
		std::lock_guard<std::mutex> lock(mutex_);
		if (instance_ == nullptr) {
			instance_ = new TestBackendManager();
		}
		return *instance_;
	}

	// Get resource for MPI operations (all ranks have initialized resources)
	const ARBD::Resource& get_resource_for_mpi() const {
		return device_resource_;
	}

	// Cleanup singleton instance
	static void cleanup() {
		std::lock_guard<std::mutex> lock(mutex_);
		if (instance_ != nullptr) {
			instance_->finalize();
			delete instance_;
			instance_ = nullptr;
		}
	}

	~TestBackendManager() {
		finalize();
	}

	void initialize() {
		if (initialized_)
			return;

		try {
#ifdef USE_MPI
			// Initialize MPI once for the test session (all ranks)
			ARBD::MPI::Manager::instance().init();
			// Each rank initializes its own backend resources
#endif
#ifdef USE_CUDA
			ARBD::SignalManager::manage_segfault();
			// Build ResourceCollection from all CUDA devices
			ARBD::CUDA::Manager::init();
			ARBD::CUDA::Manager::load_info();
			int cuda_count = ARBD::CUDA::Manager::device_count();
			for (int i = 0; i < cuda_count; ++i) {
				resources_.push_back(
					ARBD::Resource::create_cuda_device(static_cast<short>(i)));
			}
			device_resource_ =
				resources_.empty() ? ARBD::Resource{} : resources_.front();
#elif defined(USE_SYCL)
			// Build ResourceCollection from all SYCL devices
			// Note: SYCL Manager will show "[WARN] SYCL Manager already initialized" on non-rank-0
			// This is expected behavior when multiple MPI ranks initialize the backend
			ARBD::SYCL::Manager::init();
			ARBD::SYCL::Manager::load_info();
			size_t sycl_count = ARBD::SYCL::Manager::device_count();
			for (size_t i = 0; i < sycl_count; ++i) {
				resources_.push_back(
					ARBD::Resource::create_sycl_device(static_cast<short>(i)));
			}
			device_resource_ =
				resources_.empty() ? ARBD::Resource{} : resources_.front();
#elif defined(USE_METAL)
			// Build ResourceCollection from all METAL devices
			ARBD::METAL::Manager::init();
			ARBD::METAL::Manager::load_info();
			resources_.emplace_back(ARBD::ResourceType::METAL, static_cast<short>(0));
			device_resource_ =
				resources_.empty() ? ARBD::Resource{} : resources_.front();
#else
			// CPU fallback: single resource
			device_resource_ = ARBD::Resource(ARBD::ResourceType::CPU, 0);
			resources_.push_back(device_resource_);
#endif
			device_resource_.ensure_context();
			initialized_ = true;
		} catch (const ARBD::Exception& e) {
			std::cerr << "Warning: Backend initialization failed: " << e.what() << std::endl;
			return;
		}
	}

	void finalize() {
		if (!initialized_)
			return;

#ifdef USE_MPI
		// Finalize MPI at the end of tests
		if (ARBD::MPI::Manager::instance().is_initialized()) {
			ARBD::MPI::Manager::instance().finalize();
		}
#endif
		initialized_ = false;
	}

	bool isInitialized() const {
		return initialized_;
	}
};

// =============================================================================
// Unified Test Runner
// =============================================================================

/**
 * @brief Run a test operation across different backends
 */
template<typename Op_t, typename R, typename... T>
void run_trial(std::string name, R expected_result, T... args) {
	using namespace ARBD;

	INFO(name);

	// Test CPU execution
	R cpu_result = Op_t::op(args...);
	CAPTURE(cpu_result);
	CAPTURE(expected_result);
	REQUIRE(cpu_result == expected_result);

	// Test the current backend (determined at compile time)
	TestBackendManager& manager = TestBackendManager::getInstance();

	// Check if backend is properly initialized
	if (!manager.isInitialized()) {
		WARN("Backend not properly initialized, skipping device execution");
		return;
	}

	// Use DeviceBuffer for device memory and copies
	ARBD::DeviceBuffer<R> device_buffer(1, manager.get_resource());

	R device_result;
	device_buffer.copy_to_host(&device_result, 1, true);

	CAPTURE(device_result);
	CHECK(cpu_result == device_result);
}

// Cleanup function for test suite
inline void cleanup() {
	TestBackendManager::cleanup();
}

} // namespace Tests

// =============================================================================
// Operation definitions (unchanged from original)
// ================
// =============================================================================
// Static member definitions
// =============================================================================

namespace Tests {
inline TestBackendManager* TestBackendManager::instance_ = nullptr;
inline std::mutex TestBackendManager::mutex_;
} // namespace Tests
