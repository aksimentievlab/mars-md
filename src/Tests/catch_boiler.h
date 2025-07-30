#include <cstdio>
#include <iostream>
#include <memory>
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
#include "Math/TypeName.h"
#include "Math/Types.h"

// Use Catch2 v3 amalgamated header (self-contained)
#include "../../extern/Catch2/extras/catch_amalgamated.hpp"

// Macro for run_trial function - defines run_trial as an alias to run_trial function
#define DEF_RUN_TRIAL using Tests::run_trial;

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
	bool initialized_ = false;

  public:
	TestBackendManager() {
		initialize();
	}

	~TestBackendManager() {
		finalize();
	}

	void initialize() {
		if (initialized_)
			return;

#ifdef USE_CUDA
		ARBD::SignalManager::manage_segfault();
		// Initialize CUDA GPU Manager
		ARBD::CUDA::CUDAManager::init();
		ARBD::CUDA::CUDAManager::load_info();
#endif
#ifdef USE_SYCL
		ARBD::SYCL::SYCLManager::init();
		ARBD::SYCL::SYCLManager::load_info();
#endif
#ifdef USE_METAL
		ARBD::METAL::METALManager::init();
		ARBD::METAL::METALManager::load_info();
#endif
		initialized_ = true;
	}

	void finalize() {
		if (!initialized_)
			return;

#ifdef USE_CUDA
		ARBD::CUDA::CUDAManager::finalize();
#endif
#ifdef USE_SYCL
		ARBD::SYCL::SYCLManager::finalize();
#endif
#ifdef USE_METAL
		ARBD::METAL::METALManager::finalize();
#endif
		initialized_ = false;
	}

	void synchronize() {
#ifdef USE_CUDA
		cudaDeviceSynchronize();
#endif
#ifdef USE_SYCL
		ARBD::SYCL::SYCLManager::sync();
#endif
#ifdef USE_METAL
		ARBD::METAL::METALManager::sync();
#endif
	}

	template<typename R>
	R* allocate_device_memory(size_t count) {
#ifdef USE_CUDA
		R* ptr;
		ARBD::check_cuda_error(cudaMalloc((void**)&ptr, count * sizeof(R)), __FILE__, __LINE__);
		// Initialize device memory to zero
		ARBD::check_cuda_error(cudaMemset(ptr, 0, count * sizeof(R)), __FILE__, __LINE__);
		return ptr;
#elif defined(USE_SYCL)
		auto& queue = ARBD::SYCL::SYCLManager::get_current_queue();
		return sycl::malloc_device<R>(count, queue.get());
#elif defined(USE_METAL)
		auto& device = ARBD::METAL::METALManager::get_current_device();
		// Metal uses unified memory, so we can allocate using DeviceMemory
		// For simplicity, we'll use the manager's allocate function
		// Note: This is conceptual - actual Metal allocation would be different
		return static_cast<R*>(std::malloc(count * sizeof(R)));
#else
		// CPU fallback
		return static_cast<R*>(std::malloc(count * sizeof(R)));
#endif
	}

	template<typename R>
	void free_device_memory(R* ptr) {
		if (!ptr)
			return;

#ifdef USE_CUDA
		cudaFree(ptr);
#elif defined(USE_SYCL)
		auto& queue = ARBD::SYCL::SYCLManager::get_current_queue();
		sycl::free(ptr, queue.get());
#elif defined(USE_METAL)
		std::free(ptr);
#else
		// CPU fallback
		std::free(ptr);
#endif
	}

	template<typename R>
	void copy_to_device(R* device_ptr, const R* host_ptr, size_t count) {
#ifdef USE_CUDA
		ARBD::check_cuda_error(
			cudaMemcpy(device_ptr, host_ptr, count * sizeof(R), cudaMemcpyHostToDevice),
			__FILE__,
			__LINE__);
#elif defined(USE_SYCL)
		auto& queue = ARBD::SYCL::SYCLManager::get_current_queue();
		queue.get().memcpy(device_ptr, host_ptr, count * sizeof(R)).wait();
#elif defined(USE_METAL)
		std::memcpy(device_ptr, host_ptr, count * sizeof(R));
#else
		// CPU fallback
		std::memcpy(device_ptr, host_ptr, count * sizeof(R));
#endif
	}

	template<typename R>
	void copy_from_device(R* host_ptr, const R* device_ptr, size_t count) {
#ifdef USE_CUDA
		ARBD::check_cuda_error(
			cudaMemcpy(host_ptr, device_ptr, count * sizeof(R), cudaMemcpyDeviceToHost),
			__FILE__,
			__LINE__);
#elif defined(USE_SYCL)
		auto& queue = ARBD::SYCL::SYCLManager::get_current_queue();
		queue.get().memcpy(host_ptr, device_ptr, count * sizeof(R)).wait();
#elif defined(USE_METAL)
		std::memcpy(host_ptr, device_ptr, count * sizeof(R));
#else
		// CPU fallback
		std::memcpy(host_ptr, device_ptr, count * sizeof(R));
#endif
	}

	template<typename Op_t, typename R, typename... T>
	void execute_kernel(R* result_device, T... args) {
#ifdef USE_CUDA
#if defined(__CUDACC__)
		cuda_op_kernel<Op_t, R, T...><<<1, 1>>>(result_device, args...);
		ARBD::check_cuda_error(cudaGetLastError(), __FILE__, __LINE__);
		ARBD::check_cuda_error(cudaDeviceSynchronize(), __FILE__, __LINE__);
#else
		// Fallback: execute on host when not compiled with nvcc
		*result_device = Op_t::op(args...);
#endif
#elif defined(USE_SYCL)
		auto& queue = ARBD::SYCL::SYCLManager::get_current_queue();
		queue
			.submit([=](sycl::handler& h) {
				h.single_task([=]() { *result_device = Op_t::op(args...); });
			})
			.wait();
#elif defined(USE_METAL)
		// For Metal, we'll execute on CPU for now since compute shaders
		// require more complex setup. In a full implementation, this would
		// dispatch a Metal compute shader.
		*result_device = Op_t::op(args...);
#else
		// CPU fallback
		*result_device = Op_t::op(args...);
#endif
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
	TestBackendManager manager;

	R* device_result_d = manager.allocate_device_memory<R>(1);

	manager.execute_kernel<Op_t, R, T...>(device_result_d, args...);

	R device_result;
	manager.copy_from_device(&device_result, device_result_d, 1);
	manager.synchronize();

	manager.free_device_memory(device_result_d);

	CAPTURE(device_result);
	CHECK(cpu_result == device_result);
}

} // namespace Tests

// =============================================================================
// Operation definitions (unchanged from original)
// =============================================================================

namespace Tests::Unary {
template<typename R, typename T>
struct NegateOp {
	HOST DEVICE static R op(T in) {
		return static_cast<R>(-in);
	}
};

template<typename R, typename T>
struct NormalizedOp {
	HOST DEVICE static R op(T in) {
		return static_cast<R>(in.normalized());
	}
};
} // namespace Tests::Unary

namespace Tests::Binary {
// R is return type, T and U are types of operands
template<typename R, typename T, typename U>
struct AddOp {
	HOST DEVICE static R op(T a, U b) {
		return static_cast<R>(a + b);
	}
};

template<typename R, typename T, typename U>
struct SubOp {
	HOST DEVICE static R op(T a, U b) {
		return static_cast<R>(a - b);
	}
};

template<typename R, typename T, typename U>
struct MultOp {
	HOST DEVICE static R op(T a, U b) {
		return static_cast<R>(a * b);
	}
};

template<typename R, typename T, typename U>
struct DivOp {
	HOST DEVICE static R op(T a, U b) {
		return static_cast<R>(a / b);
	}
};
} // namespace Tests::Binary
