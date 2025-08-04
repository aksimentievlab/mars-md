#pragma once

#include "Header.h"
#ifdef USE_CUDA
#include "Backend/CUDA/CUDAManager.h"
#endif

#ifdef USE_SYCL
#include "Backend/SYCL/SYCLManager.h"
#include <sycl/sycl.hpp>
#endif

#ifdef USE_METAL
#include "Backend/METAL/METALManager.h"
#endif

/**
 * @brief Get current device ID for device computing backends
 */
inline size_t get_device_id() {
#ifdef USE_CUDA
	if (cudaGetDevice(nullptr) == cudaSuccess) {
		int device;
		cudaGetDevice(&device);
		return static_cast<size_t>(device);
	}
#endif

#ifdef USE_SYCL
	try {
		return static_cast<size_t>(ARBD::SYCL::Manager::get_current_device().id());
	} catch (...) {
		return 0;
	}
#endif

#ifdef USE_METAL
	try {
		return static_cast<size_t>(ARBD::METAL::Manager::get_current_device().id());
	} catch (...) {
		return 0;
	}
#endif

	return 0;
}

/**
 * @brief Resource representation for heterogeneous computing
 * Supports CUDA, SYCL, and MPI resource types
 */
namespace ARBD {

enum class ResourceType { CPU, CUDA, SYCL, METAL };

/**
 * @brief Memory transfer policies
 */
enum class TransferType { HOST_TO_DEVICE, DEVICE_TO_HOST, DEVICE_TO_DEVICE, HOST_TO_HOST };

/**
 * @brief Backend capability traits for compile-time feature detection
 */
template<typename Backend>
struct BackendTraits {
	static constexpr bool supports_device_memory = false;
	static constexpr bool supports_async_execution = false;
	static constexpr bool supports_peer_access = false;
	static constexpr bool requires_explicit_sync = false;

	using context_type = void;
	using event_type = void;
	using stream_type = void;
};

/**
 * @brief Device metrics
 */
struct DeviceMetrics {
	ResourceType backend_type;
	size_t device_id;

	// Real-time memory state
	size_t total_memory = 0;
	size_t free_memory = 0;
	float memory_utilization = 0.0f; // Current percentage used

	// Real-time activity state
	bool has_active_kernels = false;
	size_t active_queues = 0;
	bool device_busy = false;

	// Load score based on real-time state
	float load_score = 0.0f;

	std::chrono::high_resolution_clock::time_point timestamp;
};
/**
 * @brief Load balancing strategies
 */
enum class LoadBalancingStrategy {
	ROUND_ROBIN,	// Simple round-robin
	LEAST_LOADED,	// Choose least loaded device
	MEMORY_BASED,	// Choose device with most free memory
	ACTIVITY_BASED, // Choose device with least activity
	HYBRID			// Combination of metrics
};
/**
 * @brief CUDA Backend Traits
 */
struct CUDABackend {
	static constexpr const char* name = "CUDA";
	static constexpr ResourceType resource_type = ResourceType::CUDA;
};

template<>
struct BackendTraits<CUDABackend> {
	static constexpr bool supports_device_memory = true;
	static constexpr bool supports_async_execution = true;
	static constexpr bool supports_peer_access = true;
	static constexpr bool requires_explicit_sync = true;

#ifdef USE_CUDA
	using context_type = int; // CUDA device ID
	using event_type = cudaEvent_t;
	using stream_type = cudaStream_t;
#else
	using context_type = void;
	using event_type = void;
	using stream_type = void;
#endif
};

/**
 * @brief SYCL Backend Traits
 */
struct SYCLBackend {
	static constexpr const char* name = "SYCL";
	static constexpr ResourceType resource_type = ResourceType::SYCL;
};

template<>
struct BackendTraits<SYCLBackend> {
	static constexpr bool supports_device_memory = true;
	static constexpr bool supports_async_execution = true;
	static constexpr bool supports_peer_access = false;
	static constexpr bool requires_explicit_sync = false;

#ifdef USE_SYCL
	using context_type = sycl::queue*; // void*
	using event_type = sycl::event*;   // void*
	using stream_type = sycl::queue*;  // void*
#else
	using context_type = void;
	using event_type = void;
	using stream_type = void;
#endif
};

/**
 * @brief METAL Backend Traits
 */
struct METALBackend {
	static constexpr const char* name = "METAL";
	static constexpr ResourceType resource_type = ResourceType::METAL;
};

template<>
struct BackendTraits<METALBackend> {
	static constexpr bool supports_device_memory = true;
	static constexpr bool supports_async_execution = true;
	static constexpr bool supports_peer_access = false;
	static constexpr bool requires_explicit_sync = false;

	using context_type = void; // METAL doesn't expose contexts directly
	using event_type = void;   // METAL command buffer events
	using stream_type = void;  // METAL command queues
};

#if !defined(__METAL_VERSION__) && !defined(__CUDA_ARCH__) && !defined(__SYCL_DEVICE_ONLY__)
/**
 * @brief Concept to check if a type is a valid backend
 */
template<typename T>
concept ValidBackend = requires {
	typename BackendTraits<T>;
	{ T::name } -> std::convertible_to<const char*>;
	{ T::resource_type } -> std::same_as<ResourceType>;
};
#endif

/**
 * @brief Resource representation for device computing environments
 *
 * The Resource class provides a unified interface for representing and managing
 * computational resources for device computing (CUDA, SYCL, METAL).
 * For distributed computing (MPI), use MPIResource from MPIBackend.h instead.
 *
 * @details This class manages different compute devices on a single machine:
 * - CUDA GPU devices for NVIDIA GPU computing
 * - SYCL devices for cross-platform parallel computing
 * - METAL devices for Apple GPU computing
 *
 * @example Basic Usage:
 * ```cpp
 * // Create a CUDA resource for device 0
 * ARBD::Resource cuda_res(ARBD::Resource::CUDA, 0);
 *
 * // Create a SYCL resource for device 1
 * ARBD::Resource sycl_res(ARBD::Resource::SYCL, 1);
 *
 * // Check if resource is local to current execution context
 * if (cuda_res.is_local()) {
 *     // Perform local operations
 * }
 * ```
 *
 * @see MPIBackend.h for distributed computing resources
 * @see ResourceType for available device resource types
 * @see is_local() for locality checking
 * @see getTypeString() for human-readable type names
 */

struct Resource {
#ifdef USE_CUDA
	static constexpr ResourceType DEFAULT_DEVICE = ResourceType::CUDA;
#elif defined(USE_SYCL)
	static constexpr ResourceType DEFAULT_DEVICE = ResourceType::SYCL;
#elif defined(USE_METAL)
	static constexpr ResourceType DEFAULT_DEVICE = ResourceType::METAL;
#else
	LOGERROR("Resource::Resource(): No device backend defined, using HOST only");
#endif

	ResourceType type;
	size_t id;
	Resource* parent;

	HOST DEVICE Resource() : type(DEFAULT_DEVICE), id(0), parent(nullptr) {}
	HOST DEVICE Resource(ResourceType t, size_t i) : type(t), id(i), parent(nullptr) {}
	HOST DEVICE Resource(ResourceType t, size_t i, Resource* p) : type(t), id(i), parent(p) {}

	HOST DEVICE constexpr const char* getTypeString() const {
		switch (type) {
		case ResourceType::CUDA:
			return "CUDA";
		case ResourceType::SYCL:
			return "SYCL";
		case ResourceType::METAL:
			return "METAL";
		default:
			return "Unknown";
		}
	}
#if !defined(__METAL_VERSION__) && !defined(__CUDA_ARCH__) && !defined(__SYCL_DEVICE_ONLY__)
	HOST DEVICE bool is_local() const {

		// We are executing on a device
		if (type == ResourceType::CPU) {
			return false;
		}
		// Check if the resource's device type and ID match the current device
		// This part requires device-specific ways to get current device ID,
		// which can be complex. Assuming for now that if it's not CPU,
		// and we are on a device, we are on the right one if the code is launched correctly.
		// A more robust implementation might be needed here.
		return true; // Simplified for device code
#else
	// We are executing on the host
	if (type == ResourceType::CPU) {
		return true;
	}

	bool ret = false;
#ifdef USE_CUDA
	if (type == ResourceType::CUDA) {
		int current_device;
		if (cudaGetDevice(&current_device) == cudaSuccess) {
			ret = (current_device == static_cast<int>(id));
		}
	}
#endif
#ifdef USE_SYCL
	if (type == ResourceType::SYCL) {
		try {
			auto& current_device = ARBD::SYCL::Manager::get_current_device();
			ret = (current_device.id() == id);
		} catch (...) {
			ret = false;
		}
	}
#endif
#ifdef USE_METAL
	if (type == ResourceType::METAL) {
		try {
			auto& current_device = ARBD::METAL::Manager::get_current_device();
			ret = (current_device.id() == id);
		} catch (...) {
			ret = false;
		}
	}
#endif
	return ret;
#endif
	}

	static Resource Local() {
#ifdef USE_CUDA
		int device;
		if (cudaGetDevice(&device) == cudaSuccess) {
			return Resource{ResourceType::CUDA, static_cast<size_t>(device)};
		}
#endif
#ifdef USE_SYCL
		try {
			auto& current_device = ARBD::SYCL::Manager::get_current_device();
			return Resource{ResourceType::SYCL, static_cast<size_t>(current_device.id())};
		} catch (...) {
		}
#endif
#ifdef USE_METAL
		try {
			auto& current_device = ARBD::METAL::Manager::get_current_device();
			return Resource{ResourceType::METAL, static_cast<size_t>(current_device.id())};
		} catch (...) {
		}
#endif
		// Default to HOST if no device context is active.
		return Resource{ResourceType::CPU, 0};
	}

	HOST DEVICE bool operator==(const Resource& other) const {
		return type == other.type && id == other.id;
	}
#if !defined(__METAL_VERSION__) && !defined(__CUDA_ARCH__) && !defined(__SYCL_DEVICE_ONLY__)
	HOST std::string toString() const {
		return std::string(getTypeString()) + "[" + std::to_string(id) + "]";
	}
#endif
	/**
	 * @brief Check if the resource supports asynchronous operations
	 */
	HOST DEVICE bool supports_async() const {
		switch (type) {
		case ResourceType::CUDA:
		case ResourceType::SYCL:
		case ResourceType::METAL:
			return true;

		default:
			return false;
		}
	}

	/**
	 * @brief Get the memory space type for this resource
	 */
	HOST DEVICE constexpr const char* getMemorySpace() const {
		switch (type) {
		case ResourceType::CPU:
			return "host";
		case ResourceType::CUDA:
			return "device";
		case ResourceType::SYCL:
			return "device";
		case ResourceType::METAL:
			return "device";
		default:
			return "host";
		}
	}

	/**
	 * @brief Check if this resource represents a device (GPU)
	 */
	HOST DEVICE bool is_device() const {
		return type == ResourceType::CUDA || type == ResourceType::SYCL ||
			   type == ResourceType::METAL;
	}

	/**
	 * @brief Check if this resource represents a host (CPU)
	 * @note Always returns false since Resource only handles device computing.
	 *       For distributed computing, use MPIResource from MPIBackend.h
	 */
	HOST DEVICE bool is_host() const {
		return type == ResourceType::CPU;
	}
};

#ifdef USE_CUDA
namespace CUDA {
class Metrics {
  public:
	static DeviceMetrics get_device_metrics(int device_id) {
		DeviceMetrics metrics;
		metrics.backend_type = ResourceType::CUDA;
		metrics.device_id = device_id;
		metrics.timestamp = std::chrono::high_resolution_clock::now();

		// Set device context
		int current_device;
		cudaGetDevice(&current_device);
		cudaSetDevice(device_id);

		// Get real-time memory info (includes all processes)
		size_t free, total;
		cudaMemGetInfo(&free, &total);
		metrics.total_memory = total;
		metrics.free_memory = free;
		metrics.memory_utilization =
			static_cast<float>(total - free) / static_cast<float>(total) * 100.0f;

		// Check if device is currently busy (real-time)
		metrics.device_busy = check_device_busy(device_id);

		// Estimate active kernels (real-time check)
		metrics.has_active_kernels = check_active_kernels(device_id);

		// Calculate real-time load score
		metrics.load_score = calculate_real_time_load_score(metrics);

		// Restore device context
		cudaSetDevice(current_device);

		return metrics;
	}

  private:
	static bool check_device_busy(int device_id) {
		// Try to synchronize device - if it takes time, device is busy
		auto start = std::chrono::high_resolution_clock::now();
		cudaError_t result = cudaDeviceSynchronize();
		auto end = std::chrono::high_resolution_clock::now();

		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

		// If synchronization took more than 1ms, device is likely busy
		return (duration.count() > 1000) && (result == cudaSuccess);
	}

	static bool check_active_kernels(int device_id) {
		// Quick check for active kernels by trying to query device
		cudaError_t result = cudaGetLastError();
		if (result == cudaSuccess) {
			// Try a quick kernel launch to see if device is responsive
			// This is a heuristic - if device is busy, this might take longer
			return check_device_busy(device_id);
		}
		return false;
	}

	static float calculate_real_time_load_score(const RealTimeDeviceMetrics& metrics) {
		float memory_score = metrics.memory_utilization / 100.0f;
		float activity_score = metrics.device_busy ? 0.7f : 0.0f;
		float kernel_score = metrics.has_active_kernels ? 0.3f : 0.0f;

		// Weighted combination focusing on real-time state
		return (memory_score * 0.4f + activity_score * 0.4f + kernel_score * 0.2f);
	}
}
}; // namespace CUDA

#endif
#ifdef USE_SYCL
/**
 * @brief SYCL Metrics
 *
 * @details
 * - Maintains device metrics for informed decision making.
 * - Supports periodic metric updates for dynamic load balancing.
 * @note This is a simplified implementation and may not be accurate.
 * SYCL has limited information about device state.
 * @see DeviceMetrics for tracked device statistics.
 */
namespace SYCL {
class Metrics {
  public:
	static DeviceMetrics get_device_metrics(const sycl::device& device, unsigned int device_id) {
		DeviceMetrics metrics;
		metrics.backend_type = ResourceType::SYCL;
		metrics.device_id = device_id;
		metrics.timestamp = std::chrono::high_resolution_clock::now();

		try {
			// Get real-time memory info
			metrics.total_memory = device.get_info<sycl::info::device::global_mem_size>();

			// SYCL doesn't provide free memory, so we estimate based on device responsiveness
			metrics.free_memory = estimate_free_memory(device);
			metrics.memory_utilization =
				static_cast<float>(metrics.total_memory - metrics.free_memory) /
				static_cast<float>(metrics.total_memory) * 100.0f;

			// Check device responsiveness (real-time)
			metrics.device_busy = check_device_responsiveness(device);

			// Check for active kernels (real-time)
			metrics.has_active_kernels = check_kernel_activity(device);

			// Calculate real-time load score
			metrics.load_score = calculate_real_time_load_score(metrics);

		} catch (const sycl::exception& e) {
			LOGWARN("Failed to get SYCL real-time metrics: {}", e.what());
			metrics.load_score = 1.0f; // Assume busy on error
		}

		return metrics;
	}

  private:
	static size_t estimate_free_memory(const sycl::device& device) {
		// Try to allocate a small buffer to test available memory
		// This is a heuristic for shared environments
		try {
			sycl::queue test_queue(device);
			size_t test_size = 1024 * 1024; // 1MB test

			// Try to allocate progressively larger chunks
			for (size_t size = test_size; size <= 1024 * 1024 * 1024; size *= 2) {
				try {
					void* test_ptr = sycl::malloc_device(size, test_queue);
					if (test_ptr) {
						sycl::free(test_ptr, test_queue);
						return size; // Found available memory
					}
				} catch (...) {
					break; // Can't allocate this much
				}
			}
		} catch (...) {
			// Device might be busy
		}

		return 0; // Conservative estimate
	}

	static bool check_device_responsiveness(const sycl::device& device) {
		try {
			sycl::queue test_queue(device);

			// Try to submit a simple task and see how long it takes
			auto start = std::chrono::high_resolution_clock::now();

			auto event = test_queue.submit([&](sycl::handler& h) {
				h.single_task([=]() {
					// Empty kernel to test responsiveness
				});
			});

			// Use SYCL's actual event status checking
			auto status = event.get_info<sycl::info::event::command_execution_status>();
			auto end = std::chrono::high_resolution_clock::now();

			auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

			// If it took more than 1ms or status is not complete, device might be busy
			bool is_busy =
				(duration.count() > 1000) || (status != sycl::info::event_command_status::complete);

			// Wait for completion to clean up
			event.wait();

			return is_busy;

		} catch (const sycl::exception& e) {
			return true; // Assume busy if we can't test
		}
	}

	static bool check_kernel_activity(const sycl::device& device) {
		// SYCL doesn't provide direct kernel activity info
		// We can only infer from device responsiveness
		return check_device_responsiveness(device);
	}

	static float calculate_real_time_load_score(const DeviceMetrics& metrics) {
		float memory_score = metrics.memory_utilization / 100.0f;
		float activity_score = metrics.device_busy ? 0.7f : 0.0f;
		float kernel_score = metrics.has_active_kernels ? 0.3f : 0.0f;

		return (memory_score * 0.4f + activity_score * 0.4f + kernel_score * 0.2f);
	}
};
} // namespace SYCL
#endif

/**
 * @class LoadBalancer
 * @brief Provides device selection strategies for multi-device environments.
 *
 * The LoadBalancer class implements several strategies for distributing computational
 * workloads across multiple devices (CUDA, SYCL, METAL, etc.). It supports round-robin,
 * least-loaded, memory-based, activity-based, and hybrid load balancing policies.
 *
 * @details
 * - Maintains device metrics for informed decision making.
 * - Selects the next device for work submission based on the chosen strategy.
 * - Supports periodic metric updates for dynamic load balancing.
 *
 * @see LoadBalancingStrategy for available strategies.
 * @see DeviceMetrics for tracked device statistics.
 * @see select_device() for device selection logic.
 */
class LoadBalancer {
  public:
	explicit LoadBalancer(LoadBalancingStrategy strategy = LoadBalancingStrategy::HYBRID)
		: strategy_(strategy), current_device_(0) {}

	// Get next device based on strategy
	Resource select_device(const std::vector<Resource>& devices) {
		if (devices.empty()) {
			return Resource{};
		}

		switch (strategy_) {
		case LoadBalancingStrategy::ROUND_ROBIN:
			return select_round_robin(devices);

		case LoadBalancingStrategy::LEAST_LOADED:
			return select_least_loaded(devices);

		case LoadBalancingStrategy::MEMORY_BASED:
			return select_memory_based(devices);

		case LoadBalancingStrategy::ACTIVITY_BASED:
			return select_activity_based(devices);

		case LoadBalancingStrategy::HYBRID:
			return select_hybrid(devices);

		default:
			return devices[0];
		}
	}

	// Update device metrics (call this periodically)
	void update_metrics(const std::vector<Resource>& devices) {
		device_metrics_.clear();
		device_metrics_.reserve(devices.size());

		for (const auto& device : devices) {
			DeviceMetrics metrics;

			switch (device.type) {
			case ResourceType::CUDA:
#ifdef USE_CUDA
				metrics = CUDA::Metrics::get_device_metrics(device.id);
#endif
				break;

			case ResourceType::SYCL:
#ifdef USE_SYCL
				try {
					auto& sycl_device = ARBD::SYCL::Manager::get_device(device.id).get_device();
					metrics = SYCL::Metrics::get_device_metrics(sycl_device, device.id);
				} catch (...) {
					metrics.load_score = 1.0f; // Assume busy on error
				}
#endif
				break;

			default:
				metrics.load_score = 1.0f; // Assume busy for unknown types
				break;
			}

			device_metrics_.push_back(metrics);
		}
	}

	// Get current metrics
	const std::vector<DeviceMetrics>& get_metrics() const {
		return device_metrics_;
	}

	// Set strategy
	void set_strategy(LoadBalancingStrategy strategy) {
		strategy_ = strategy;
	}

  private:
	LoadBalancingStrategy strategy_;
	size_t current_device_;
	std::vector<DeviceMetrics> device_metrics_;

	Resource select_round_robin(const std::vector<Resource>& devices) {
		Resource selected = devices[current_device_ % devices.size()];
		current_device_++;
		return selected;
	}

	Resource select_least_loaded(const std::vector<Resource>& devices) {
		if (device_metrics_.empty()) {
			return devices[0];
		}

		size_t best_device = 0;
		float min_load = std::numeric_limits<float>::max();

		for (size_t i = 0; i < device_metrics_.size(); ++i) {
			if (device_metrics_[i].load_score < min_load) {
				min_load = device_metrics_[i].load_score;
				best_device = i;
			}
		}

		return devices[best_device];
	}

	Resource select_memory_based(const std::vector<Resource>& devices) {
		if (device_metrics_.empty()) {
			return devices[0];
		}

		size_t best_device = 0;
		size_t max_free_memory = 0;

		for (size_t i = 0; i < device_metrics_.size(); ++i) {
			if (device_metrics_[i].free_memory > max_free_memory) {
				max_free_memory = device_metrics_[i].free_memory;
				best_device = i;
			}
		}

		return devices[best_device];
	}

	Resource select_activity_based(const std::vector<Resource>& devices) {
		if (device_metrics_.empty()) {
			return devices[0];
		}

		size_t best_device = 0;
		size_t min_activity = std::numeric_limits<size_t>::max();

		for (size_t i = 0; i < device_metrics_.size(); ++i) {
			size_t activity = device_metrics_[i].active_queues;
			if (activity < min_activity) {
				min_activity = activity;
				best_device = i;
			}
		}

		return devices[best_device];
	}

	Resource select_hybrid(const std::vector<Resource>& devices) {
		if (device_metrics_.empty()) {
			return devices[0];
		}

		// Hybrid approach: consider both memory and activity
		size_t best_device = 0;
		float best_score = std::numeric_limits<float>::max();

		for (size_t i = 0; i < device_metrics_.size(); ++i) {
			const auto& metrics = device_metrics_[i];

			// Normalize memory utilization (lower is better)
			float memory_score = metrics.memory_utilization / 100.0f;

			// Normalize activity (lower is better)
			float activity_score =
				std::min(static_cast<float>(metrics.active_queues) / 10.0f, 1.0f);

			// Combined score (weighted)
			float combined_score = memory_score * 0.6f + activity_score * 0.4f;

			if (combined_score < best_score) {
				best_score = combined_score;
				best_device = i;
			}
		}

		return devices[best_device];
	}
};

} // namespace ARBD