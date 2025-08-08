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

#ifndef __METAL_VERSION__
#include "ARBDException.h"
#include <cstddef>
#include <string>
#endif

namespace ARBD {

/**
 * @brief Enumeration of supported compute resource types.
 */
enum class ResourceType : uint8_t {
	CPU = 0,  ///< CPU resource (default)
	CUDA = 1, ///< NVIDIA CUDA GPU
	SYCL = 2, ///< SYCL-compatible device
	METAL = 3 ///< Apple Metal GPU
};

/**
 * @brief A production-ready resource identifier that explicitly specifies
 * compute devices for thread-safe, multi-GPU operations.
 *
 * This class eliminates the global state dependency that causes race conditions
 * in multi-threaded environments by requiring explicit resource specification
 * for all memory and compute operations.
 */
class Resource {
  public:
	ResourceType type{ResourceType::CPU}; ///< Resource type (defaults to CPU)
	size_t id{0};						  ///< Device ID within the resource type

	/**
	 * @brief Default constructor creates a CPU resource.
	 */
	HOST DEVICE constexpr Resource() = default;

	/**
	 * @brief Construct a resource with specified type and optional ID.
	 *
	 * @param resource_type The type of compute resource
	 * @param device_id The device ID within that resource type (defaults to 0)
	 */
	HOST DEVICE constexpr Resource(ResourceType resource_type, size_t device_id = 0)
		: type(resource_type), id(device_id) {}

	/**
	 * @brief Check if this resource is valid (always true since CPU is always available).
	 */
	HOST DEVICE constexpr bool is_valid() const {
		return true; // CPU is always available, so all resources are valid
	}

	/**
	 * @brief Get a human-readable string for the resource type.
	 */
	HOST DEVICE constexpr const char* getTypeString() const {
		switch (type) {
		case ResourceType::CPU:
			return "CPU, refrain from using this";
		case ResourceType::CUDA:
			return "CUDA";
		case ResourceType::SYCL:
			return "SYCL";
		case ResourceType::METAL:
			return "METAL";
		default:
			return "No Device Selected"; // Fallback to CPU
		}
	}

	/**
	 * @brief Check if this resource is currently active/set as the current device.
	 *
	 * This method provides a way to verify that operations will execute on the
	 * intended device, which is crucial for debugging and validation in
	 * multi-device environments.
	 *
	 * @return true if this resource matches the currently active device context
	 */
	HOST DEVICE bool is_current() const {
#if defined(__CUDA_ARCH__) || defined(__SYCL_DEVICE_ONLY__) || defined(__METAL_VERSION__)
		// On device: assume we're on the right device if code is executing
		// A more robust implementation might be needed for cross-device kernels
		return true;
#else
		// Host-side validation
		if (type == ResourceType::CPU) {
			return true; // CPU is always "current" on host
		}

		bool is_current_device = false;

#ifdef USE_CUDA
		if (type == ResourceType::CUDA) {
			int current_device;
			if (cudaGetDevice(&current_device) == cudaSuccess) {
				is_current_device = (current_device == static_cast<int>(id));
			}
		}
#endif

#ifdef USE_SYCL
		if (type == ResourceType::SYCL) {
			try {
				auto& current_device = ARBD::SYCL::Manager::get_current_device();
				is_current_device = (current_device.id() == id);
			} catch (...) {
				is_current_device = false;
			}
		}
#endif

#ifdef USE_METAL
		if (type == ResourceType::METAL) {
			try {
				auto& current_device = ARBD::METAL::Manager::get_current_device();
				is_current_device = (current_device.id() == id);
			} catch (...) {
				is_current_device = false;
			}
		}
#endif

		return is_current_device;
#endif
	}

	/**
	 * @brief Create a Resource representing the currently active device.
	 *
	 * This method queries the backend to determine which device is currently
	 * active and returns a Resource representing that device. This is useful
	 * for creating buffers that should be allocated on "whatever device is
	 * currently active" while still maintaining explicit resource tracking.
	 *
	 * @return Resource representing the currently active device
	 */
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
			// Fall through to CPU default
		}
#endif

#ifdef USE_METAL
		try {
			auto& current_device = ARBD::METAL::Manager::get_current_device();
			return Resource{ResourceType::METAL, static_cast<size_t>(current_device.id())};
		} catch (...) {
			// Fall through to CPU default
		}
#endif

		// Default to CPU if no device context is active
		return Resource{ResourceType::CPU, 0};
	}

	/**
	 * @brief Create a CPU resource.
	 *
	 * @param cpu_id The CPU ID (typically 0 for single-socket systems)
	 * @return Resource representing the specified CPU
	 */
	static constexpr Resource CPU(size_t cpu_id = 0) {
		return Resource{ResourceType::CPU, cpu_id};
	}

#ifdef USE_CUDA
	/**
	 * @brief Create a CUDA resource.
	 *
	 * @param device_id The CUDA device ID (defaults to 0)
	 * @return Resource representing the specified CUDA device
	 */
	static constexpr Resource CUDA(size_t device_id = 0) {
		return Resource{ResourceType::CUDA, device_id};
	}
#endif

#ifdef USE_SYCL
	/**
	 * @brief Create a SYCL resource.
	 *
	 * @param device_id The SYCL device ID (defaults to 0)
	 * @return Resource representing the specified SYCL device
	 */
	static constexpr Resource SYCL(size_t device_id = 0) {
		return Resource{ResourceType::SYCL, device_id};
	}
#endif

#ifdef USE_METAL
	/**
	 * @brief Create a Metal resource.
	 *
	 * @param device_id The Metal device ID (defaults to 0)
	 * @return Resource representing the specified Metal device
	 */
	static constexpr Resource METAL(size_t device_id = 0) {
		return Resource{ResourceType::METAL, device_id};
	}
#endif

	/**
	 * @brief Equality comparison operator.
	 */
	HOST DEVICE constexpr bool operator==(const Resource& other) const {
		return type == other.type && id == other.id;
	}

	/**
	 * @brief Inequality comparison operator.
	 */
	HOST DEVICE constexpr bool operator!=(const Resource& other) const {
		return !(*this == other);
	}

	/**
	 * @brief Less-than comparison for use in containers.
	 */
	HOST DEVICE constexpr bool operator<(const Resource& other) const {
		if (type != other.type) {
			return static_cast<uint8_t>(type) < static_cast<uint8_t>(other.type);
		}
		return id < other.id;
	}

	/**
	 * @brief Get a string representation of this resource.
	 *
	 * @return String in format "TYPE[ID]" (e.g., "CUDA[0]", "METAL[1]")
	 */
	std::string toString() const {
		return std::string(getTypeString()) + "[" + std::to_string(id) + "]";
	}

	/**
	 * @brief Check if the resource supports asynchronous operations.
	 */
	HOST DEVICE constexpr bool supports_async() const {
		switch (type) {
		case ResourceType::CUDA:
		case ResourceType::SYCL:
		case ResourceType::METAL:
			return true;
		case ResourceType::CPU:
		default:
			return false;
		}
	}

	/**
	 * @brief Get the memory space type for this resource.
	 */
	HOST DEVICE constexpr const char* getMemorySpace() const {
		switch (type) {
		case ResourceType::CPU:
			return "host";
		case ResourceType::CUDA:
		case ResourceType::SYCL:
		case ResourceType::METAL:
			return "device";
		default:
			return "host"; // Fallback to host
		}
	}

	/**
	 * @brief Check if this resource represents a device (GPU).
	 */
	HOST DEVICE constexpr bool is_device() const {
		return type == ResourceType::CUDA || type == ResourceType::SYCL ||
			   type == ResourceType::METAL;
	}

	/**
	 * @brief Check if this resource represents a host (CPU).
	 */
	HOST DEVICE constexpr bool is_host() const {
		return type == ResourceType::CPU;
	}

	/**
	 * @brief Check if this resource can access peer memory from another resource.
	 *
	 * This method helps determine if direct device-to-device memory transfers
	 * are possible between two resources, which is important for optimization
	 * in multi-GPU scenarios.
	 *
	 * @param other The other resource to check peer access with
	 * @return true if peer access is supported and enabled
	 */
	bool can_access_peer(const Resource& other) const {
		// Same resource can always access itself
		if (*this == other) {
			return true;
		}

		// CPU can access all resources (through host memory)
		if (type == ResourceType::CPU || other.type == ResourceType::CPU) {
			return true;
		}

		// Cross-backend peer access is not supported
		if (type != other.type) {
			return false;
		}

#ifdef USE_CUDA
		if (type == ResourceType::CUDA) {
			try {
				return ARBD::CUDA::Manager::can_access_peer(static_cast<int>(id),
															static_cast<int>(other.id));
			} catch (...) {
				return false;
			}
		}
#endif

		// For SYCL and Metal, assume no peer access for now
		// This could be extended with backend-specific peer access queries
		return false;
	}

	/**
	 * @brief Validate that this resource exists and is accessible.
	 *
	 * This method performs runtime validation to ensure the resource
	 * represents a real, accessible device. Useful for debugging and
	 * error checking in production code.
	 *
	 * @throws ARBDException if the resource is invalid or inaccessible
	 */
	void validate() const {
#if !defined(__METAL_VERSION__) && !defined(__CUDA_ARCH__) && !defined(__SYCL_DEVICE_ONLY__)
		if (type == ResourceType::CPU) {
			// CPU resources are always valid
			return;
		}

#ifdef USE_CUDA
		if (type == ResourceType::CUDA) {
			int device_count;
			if (cudaGetDeviceCount(&device_count) != cudaSuccess ||
				static_cast<int>(id) >= device_count) {
				ARBD_Exception(
					ExceptionType::ValueError,
					"Resource validation failed: CUDA device {} does not exist (count: {})",
					id,
					device_count);
			}
			return;
		}
#endif

#ifdef USE_SYCL
		if (type == ResourceType::SYCL) {
			try {
				auto& device_manager = ARBD::SYCL::Manager::get_device(id);
				// If we can get the device, it exists
				return;
			} catch (...) {
				ARBD_Exception(ExceptionType::ValueError,
							   "Resource validation failed: SYCL device {} does not exist",
							   id);
			}
		}
#endif

#ifdef USE_METAL
		if (type == ResourceType::METAL) {
			try {
				auto& device_manager = ARBD::METAL::Manager::get_device(id);
				// If we can get the device, it exists
				return;
			} catch (...) {
				ARBD_Exception(ExceptionType::ValueError,
							   "Resource validation failed: Metal device {} does not exist",
							   id);
			}
		}
#endif

		ARBD_Exception(ExceptionType::ValueError,
					   "Resource validation failed: Unsupported resource type {}",
					   static_cast<int>(type));
#endif
	}

	/**
	 * @brief Get the compute capability or performance tier of this resource.
	 *
	 * Returns a relative performance indicator where higher numbers indicate
	 * more capable devices. Useful for load balancing and device selection.
	 *
	 * @return Performance tier (0 = lowest, higher = better)

	uint32_t get_compute_capability() const {
#if !defined(__METAL_VERSION__) && !defined(__CUDA_ARCH__) && !defined(__SYCL_DEVICE_ONLY__)
		if (type == ResourceType::CPU) {
			return 1; // Baseline performance
		}

#ifdef USE_CUDA
		if (type == ResourceType::CUDA) {
			try {
				cudaDeviceProp prop;
				if (cudaGetDeviceProperties(&prop, static_cast<int>(id)) == cudaSuccess) {
					// Combine major and minor compute capability
					return static_cast<uint32_t>(prop.major * 10 + prop.minor);
				}
			} catch (...) {
				// Fall through to default
			}
		}
#endif

		// For SYCL and Metal, return a generic "device" tier for now
		// This could be extended with backend-specific capability queries
		return 10; // Higher than CPU but lower than modern CUDA
#else
		return 0; // Device code doesn't have access to capability queries
#endif
	}
	*/
};

} // namespace ARBD
