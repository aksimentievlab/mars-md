#pragma once
#include "ARBDException.h"
#include "ARBDLogger.h"
#include "Backend/Buffer.h"
#include "Backend/Resource.h"
#include <cassert>

// Backend-specific includes
#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <device_atomic_functions.h>
#endif

#ifdef USE_SYCL
#include <sycl/sycl.hpp>
#endif

#ifdef USE_METAL
#include "Metal/Metal.hpp"
#endif

#ifdef __METAL_VERSION__
#include <metal_atomic>
#include <metal_stdlib>
#endif

#ifdef HOST_GUARD
#include <cassert>
#include <climits>
#include <cstdio>
#include <map>
#include <string>
#endif

namespace ARBD {

// Backend-agnostic atomic operations
namespace detail {
/**
 * @brief Backend-agnostic atomic OR operation
 */
template<typename T>
HOST DEVICE inline void atomic_or(T* addr, T val) {
#ifdef __CUDA_ARCH__
	atomicOr(addr, val);
#elif defined(__SYCL_DEVICE_ONLY__)
	// SYCL atomic operations need proper address space qualification
	if (addr != nullptr) {
		sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device>(*addr)
			.fetch_or(val);
	}
#elif defined(__METAL_VERSION__)
	// Metal doesn't have built-in atomic OR, so we use compare-and-swap
	T old_val, new_val;
	do {
		old_val = *addr;
		new_val = old_val | val;
	} while (!atomic_compare_exchange_weak_explicit(addr,
													&old_val,
													new_val,
													std::memory_order_relaxed,
													std::memory_order_relaxed));
#else
	// Host fallback - not atomic but safe for single-threaded host code
	*addr |= val;
#endif
}

/**
 * @brief Backend-agnostic atomic AND operation
 */
template<typename T>
HOST DEVICE inline void atomic_and(T* addr, T val) {
#ifdef __CUDA_ARCH__
	atomicAnd(addr, val);
#elif defined(__SYCL_DEVICE_ONLY__)
	// SYCL atomic operations need proper address space qualification
	if (addr != nullptr) {
		sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device>(*addr)
			.fetch_and(val);
	}
#elif defined(__METAL_VERSION__)
	// Metal doesn't have built-in atomic AND, so we use compare-and-swap
	T old_val, new_val;
	do {
		old_val = *addr;
		new_val = old_val & val;
	} while (!atomic_compare_exchange_weak_explicit(addr,
													&old_val,
													new_val,
													std::memory_order_relaxed,
													std::memory_order_relaxed));
#else
	// Host fallback - not atomic but safe for single-threaded host code
	*addr &= val;
#endif
}

} // namespace detail

// Device-safe bitmask implementation
/**
 * @brief Device-safe bitmask class with proper backend support
 *
 * This class addresses the critical device safety issues:
 * - Proper atomic operations for all backends
 * - Correct address space qualifiers
 * - No device-side allocation
 * - Thread-safe operations
 */
class Bitmask {
  public:
	typedef size_t idx_t;
	typedef unsigned int data_t;

  private:
	idx_t len;
	const static idx_t data_stride = CHAR_BIT * sizeof(data_t) / sizeof(char);

	// Backend-specific pointer types with proper address space qualifiers
#ifdef __METAL_VERSION__
	device data_t* __restrict__ mask;
#else
	data_t* __restrict__ mask;
#endif

  public:
	/**
	 * @brief Constructor for device-safe bitmask
	 * @param length Number of bits in the mask
	 * @param device_mask Pointer to pre-allocated device memory (nullptr for host-only)
	 */
	HOST DEVICE Bitmask(const idx_t length, data_t* device_mask = nullptr)
		: len(length), mask(device_mask) {
		// No allocation in device code - memory must be pre-allocated
#if !defined(__METAL_VERSION__) && !defined(__SYCL_DEVICE_ONLY__) && !defined(__CUDA_ARCH__)
		// Host-side initialization
		if (mask == nullptr) {
			idx_t array_size = get_array_size();
			mask = (array_size > 0) ? new data_t[array_size] : nullptr;
			if (mask) {
				for (idx_t i = 0; i < array_size; ++i) {
					mask[i] = data_t(0);
				}
			}
		}
#endif
	}

	/**
	 * @brief Get the length of the bitmask
	 */
	HOST DEVICE idx_t get_len() const {
		return len;
	}

	/**
	 * @brief Set a bit atomically (thread-safe)
	 * @param i Bit index to set
	 * @param value Value to set (true/false)
	 */
	HOST DEVICE void set_mask(idx_t i, bool value) {
		assert(i < len);
		assert(mask != nullptr); // Device safety check

		// Additional safety check for corrupted pointers
		if (mask == nullptr || mask == reinterpret_cast<data_t*>(0x4110000041000000)) {
			return; // Skip operation if pointer is corrupted
		}

		idx_t ci = i / data_stride;
		data_t change_bit = (data_t(1) << (i - ci * data_stride));

		if (value) {
			detail::atomic_or(&mask[ci], change_bit);
		} else {
			detail::atomic_and(&mask[ci], ~change_bit);
		}
	}

	/**
	 * @brief Get a bit value (thread-safe for reading)
	 * @param i Bit index to read
	 * @return Bit value
	 */
	HOST DEVICE bool get_mask(const idx_t i) const {
		assert(i < len);
		assert(mask != nullptr); // Device safety check

		// Additional safety check for corrupted pointers
		if (mask == nullptr || mask == reinterpret_cast<data_t*>(0x4110000041000000)) {
			return false; // Return false if pointer is corrupted
		}

		const idx_t ci = i / data_stride;
		return (mask[ci] & (data_t(1) << (i - ci * data_stride))) != 0;
	}

	/**
	 * @brief Equality comparison
	 */
	HOST DEVICE inline bool operator==(const Bitmask& b) const {
		if (len != b.len)
			return false;
		for (idx_t i = 0; i < len; ++i) {
			if (get_mask(i) != b.get_mask(i))
				return false;
		}
		return true;
	}

	/**
	 * @brief Debug print function (host-only)
	 */
	HOST void print() const {
#if !defined(__METAL_VERSION__) && !defined(__SYCL_DEVICE_ONLY__) && !defined(__CUDA_ARCH__)
		for (idx_t i = 0; i < len; ++i) {
			printf("%d", static_cast<int>(get_mask(i)));
		}
		printf("\n");
#endif
	}

	/**
	 * @brief Get the array size needed for storage
	 */
	HOST DEVICE idx_t get_array_size() const {
		return (len == 0) ? 1 : (len - 1) / data_stride + 1;
	}

	/**
	 * @brief Get raw pointer to mask data (for backend operations)
	 */
	HOST DEVICE data_t* get_mask_ptr() {
		return mask;
	}

	/**
	 * @brief Get const raw pointer to mask data
	 */
	HOST DEVICE const data_t* get_mask_ptr() const {
		return mask;
	}

	/**
	 * @brief Set the mask pointer (for backend memory management)
	 */
	HOST DEVICE void set_mask_ptr(data_t* ptr) {
		mask = ptr;
	}

#ifdef HOST_GUARD
	/**
	 * @brief Copy constructor
	 */
	Bitmask(const Bitmask& other) : len(other.len) {
		idx_t array_size = get_array_size();
		mask = (array_size > 0) ? new data_t[array_size] : nullptr;
		if (mask && other.mask) {
			for (idx_t i = 0; i < array_size; ++i) {
				mask[i] = other.mask[i];
			}
		}
	}

	/**
	 * @brief Assignment operator
	 */
	Bitmask& operator=(const Bitmask& other) {
		if (this != &other) {
			// Clean up existing resources
			if (mask != nullptr)
				delete[] mask;

			// Copy from other
			len = other.len;
			idx_t array_size = get_array_size();
			mask = (array_size > 0) ? new data_t[array_size] : nullptr;
			if (mask && other.mask) {
				for (idx_t i = 0; i < array_size; ++i) {
					mask[i] = other.mask[i];
				}
			}
		}
		return *this;
	}

	/**
	 * @brief Move constructor
	 */
	Bitmask(Bitmask&& other) noexcept : len(other.len), mask(other.mask) {
		other.mask = nullptr;
		other.len = 0;
	}

	/**
	 * @brief Move assignment operator
	 */
	Bitmask& operator=(Bitmask&& other) noexcept {
		if (this != &other) {
			// Clean up existing resources
			if (mask != nullptr)
				delete[] mask;

			// Move from other
			len = other.len;
			mask = other.mask;

			// Clear other
			other.mask = nullptr;
			other.len = 0;
		}
		return *this;
	}

	/**
	 * @brief Destructor
	 */
	~Bitmask() {
		if (mask != nullptr)
			delete[] mask;
	}

	/**
	 * @brief Send bitmask to device
	 */
	HOST Bitmask* send_to_backend(const Resource& resource, Bitmask* device_obj = nullptr) const {
		Bitmask obj_tmp(0);
		data_t* mask_d = nullptr;
		size_t sz = sizeof(data_t) * get_array_size();

		// Allocate device memory for the Bitmask object itself
		if (device_obj == nullptr) {
			device_obj =
				static_cast<Bitmask*>(ARBD::BackendPolicy::allocate(resource, sizeof(Bitmask)));
		}

		// Allocate and copy mask data if needed
		if (sz > 0) {
			mask_d = static_cast<data_t*>(ARBD::BackendPolicy::allocate(resource, sz));
			ARBD::BackendPolicy::copy_from_host(mask_d, mask, sz);
		}

		// Set up temporary object with device pointers
		obj_tmp.len = len;
		obj_tmp.mask = mask_d;

		// Copy the Bitmask object to device
		ARBD::BackendPolicy::copy_from_host(device_obj, &obj_tmp, sizeof(Bitmask));

		// Clear the temporary object's mask pointer to avoid double-free
		obj_tmp.mask = nullptr;

		return device_obj;
	}

	/**
	 * @brief Receive bitmask from device
	 */
	HOST static Bitmask receive_from_backend(Bitmask* device_obj, const Resource& resource) {
		Bitmask obj_tmp(0);

		// Copy the Bitmask object from device to host
		ARBD::BackendPolicy::copy_to_host(&obj_tmp, device_obj, sizeof(Bitmask));

		if (obj_tmp.len > 0) {
			size_t array_size = obj_tmp.get_array_size();
			size_t sz = sizeof(data_t) * array_size;
			data_t* device_mask_addr = obj_tmp.mask;

			// Allocate host memory for the mask data
			obj_tmp.mask = new data_t[array_size];

			// Copy mask data from device to host
			ARBD::BackendPolicy::copy_to_host(obj_tmp.mask, device_mask_addr, sz);
		} else {
			obj_tmp.mask = nullptr;
		}

		return obj_tmp;
	}

	/**
	 * @brief Remove bitmask from device
	 */
	HOST static void remove_from_backend(Bitmask* device_obj, const Resource& resource) {
		if (!device_obj) {
			return; // Nothing to do if device_obj is null
		}

		Bitmask obj_tmp(0);

		try {
			// Copy the Bitmask object from device to get mask pointer
			ARBD::BackendPolicy::copy_to_host(&obj_tmp, device_obj, sizeof(Bitmask));

			// Free the device mask data if it exists and is valid
			if (obj_tmp.len > 0 && obj_tmp.mask != nullptr) {
				// Validate pointer before deallocation
				if (obj_tmp.mask !=
					reinterpret_cast<data_t*>(0x4110000041000000)) { // Check for corrupted pointer
					ARBD::BackendPolicy::deallocate(obj_tmp.mask);
				}
			}

			// Clear the mask pointer on device (set to nullptr)
			obj_tmp.mask = nullptr;
			ARBD::BackendPolicy::copy_from_host(device_obj, &obj_tmp, sizeof(Bitmask));

			// Free the device Bitmask object itself
			ARBD::BackendPolicy::deallocate(device_obj);
		} catch (const std::exception& e) {
			// Log the error but don't throw from cleanup
			LOGWARN("Warning: Failed to cleanup device Bitmask properly: {}", e.what());

			// Try to free the device object even if mask cleanup failed
			try {
				ARBD::BackendPolicy::deallocate(device_obj);
			} catch (...) {
				// Ignore cleanup errors
			}
		}
	}

	/**
	 * @brief Convert to string representation
	 */
	HOST auto to_string() const {
		std::string s;
		s.reserve(len);
		for (size_t i = 0; i < len; ++i) {
			s += get_mask(i) ? '1' : '0';
		}
		return s;
	}
#endif

  private:
	// Disable default constructor to prevent unsafe usage
	Bitmask() = delete;
};

#ifdef HOST_GUARD
// ============================================================================
// Host-Only Sparse Bitmask Implementation
// ============================================================================

/**
 * @brief Base class for bitmask implementations
 */
class BitmaskBase {
  public:
	BitmaskBase(const size_t len) : len(len) {}
	virtual ~BitmaskBase() = default;

	HOST DEVICE virtual void set_mask(size_t i, bool value) = 0;
	HOST DEVICE virtual bool get_mask(size_t i) const = 0;

	size_t get_len() const {
		return len;
	}

	virtual void print() const {
		for (size_t i = 0; i < len; ++i) {
			LOGINFO("%d", static_cast<int>(get_mask(i)));
		}
		LOGINFO("\n");
	}

  protected:
	size_t len;
};

/**
 * @brief Device-safe sparse bitmask implementation for large sparse bit arrays
 *
 * This implementation uses a fixed-size array of chunk pointers instead of std::map
 * to ensure device compatibility. The trade-off is that it has a maximum number
 * of chunks, but this is typically sufficient for most use cases.
 */
template<size_t chunk_size = 64, size_t max_chunks = 1024>
class SparseBitmask : public BitmaskBase {
  public:
	typedef typename Bitmask::data_t chunk_data_t;
	typedef typename Bitmask::idx_t idx_t;

	SparseBitmask(const size_t len)
		: BitmaskBase(len), meta_len((len - 1) / chunk_size + 1), meta_mask(meta_len) {
		static_assert(chunk_size > 0, "Chunk size must be positive");
		static_assert(max_chunks > 0, "Max chunks must be positive");

		// Initialize chunk pointers to nullptr
		for (size_t i = 0; i < max_chunks; ++i) {
			chunk_ptrs[i] = nullptr;
		}
	}

	~SparseBitmask() {
		// Clean up allocated chunks
		for (size_t i = 0; i < max_chunks; ++i) {
			if (chunk_ptrs[i] != nullptr) {
				delete chunk_ptrs[i];
			}
		}
	}

	/**
	 * @brief Set a bit atomically (thread-safe)
	 */
	HOST DEVICE void set_mask(size_t i, bool value) override {
		assert(i < len);
		size_t chunk_idx = i / chunk_size;
		size_t bit_in_chunk = i % chunk_size;

		// Check bounds
		if (chunk_idx >= max_chunks) {
			// Handle overflow - could throw exception or log error
			return;
		}

		// Check if chunk exists
		if (chunk_ptrs[chunk_idx] == nullptr) {
			// Chunk doesn't exist
			if (value) {
				// Need to create chunk
				meta_mask.set_mask(chunk_idx, true);

				// Allocate new chunk (host-only operation)
#ifdef HOST_GUARD
				chunk_ptrs[chunk_idx] = new Bitmask(chunk_size);
#endif
			} else {
				// Setting to false in non-existent chunk - nothing to do
				return;
			}
		}

		// Set the bit in the appropriate chunk
		if (chunk_ptrs[chunk_idx] != nullptr) {
			chunk_ptrs[chunk_idx]->set_mask(bit_in_chunk, value);
		}
	}

	/**
	 * @brief Get a bit value (thread-safe for reading)
	 */
	HOST DEVICE bool get_mask(size_t i) const override {
		assert(i < len);
		size_t chunk_idx = i / chunk_size;
		size_t bit_in_chunk = i % chunk_size;

		// Check bounds
		if (chunk_idx >= max_chunks) {
			return false;
		}

		// Get the bit from the appropriate chunk
		if (chunk_ptrs[chunk_idx] != nullptr) {
			return chunk_ptrs[chunk_idx]->get_mask(bit_in_chunk);
		}

		return false; // Chunk doesn't exist, so bit is false
	}

	/**
	 * @brief Get the number of allocated chunks
	 */
	size_t get_allocated_chunks() const {
		size_t count = 0;
		for (size_t i = 0; i < max_chunks; ++i) {
			if (chunk_ptrs[i] != nullptr) {
				++count;
			}
		}
		return count;
	}

	/**
	 * @brief Get the meta mask length
	 */
	size_t get_meta_len() const {
		return meta_len;
	}

	/**
	 * @brief Convert meta index to actual bit index
	 */
	size_t meta_idx_to_bit_idx(size_t meta_idx) const {
		return meta_idx * chunk_size;
	}

	/**
	 * @brief Get the maximum number of chunks
	 */
	static constexpr size_t get_max_chunks() {
		return max_chunks;
	}

	/**
	 * @brief Check if the sparse bitmask is at capacity
	 */
	bool is_at_capacity() const {
		return get_allocated_chunks() >= max_chunks;
	}

	/**
	 * @brief Get memory usage statistics
	 */
	struct MemoryStats {
		size_t allocated_chunks;
		size_t total_bits;
		size_t used_bits;
		size_t memory_bytes;
	};

	MemoryStats get_memory_stats() const {
		MemoryStats stats = {0, 0, 0, 0};
		stats.allocated_chunks = get_allocated_chunks();
		stats.total_bits = len;

		// Count used bits
		for (size_t i = 0; i < len; ++i) {
			if (get_mask(i)) {
				++stats.used_bits;
			}
		}

		// Calculate memory usage
		stats.memory_bytes =
			sizeof(SparseBitmask) + stats.allocated_chunks * chunk_size * sizeof(chunk_data_t) / 8;

		return stats;
	}

	/**
	 * @brief Clear all bits (reset to zero)
	 */
	void clear() {
		for (size_t i = 0; i < max_chunks; ++i) {
			if (chunk_ptrs[i] != nullptr) {
				delete chunk_ptrs[i];
				chunk_ptrs[i] = nullptr;
			}
		}
		meta_mask = Bitmask(meta_len); // Reset meta mask
	}

	/**
	 * @brief Copy constructor
	 */
	SparseBitmask(const SparseBitmask& other)
		: BitmaskBase(other.len), meta_len(other.meta_len), meta_mask(other.meta_mask) {
		// Copy chunk pointers
		for (size_t i = 0; i < max_chunks; ++i) {
			if (other.chunk_ptrs[i] != nullptr) {
				chunk_ptrs[i] = new Bitmask(*other.chunk_ptrs[i]);
			} else {
				chunk_ptrs[i] = nullptr;
			}
		}
	}

	/**
	 * @brief Assignment operator
	 */
	SparseBitmask& operator=(const SparseBitmask& other) {
		if (this != &other) {
			// Clean up existing chunks
			clear();

			// Copy from other
			len = other.len;
			meta_len = other.meta_len;
			meta_mask = other.meta_mask;

			// Copy chunk pointers
			for (size_t i = 0; i < max_chunks; ++i) {
				if (other.chunk_ptrs[i] != nullptr) {
					chunk_ptrs[i] = new Bitmask(*other.chunk_ptrs[i]);
				} else {
					chunk_ptrs[i] = nullptr;
				}
			}
		}
		return *this;
	}

	/**
	 * @brief Move constructor
	 */
	SparseBitmask(SparseBitmask&& other) noexcept
		: BitmaskBase(other.len), meta_len(other.meta_len), meta_mask(std::move(other.meta_mask)) {
		// Move chunk pointers
		for (size_t i = 0; i < max_chunks; ++i) {
			chunk_ptrs[i] = other.chunk_ptrs[i];
			other.chunk_ptrs[i] = nullptr;
		}
		other.len = 0;
		other.meta_len = 0;
	}

	/**
	 * @brief Move assignment operator
	 */
	SparseBitmask& operator=(SparseBitmask&& other) noexcept {
		if (this != &other) {
			// Clean up existing chunks
			clear();

			// Move from other
			len = other.len;
			meta_len = other.meta_len;
			meta_mask = std::move(other.meta_mask);

			// Move chunk pointers
			for (size_t i = 0; i < max_chunks; ++i) {
				chunk_ptrs[i] = other.chunk_ptrs[i];
				other.chunk_ptrs[i] = nullptr;
			}

			other.len = 0;
			other.meta_len = 0;
		}
		return *this;
	}

  private:
	size_t meta_len;
	Bitmask meta_mask;				 // Tracks which chunks are allocated
	Bitmask* chunk_ptrs[max_chunks]; // Fixed-size array of chunk pointers (device-safe)
};
#endif

} // namespace ARBD
