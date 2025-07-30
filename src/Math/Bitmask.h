#pragma once
#include "ARBDException.h"
#include "ARBDLogger.h"
#include "Backend/Buffer.h"
#include "Backend/Resource.h"
#include <algorithm>
#include <cassert>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <map>
#include <memory>
#include <string>

namespace ARBD {
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

// Backend-agnostic atomic operations
namespace detail {
template<typename T>
HOST DEVICE inline void atomic_or(T* addr, T val) {
#ifdef __CUDA_ARCH__
	atomicOr(addr, val);
#else
	*addr |= val;
#endif
}

template<typename T>
HOST DEVICE inline void atomic_and(T* addr, T val) {
#ifdef __CUDA_ARCH__
	atomicAnd(addr, val);
#else
	*addr &= val;
#endif
}
} // namespace detail

// Don't use base because virtual member functions require device malloc
class Bitmask {
	typedef size_t idx_t;
	typedef unsigned int data_t;

  public:
	Bitmask(const idx_t len) : len(len) {
		idx_t tmp = get_array_size();
		assert(tmp * data_stride >= len);
		mask = (tmp > 0) ? new data_t[tmp] : nullptr;
		for (int i = 0; i < tmp; ++i)
			mask[i] = data_t(0);
	}

	// Copy constructor
	Bitmask(const Bitmask& other) : len(other.len) {
		idx_t tmp = get_array_size();
		mask = (tmp > 0) ? new data_t[tmp] : nullptr;
		for (int i = 0; i < tmp; ++i)
			mask[i] = other.mask[i];
	}

	// Assignment operator
	Bitmask& operator=(const Bitmask& other) {
		if (this != &other) {
			// Clean up existing resources
			if (mask != nullptr)
				delete[] mask;

			// Copy from other
			len = other.len;
			idx_t tmp = get_array_size();
			mask = (tmp > 0) ? new data_t[tmp] : nullptr;
			for (int i = 0; i < tmp; ++i)
				mask[i] = other.mask[i];
		}
		return *this;
	}

	// Move constructor
	Bitmask(Bitmask&& other) noexcept : len(other.len), mask(other.mask) {
		other.mask = nullptr;
		other.len = 0;
	}

	// Move assignment operator
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

	~Bitmask() {
		if (mask != nullptr)
			delete[] mask;
	}

	HOST DEVICE idx_t get_len() const {
		return len;
	}

	HOST DEVICE void set_mask(idx_t i, bool value) {
		assert(i < len);
		idx_t ci = i / data_stride;
		data_t change_bit = (data_t(1) << (i - ci * data_stride));

		if (value) {
			detail::atomic_or(&mask[ci], change_bit);
		} else {
			detail::atomic_and(&mask[ci], ~change_bit);
		}
	}

	HOST DEVICE bool get_mask(const idx_t i) const {
		assert(i < len);
		const idx_t ci = i / data_stride;
		return mask[ci] & (data_t(1) << (i - ci * data_stride));
	}

	HOST DEVICE inline bool operator==(const Bitmask& b) const {
		if (len != b.len)
			return false;
		for (idx_t i = 0; i < len; ++i) {
			if (get_mask(i) != b.get_mask(i))
				return false;
		}
		return true;
	}

	HOST Bitmask* send_to_backend(const Resource& resource, Bitmask* device_obj = nullptr) const {
		Bitmask obj_tmp(0);
		data_t* mask_d = nullptr;
		size_t sz = sizeof(data_t) * get_array_size();

		// Allocate device memory for the Bitmask object itself
		if (device_obj == nullptr) {
			device_obj = static_cast<Bitmask*>(ARBD::BackendPolicy::allocate(sizeof(Bitmask)));
		}

		// Allocate and copy mask data if needed
		if (sz > 0) {
			mask_d = static_cast<data_t*>(ARBD::BackendPolicy::allocate(sz));
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

	HOST static void remove_from_backend(Bitmask* device_obj, const Resource& resource) {
		Bitmask obj_tmp(0);

		// Copy the Bitmask object from device to get mask pointer
		ARBD::BackendPolicy::copy_to_host(&obj_tmp, device_obj, sizeof(Bitmask));

		// Free the device mask data if it exists
		if (obj_tmp.len > 0 && obj_tmp.mask != nullptr) {
			ARBD::BackendPolicy::deallocate(obj_tmp.mask);
		}

		// Clear the mask pointer on device (set to nullptr)
		obj_tmp.mask = nullptr;
		ARBD::BackendPolicy::copy_from_host(device_obj, &obj_tmp, sizeof(Bitmask));

		// Free the device Bitmask object itself
		ARBD::BackendPolicy::deallocate(device_obj);
	}

	HOST auto to_string() const {
		std::string s;
		s.reserve(len);
		for (size_t i = 0; i < len; ++i) {
			s += get_mask(i) ? '1' : '0';
		}
		return s;
	}

	HOST DEVICE void print() {
		for (int i = 0; i < len; ++i) {
			printf("%d", (int)get_mask(i));
		}
		printf("\n");
	}

	// Make get_array_size public so it can be used by backend operations
	HOST DEVICE idx_t get_array_size() const {
		return (len == 0) ? 1 : (len - 1) / data_stride + 1;
	}

  private:
	idx_t len;
	const static idx_t data_stride = CHAR_BIT * sizeof(data_t) / sizeof(char);
	data_t* __restrict__ mask;
};

// SparseBitmask implementation for large sparse bit arrays
template<size_t chunk_size = 64>
class SparseBitmask : public BitmaskBase {
  public:
	SparseBitmask(const size_t len)
		: BitmaskBase(len), meta_len((len - 1) / chunk_size + 1), meta_mask(meta_len) {
		static_assert(chunk_size > 0, "Chunk size must be positive");
	}

	~SparseBitmask() = default; // std::map handles cleanup automatically

	HOST DEVICE void set_mask(size_t i, bool value) override {
		assert(i < len);
		size_t chunk_idx = i / chunk_size;
		size_t bit_in_chunk = i % chunk_size;

		// Check if chunk exists
		auto it = chunks.find(chunk_idx);
		if (it == chunks.end()) {
			// Chunk doesn't exist
			if (value) {
				// Need to create chunk
				meta_mask.set_mask(chunk_idx, true);
				// Create the specific chunk we need
				auto result = chunks.emplace(chunk_idx, Bitmask(chunk_size));
				it = result.first; // Get iterator to the newly inserted element
			} else {
				// Setting to false in non-existent chunk - nothing to do
				return;
			}
		}

		// Set the bit in the appropriate chunk
		it->second.set_mask(bit_in_chunk, value);

		// If we're clearing the last bit in a chunk, we could potentially remove it
		// For simplicity, we'll keep allocated chunks (optimization opportunity)
	}

	HOST DEVICE bool get_mask(size_t i) const override {
		assert(i < len);
		size_t chunk_idx = i / chunk_size;
		size_t bit_in_chunk = i % chunk_size;

		// Get the bit from the appropriate chunk
		auto it = chunks.find(chunk_idx);
		if (it != chunks.end()) {
			return it->second.get_mask(bit_in_chunk);
		}

		return false; // Chunk doesn't exist, so bit is false
	}

	size_t get_allocated_chunks() const {
		return chunks.size();
	}

	size_t get_meta_len() const {
		return meta_len;
	}

	// Convert meta index to actual bit index
	size_t meta_idx_to_bit_idx(size_t meta_idx) const {
		return meta_idx * chunk_size;
	}

  private:
	size_t meta_len;
	Bitmask meta_mask;				  // Tracks which chunks are allocated
	std::map<size_t, Bitmask> chunks; // Actual bit chunks (sparse storage)
};

} // namespace ARBD
