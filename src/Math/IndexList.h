/*********************************************************************
 * @file  IndexList.h
 *
 * @brief Device-safe IndexList for CUDA/SYCL/CPU compatibility
 *
 * @author V2: Pin-Yi Li <pinyili2@illinois.edu>
 *********************************************************************/
#pragma once

#if !defined(__METAL_VERSION__) && !defined(__SYCL_DEVICE_ONLY__) && !defined(__CUDA_ARCH__)
#include "ARBDException.h"
#include "ARBDLogger.h"
#include <algorithm>
#include <initializer_list>
#include <string>
#endif

#include "Backend/Header.h"

namespace ARBD {

/**
 * @brief Device-safe IndexList with fixed maximum capacity
 *
 * This replaces the std::vector-based implementation to work on GPU devices.
 * Uses stack-allocated array with compile-time maximum size.
 *
 * @tparam T Index type (typically int, idx_t)
 * @tparam MaxSize Maximum number of indices that can be stored
 */
template<typename T = int, idx_t MaxSize = 32>
class IndexList {
  public:
	using value_type = T;
	using idx_type = idx_t;

	static_assert(MaxSize > 0, "IndexList MaxSize must be positive");

  private:
	T data_[MaxSize]; ///< Fixed-size array for device compatibility
	idx_t size_ = 0; ///< Current number of elements

  public:
	/*===================*\
	|  CONSTRUCTORS       |
	\*===================*/

	HOST DEVICE constexpr IndexList() = default;

	HOST DEVICE constexpr IndexList(const IndexList& other) : size_(other.size_) {
		for (idx_t i = 0; i < size_; ++i) {
			data_[i] = other.data_[i];
		}
	}

	HOST DEVICE constexpr IndexList& operator=(const IndexList& other) {
		if (this != &other) {
			size_ = other.size_;
			for (idx_t i = 0; i < size_; ++i) {
				data_[i] = other.data_[i];
			}
		}
		return *this;
	}

#if !defined(__CUDA_ARCH__) && !defined(__SYCL_DEVICE_ONLY__)
	/**
	 * @brief Host-only constructor from initializer list
	 */
	HOST IndexList(std::initializer_list<T> init) : size_(0) {
		for (const auto& value : init) {
			if (size_ < MaxSize) {
				data_[size_++] = value;
			}
		}
	}
#endif

	/*===================*\
	|  CAPACITY           |
	\*===================*/

	HOST DEVICE constexpr idx_t size() const noexcept {
		return size_;
	}
	HOST DEVICE constexpr idx_t length() const noexcept {
		return size_;
	}
	HOST DEVICE constexpr bool empty() const noexcept {
		return size_ == 0;
	}
	HOST DEVICE constexpr idx_t capacity() const noexcept {
		return MaxSize;
	}
	HOST DEVICE constexpr idx_t max_size() const noexcept {
		return MaxSize;
	}
	HOST DEVICE constexpr bool full() const noexcept {
		return size_ == MaxSize;
	}

	/*===================*\
	|  ELEMENT ACCESS     |
	\*===================*/

	HOST DEVICE constexpr T& operator[](idx_t i) noexcept {
		return data_[i];
	}
	HOST DEVICE constexpr const T& operator[](idx_t i) const noexcept {
		return data_[i];
	}

	HOST DEVICE constexpr T get(idx_t i) const noexcept {
		return i < size_ ? data_[i] : T{};
	}

	HOST DEVICE constexpr T& front() noexcept {
		return data_[0];
	}
	HOST DEVICE constexpr const T& front() const noexcept {
		return data_[0];
	}

	HOST DEVICE constexpr T& back() noexcept {
		return data_[size_ - 1];
	}
	HOST DEVICE constexpr const T& back() const noexcept {
		return data_[size_ - 1];
	}

	HOST DEVICE constexpr T* data() noexcept {
		return data_;
	}
	HOST DEVICE constexpr const T* data() const noexcept {
		return data_;
	}

	/*===================*\
	|  MODIFIERS          |
	\*===================*/

	/**
	 * @brief Add element to end (legacy interface compatibility)
	 */
	HOST DEVICE constexpr bool add(T value) noexcept {
		if (size_ < MaxSize) {
			data_[size_++] = value;
			return true;
		}
		return false; // List is full
	}

	/**
	 * @brief Add all elements from another IndexList
	 */
	HOST DEVICE constexpr bool add(const IndexList& other) noexcept {
		if (size_ + other.size_ <= MaxSize) {
			for (idx_t i = 0; i < other.size_; ++i) {
				data_[size_ + i] = other.data_[i];
			}
			size_ += other.size_;
			return true;
		}
		return false; // Not enough space
	}

	/**
	 * @brief Add element (std::vector-like interface)
	 */
	HOST DEVICE constexpr bool push_back(T value) noexcept {
		return add(value);
	}

	/**
	 * @brief Remove last element
	 */
	HOST DEVICE constexpr void pop_back() noexcept {
		if (size_ > 0)
			--size_;
	}

	/**
	 * @brief Clear all elements
	 */
	HOST DEVICE constexpr void clear() noexcept {
		size_ = 0;
	}

	/**
	 * @brief Remove element at specific position
	 */
	HOST DEVICE constexpr bool erase(idx_t pos) noexcept {
		if (pos >= size_)
			return false;

		// Shift elements left
		for (idx_t i = pos; i < size_ - 1; ++i) {
			data_[i] = data_[i + 1];
		}
		--size_;
		return true;
	}

	/**
	 * @brief Remove first occurrence of value
	 */
	HOST DEVICE constexpr bool remove(T value) noexcept {
		for (idx_t i = 0; i < size_; ++i) {
			if (data_[i] == value) {
				return erase(i);
			}
		}
		return false; // Value not found
	}

	/*===================*\
	|  SEARCH OPERATIONS  |
	\*===================*/

	/**
	 * @brief Find first occurrence of value
	 * @return Index of element, or MaxSize if not found
	 */
	HOST DEVICE constexpr idx_t find(T value) const noexcept {
		for (idx_t i = 0; i < size_; ++i) {
			if (data_[i] == value) {
				return i;
			}
		}
		return MaxSize; // Not found (similar to SIZE_MAX)
	}

	/**
	 * @brief Check if value exists in list
	 */
	HOST DEVICE constexpr bool contains(T value) const noexcept {
		return find(value) != MaxSize;
	}

	/*===================*\
	|  RANGE OPERATIONS   |
	\*===================*/

	/**
	 * @brief Get subrange as new IndexList
	 */
	HOST DEVICE constexpr IndexList range(idx_t start, idx_t end) const noexcept {
		IndexList result;
		if (start >= size_ || start >= end)
			return result;

		const idx_t actual_end = (end > size_) ? size_ : end;
		for (idx_t i = start; i < actual_end && result.size_ < MaxSize; ++i) {
			result.data_[result.size_++] = data_[i];
		}
		return result;
	}

	/*===================*\
	|  ALGORITHMS         |
	\*===================*/

	/**
	 * @brief Sort elements in ascending order
	 */
	HOST DEVICE constexpr void sort() noexcept {
		// Simple bubble sort for device compatibility
		for (idx_t i = 0; i < size_; ++i) {
			for (idx_t j = 0; j < size_ - 1 - i; ++j) {
				if (data_[j] > data_[j + 1]) {
					T temp = data_[j];
					data_[j] = data_[j + 1];
					data_[j + 1] = temp;
				}
			}
		}
	}

	/**
	 * @brief Remove duplicate elements (requires sorted list)
	 */
	HOST DEVICE constexpr void unique() noexcept {
		if (size_ <= 1)
			return;

		idx_t write_pos = 1;
		for (idx_t read_pos = 1; read_pos < size_; ++read_pos) {
			if (data_[read_pos] != data_[write_pos - 1]) {
				data_[write_pos++] = data_[read_pos];
			}
		}
		size_ = write_pos;
	}

	/**
	 * @brief Sort and remove duplicates
	 */
	HOST DEVICE constexpr void sort_unique() noexcept {
		sort();
		unique();
	}

	/*===================*\
	|  STATISTICS         |
	\*===================*/

	HOST DEVICE constexpr T min_element() const noexcept {
		if (size_ == 0)
			return T{};

		T min_val = data_[0];
		for (idx_t i = 1; i < size_; ++i) {
			if (data_[i] < min_val) {
				min_val = data_[i];
			}
		}
		return min_val;
	}

	HOST DEVICE constexpr T max_element() const noexcept {
		if (size_ == 0)
			return T{};

		T max_val = data_[0];
		for (idx_t i = 1; i < size_; ++i) {
			if (data_[i] > max_val) {
				max_val = data_[i];
			}
		}
		return max_val;
	}

	/*===================*\
	|  ITERATION          |
	\*===================*/

	// Device-safe iterator-like access
	HOST DEVICE constexpr T* begin() noexcept {
		return data_;
	}
	HOST DEVICE constexpr T* end() noexcept {
		return data_ + size_;
	}
	HOST DEVICE constexpr const T* begin() const noexcept {
		return data_;
	}
	HOST DEVICE constexpr const T* end() const noexcept {
		return data_ + size_;
	}
	HOST DEVICE constexpr const T* cbegin() const noexcept {
		return data_;
	}
	HOST DEVICE constexpr const T* cend() const noexcept {
		return data_ + size_;
	}

	/*===================*\
	|  COMPARISON         |
	\*===================*/

	HOST DEVICE constexpr bool operator==(const IndexList& other) const noexcept {
		if (size_ != other.size_)
			return false;

		for (idx_t i = 0; i < size_; ++i) {
			if (data_[i] != other.data_[i])
				return false;
		}
		return true;
	}

	HOST DEVICE constexpr bool operator!=(const IndexList& other) const noexcept {
		return !(*this == other);
	}

	/*===================*\
	|  HOST-ONLY METHODS  |
	\*===================*/

#if !defined(__CUDA_ARCH__) && !defined(__SYCL_DEVICE_ONLY__)

	/**
	 * @brief String representation for debugging (host only)
	 */
	std::string to_string() const {
		if (size_ == 0)
			return "IndexList[]";

		std::string result = "IndexList[";
		for (idx_t i = 0; i < size_; ++i) {
			if (i > 0)
				result += ", ";
			result += std::to_string(data_[i]);
		}
		result += "]";
		return result;
	}

	/**
	 * @brief Append from C-style array (host only)
	 */
	bool append(const T* values, idx_t count) {
		if (size_ + count > MaxSize)
			return false;

		for (idx_t i = 0; i < count; ++i) {
			data_[size_ + i] = values[i];
		}
		size_ += count;
		return true;
	}

#endif
};

/*===================*\
|  TYPE ALIASES       |
\*===================*/

// Common type aliases with different capacities
template<idx_t N = 32>
using IntIndexList = IndexList<int, N>;
template<idx_t N = 32>
using SizeIndexList = IndexList<idx_t, N>;
template<idx_t N = 128>
using NeighborList = IndexList<idx_t, N>; // Larger for neighbor lists
template<idx_t N = 16>
using SmallIndexList = IndexList<int, N>; // Smaller for simple cases

// Default aliases (backward compatibility)
using DefaultIndexList = IndexList<int, 32>;
using ParticleIndexList = IndexList<idx_t, 64>;

/*===================*\
|  HELPER FUNCTIONS   |
\*===================*/

/**
 * @brief Convert 3D indices to IndexList (device-safe replacement for index_to_ijk)
 */
template<typename T>
HOST DEVICE constexpr IndexList<T, 3> index_to_ijk(T linear_index, T nx, T ny, T nz) {
	IndexList<T, 3> result;
	result.add(linear_index / (ny * nz)); // ix
	result.add((linear_index / nz) % ny); // iy
	result.add(linear_index % nz);		  // iz
	return result;
}

/**
 * @brief Convert 3D indices to IndexList using dimensions vector
 */
template<typename T>
HOST DEVICE constexpr IndexList<T, 3> index_to_ijk(T linear_index, const T dims[3]) {
	return index_to_ijk(linear_index, dims[0], dims[1], dims[2]);
}

} // namespace ARBD