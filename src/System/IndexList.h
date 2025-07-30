#pragma once

#include <algorithm>
#include <string>
#include <vector>
#include "ARBDLogger.h"
#include "Backend/Resource.h"
#include "ARBDException.h"

namespace ARBD { 

/**
 * @brief Modern replacement for the legacy IndexList class from Useful.h
 * 
 * This class provides a dynamic list of indices used in simulation contexts,
 * such as particle neighbor lists, force computation lists, etc.
 * Uses std::vector as the backend for memory safety and performance.
 * 
 * @tparam T Index type (defaults to int for compatibility)
 */
template<typename T = int>
class IndexList {
private:
  std::vector<T> data_;

public:
  IndexList() = default;
  explicit IndexList(size_t reserve_size) { data_.reserve(reserve_size); }
  
  // Legacy interface compatibility
  void add(T value) { data_.push_back(value); }
  void add(const IndexList<T>& other) {
    data_.insert(data_.end(), other.data_.begin(), other.data_.end());
  }
  
  HOST DEVICE size_t length() const noexcept { return data_.size(); }
  HOST DEVICE bool empty() const noexcept { return data_.empty(); }
  
  HOST DEVICE T get(size_t i) const noexcept { 
    return i < data_.size() ? data_[i] : T{};
  }
  
  HOST DEVICE T& operator[](size_t i) noexcept { return data_[i]; }
  HOST DEVICE const T& operator[](size_t i) const noexcept { return data_[i]; }
  
  void clear() noexcept { data_.clear(); }
  void reserve(size_t size) { data_.reserve(size); }
  
  // Range operations (useful for simulation partitioning)
  IndexList<T> range(size_t start, size_t end) const {
    IndexList<T> result;
    if (start < data_.size() && end <= data_.size() && start <= end) {
      result.data_.assign(data_.begin() + start, data_.begin() + end);
    }
    return result;
  }
  
  // Search operations (useful for finding particle indices)
  size_t find(T key) const noexcept {
    auto it = std::find(data_.begin(), data_.end(), key);
    return it != data_.end() ? std::distance(data_.begin(), it) : SIZE_MAX;
  }
  
  bool contains(T key) const noexcept {
    return find(key) != SIZE_MAX;
  }
  
  // STL compatibility for modern C++ algorithms
  auto begin() noexcept { return data_.begin(); }
  auto end() noexcept { return data_.end(); }
  auto begin() const noexcept { return data_.begin(); }
  auto end() const noexcept { return data_.end(); }
  
  // Data access for interfacing with legacy code
  const T* data() const noexcept { return data_.data(); }
  T* data() noexcept { return data_.data(); }
  
  // Simulation-specific operations
  void sort() { std::sort(data_.begin(), data_.end()); }
  void unique() { 
    sort(); 
    data_.erase(std::unique(data_.begin(), data_.end()), data_.end());
  }
  
  // Remove specific indices
  void remove(T value) {
    data_.erase(std::remove(data_.begin(), data_.end(), value), data_.end());
  }
  
  // Bulk operations for performance
  void append(const std::vector<T>& values) {
    data_.insert(data_.end(), values.begin(), values.end());
  }
  
  void append(const T* values, size_t count) {
    data_.insert(data_.end(), values, values + count);
  }
  
  // String representation for debugging
  std::string to_string() const {
    if (data_.empty()) return "IndexList[]";
    std::string result = "IndexList[";
    for (size_t i = 0; i < data_.size(); ++i) {
      if (i > 0) result += ", ";
      result += std::to_string(data_[i]);
    }
    result += "]";
    return result;
  }
  
  // Statistics (useful for simulation analysis)
  T min_element() const { 
    return data_.empty() ? T{} : *std::min_element(data_.begin(), data_.end());
  }
  
  T max_element() const {
    return data_.empty() ? T{} : *std::max_element(data_.begin(), data_.end());
  }
  
  // Backend operations using Proxy system
  template<typename ResourceType>
  IndexList<T> send_to_backend(const ResourceType& resource) const {
    return IndexList<T>(*this);
  }
};

// Type aliases for common use cases
using ParticleIndexList = IndexList<size_t>;
using IntIndexList = IndexList<int>;
using NeighborList = IndexList<size_t>;

} // namespace ARBD 