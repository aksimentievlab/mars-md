#pragma once
#include "Backend/Resource.h"
#include "ARBDException.h"
#include "ARBDLogger.h"
#include "Array.h"
#include "Bitmask.h"
#include "Matrix3.h"
#include "Vector3.h"
#include <cstring>
#include <memory>   // For std::unique_ptr
#include <stdarg.h> // For va_start, etc.
#include <cstddef>
#include <cstdint>
#include <string>
#include <cstdio>
#include <string_view>
#include <sstream>
#include <charconv>
#include <ranges>
#include "TypeName.h"

namespace ARBD {

// Simplified string formatting function for CUDA compatibility
template<typename... Args>
inline std::string string_format(const char* format, Args... args) {
  // Calculate required size
  int size = snprintf(nullptr, 0, format, args...);
  if (size <= 0) return std::string();
  
  // Allocate buffer and format
  size++; // for null terminator
  std::string result(size, '\0');
  snprintf(&result[0], size, format, args...);
  result.resize(size - 1); // remove null terminator
  return result;
}

// Includes of various types (allows those to be used simply by including
// Types.h)

using Vector3 = Vector3_t<float>;
using Matrix3 = Matrix3_t<float, false>;

using VecArray = Array<Vector3>;

// Helpful routines, not sure if needed
HOST DEVICE inline Vector3_t<size_t> index_to_ijk(size_t idx, size_t nx,
                                                  size_t ny, size_t nz) {
  Vector3_t<size_t> res;
  res.z = idx % nz;
  res.y = (idx / nz) % ny;
  res.x = (idx / (ny * nz)) % nx;
  return res;
}
HOST DEVICE inline Vector3_t<size_t> index_to_ijk(size_t idx,
                                                  const size_t n[]) {
  return index_to_ijk(idx, n[0], n[1], n[2]);
}
HOST DEVICE inline Vector3_t<size_t> index_to_ijk(size_t idx,
                                                  const Vector3_t<size_t> n) {
  return index_to_ijk(idx, n.x, n.y, n.z);
}

using idx_t = size_t; 
} // namespace ARBD
