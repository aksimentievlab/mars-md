/*********************************************************************
 * @file  Array.h
 *
 * @brief Declaration of templated Array class.
 *********************************************************************/
#pragma once

#include "Header.h"

namespace ARBD {
template<typename T>
struct Array {
	size_t num;
	T* values;
	Array(size_t count) : num(count), values(new T[count]) {}
	~Array() {
		delete[] values;
	}
	T& operator[](size_t i) {
		return values[i];
	}
	const T& operator[](size_t i) const {
		return values[i];
	}
	size_t size() const {
		return num;
	}
};

} // namespace ARBD

// SYCL specialization
#ifdef USE_SYCL
#include <sycl/sycl.hpp>
template<typename T>
struct sycl::is_device_copyable<ARBD::Array<T> > : std::true_type {};
#endif
