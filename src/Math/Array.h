/*********************************************************************
 * @file  Array.h
 *
 * @brief Declaration of templated Array class.
 *********************************************************************/
#pragma once
#include "ARBDException.h"
#include "ARBDLogger.h"
#include "Backend/Buffer.h"
#include "Backend/Resource.h"
#include "TypeName.h"
#include <cassert>
#include <compare>
#include <cstdlib>
#include <cstring>
#include <new>
#include <type_traits>

namespace ARBD {
template<typename T>
struct Array {
	size_t num;
	T* values;

	// Minimal interface for legacy compatibility
	Array(size_t count) : num(count), values(new T[count]) {}
	~Array() {
		delete[] values;
	}
	T& operator[](size_t i) {
		return values[i];
	}
	size_t size() const {
		return num;
	}
};

} // namespace ARBD
