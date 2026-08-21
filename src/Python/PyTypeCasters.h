#pragma once
/**
 * @brief nanobind type_casters converting ARBD::Vector3_t<T>/Matrix3_t<T>
 * to/from numpy arrays ((3,) and (3,3) respectively).
 *
 * @note Must be included by every .cpp file binding these types, before any
 * `.def(...)`/`nb::init<...>()` call that uses them. See dev_notes.md.
 */
#include "Types/Matrix3.h"
#include "Types/Vector3.h"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace nanobind {
namespace detail {

template<typename T>
struct type_caster<ARBD::Vector3_t<T>> {
	using Array = nanobind::ndarray<T, nanobind::numpy, nanobind::shape<3>>;

	NB_TYPE_CASTER(ARBD::Vector3_t<T>, const_name("Vector3"))

	// Python -> C++
	bool from_python(handle src, uint32_t /*flags*/, cleanup_list*) noexcept {
		if (!src.is_valid() || isinstance<str>(src))
			return false;

		try {
			if (len(src) != 3)
				return false;
			value = ARBD::Vector3_t<T>(cast<T>(src[0]), cast<T>(src[1]), cast<T>(src[2]));
			return true;
		} catch (...) {
			PyErr_Clear();
			return false;
		}
	}

	// C++ -> Python
	static handle from_cpp(const ARBD::Vector3_t<T>& src, rv_policy, cleanup_list*) noexcept {
		T data[3] = {src.x, src.y, src.z};
		return Array(data).cast().release();
	}
};

/**
 * Matrix3_t stores three column vectors; numpy arrays are read row-major, so
 * both directions transpose. A (3,3) array's rows are the matrix's rows, as a
 * reader would expect from `np.eye(3)` or a rotation matrix.
 */
template<typename T>
struct type_caster<ARBD::Matrix3_t<T>> {
	using Array = nanobind::ndarray<T, nanobind::numpy, nanobind::shape<3, 3>>;

	NB_TYPE_CASTER(ARBD::Matrix3_t<T>, const_name("Matrix3"))

	bool from_python(handle src, uint32_t /*flags*/, cleanup_list*) noexcept {
		if (!src.is_valid())
			return false;

		try {
			if (len(src) != 3)
				return false;

			T r[3][3];
			for (Py_ssize_t i = 0; i < 3; ++i) {
				handle row = src[i];
				if (len(row) != 3)
					return false;
				for (Py_ssize_t j = 0; j < 3; ++j)
					r[i][j] = cast<T>(row[j]);
			}

			value = ARBD::Matrix3_t<T>(ARBD::Vector3_t<T>(r[0][0], r[1][0], r[2][0]),
										ARBD::Vector3_t<T>(r[0][1], r[1][1], r[2][1]),
										ARBD::Vector3_t<T>(r[0][2], r[1][2], r[2][2]));
			return true;
		} catch (...) {
			PyErr_Clear();
			return false;
		}
	}

	static handle from_cpp(const ARBD::Matrix3_t<T>& src, rv_policy, cleanup_list*) noexcept {
		const auto& c0 = src.ex();
		const auto& c1 = src.ey();
		const auto& c2 = src.ez();
		T data[9] = {c0.x, c1.x, c2.x, c0.y, c1.y, c2.y, c0.z, c1.z, c2.z};
		return Array(data).cast().release();
	}
};

} // namespace detail
} // namespace nanobind
