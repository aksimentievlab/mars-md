#pragma once
/**
 * @brief pybind11 type_casters for ARBD::Vector3_t<T> and ARBD::Matrix3_t<T>.
 *
 * Neither type gets a dedicated Python class. They convert transparently to
 * plain numpy arrays on the way out ((3,) and (3,3) respectively), and accept
 * any equivalently-shaped list/tuple/numpy array on the way in. This replaces
 * the earlier `py::class_` bindings, which forced every call site to wrap
 * plain coordinates in `arbd.Vector3(x, y, z)` / `arbd.Matrix3(...)` first -
 * arithmetic, dot/cross products, transpose, and inverse are just numpy's
 * now, not a bespoke reimplementation.
 *
 * type_caster specializations are resolved per translation unit, so this
 * header must be included (before any `.def(...)`/`py::init<...>()` call that
 * takes or returns one of these types) by every .cpp file that binds one.
 */
#include "Types/Matrix3.h"
#include "Types/Vector3.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace pybind11 {
namespace detail {

template<typename T>
struct type_caster<ARBD::Vector3_t<T>> {
  public:
	PYBIND11_TYPE_CASTER(ARBD::Vector3_t<T>, const_name("Vector3"));

	// Python -> C++
	bool load(handle src, bool /*convert*/) {
		if (!src)
			return false;

		if (isinstance<array>(src)) {
			auto arr = array_t<T, array::c_style | array::forcecast>::ensure(src);
			if (!arr || arr.ndim() != 1 || arr.shape(0) != 3)
				return false;
			auto r = arr.template unchecked<1>();
			value = ARBD::Vector3_t<T>(r(0), r(1), r(2));
			return true;
		}

		if (isinstance<sequence>(src) && !isinstance<str>(src)) {
			auto seq = reinterpret_borrow<sequence>(src);
			if (seq.size() != 3)
				return false;
			try {
				value = ARBD::Vector3_t<T>(seq[0].cast<T>(), seq[1].cast<T>(), seq[2].cast<T>());
			} catch (const cast_error&) {
				return false;
			}
			return true;
		}

		return false;
	}

	// C++ -> Python
	static handle cast(const ARBD::Vector3_t<T>& src, return_value_policy, handle) {
		array_t<T> result(3);
		auto r = result.template mutable_unchecked<1>();
		r(0) = src.x;
		r(1) = src.y;
		r(2) = src.z;
		return result.release();
	}
};

/**
 * Matrix3_t stores three column vectors; numpy arrays are read row-major, so
 * both directions transpose. A (3,3) array's rows are the matrix's rows, as a
 * reader would expect from `np.eye(3)` or a rotation matrix.
 */
template<typename T>
struct type_caster<ARBD::Matrix3_t<T>> {
  public:
	PYBIND11_TYPE_CASTER(ARBD::Matrix3_t<T>, const_name("Matrix3"));

	bool load(handle src, bool /*convert*/) {
		if (!src)
			return false;

		auto arr = array_t<T, array::c_style | array::forcecast>::ensure(src);
		if (!arr || arr.ndim() != 2 || arr.shape(0) != 3 || arr.shape(1) != 3)
			return false;

		auto r = arr.template unchecked<2>();
		value = ARBD::Matrix3_t<T>(ARBD::Vector3_t<T>(r(0, 0), r(1, 0), r(2, 0)),
								   ARBD::Vector3_t<T>(r(0, 1), r(1, 1), r(2, 1)),
								   ARBD::Vector3_t<T>(r(0, 2), r(1, 2), r(2, 2)));
		return true;
	}

	static handle cast(const ARBD::Matrix3_t<T>& src, return_value_policy, handle) {
		array_t<T> result({3, 3});
		auto r = result.template mutable_unchecked<2>();
		const auto& c0 = src.ex();
		const auto& c1 = src.ey();
		const auto& c2 = src.ez();
		r(0, 0) = c0.x;
		r(1, 0) = c0.y;
		r(2, 0) = c0.z;
		r(0, 1) = c1.x;
		r(1, 1) = c1.y;
		r(2, 1) = c1.z;
		r(0, 2) = c2.x;
		r(1, 2) = c2.y;
		r(2, 2) = c2.z;
	}
};

} // namespace detail
} // namespace pybind11
