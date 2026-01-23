#pragma once

#include "Types/BaseGridDevice.h"

using namespace ARBD;

namespace Test {
template<typename T>
struct Params {
	Vector3_t<T> origin;
	Matrix3_t<T> basis_inv;
	Vector3_t<idx_t> dims;
};

template<typename T>
auto kernel = [](idx_t i,
				 const Vector3_t<T>* pos,
				 const T* values,
				 T* out_interp,
				 T* out_nearest,
				 const Params<T>* params) {
	const Params p = params[0];
	const Vector3_t<T> pt = pos[i];
	const T vi =
		interpolate_grid_point<T>(values, pt, p.origin, p.basis_inv, p.dims, 2);	 // 2 = Periodic
	const T vn = get_value_nearest<T>(values, pt, p.origin, p.basis_inv, p.dims, 2); // 2 = Periodic
	out_interp[i] = vi;
	out_nearest[i] = vn;
};
} // namespace Test
