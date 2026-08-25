#pragma once

#include "Backend/Kernels.h"
#include "Types/BaseGridDevice.h"

namespace Test {

using namespace MARS;

template<typename T>
struct Params {
	Vector3_t<T> origin;
	Matrix3_t<T> basis_inv;
	Vector3_t<idx_t> dims;
};

// Named functor (as opposed to an ad hoc lambda) so its launch_cuda_kernel
// instantiation can be forced into a real CUDA compilation unit - see
// basegrid_device_instantiations.cu. A lambda defined inline in a test
// TU is a distinct, anonymous type per translation unit, so no .cu file
// could ever pre-instantiate a CUDA-capable launch_cuda_kernel for it;
// calling launch_kernel with one directly from a host-only .cpp always
// hits the NotImplementedError stub in KernelHelper.cuh.
template<typename T>
struct GridQueryKernel {
	DEVICE_PTR(const Vector3_t<T>) pos;
	DEVICE_PTR(const T) values;
	DEVICE_PTR(T) out_interp;
	DEVICE_PTR(T) out_nearest;
	DEVICE_PTR(const Params<T>) params;

	DEVICE void operator()(idx_t i) const {
		const Params<T> p = params[0];
		const Vector3_t<T> pt = pos[i];
		const T vi =
			interpolate_grid_point<T>(values, pt, p.origin, p.basis_inv, p.dims, 2);	 // 2 = Periodic
		const T vn = get_value_nearest<T>(values, pt, p.origin, p.basis_inv, p.dims, 2); // 2 = Periodic
		out_interp[i] = vi;
		out_nearest[i] = vn;
	}
};

// Launch wrapper - call this from tests instead of invoking launch_kernel
// directly with an ad hoc lambda (see GridQueryKernel comment above).
template<typename T>
inline MARS::Event launch_grid_query(const MARS::Resource& resource,
									 const MARS::DeviceBuffer<Vector3_t<T>>& pos,
									 const MARS::DeviceBuffer<T>& values,
									 MARS::DeviceBuffer<T>& out_interp,
									 MARS::DeviceBuffer<T>& out_nearest,
									 const MARS::DeviceBuffer<Params<T>>& params,
									 idx_t count) {
	GridQueryKernel<T> kernel{
		pos.data(), values.data(), out_interp.data(), out_nearest.data(), params.data()};
	MARS::KernelConfig config = MARS::KernelConfig::for_1d(count, resource);
	config.sync = true;
	return MARS::launch_kernel(resource, config, kernel);
}

} // namespace Test

// Explicit template instantiation declaration to prevent host instantiation.
// See basegrid_device_instantiations.cu for the real definition.
#ifdef USE_CUDA
#include "Backend/CUDA/KernelHelper.cuh"
namespace MARS {
extern template Event launch_cuda_kernel(const Resource& resource,
										 const KernelConfig& config,
										 Test::GridQueryKernel<float> kernel_func);
} // namespace MARS
#endif
