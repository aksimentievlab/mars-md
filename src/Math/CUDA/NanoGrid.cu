#include "Backend/CUDA/KernelHelper.cuh"
#include "Math/NanoGridKernels.h"
#include "Math/Types.h"

#include "nanovdb/GridHandle.h"
#include "nanovdb/NanoVDB.h"
#include "nanovdb/math/Stencils.h"

namespace ARBD {
using GridData = nanovdb::GridData;
using GridHandleMetaData = nanovdb::GridHandleMetaData;

// GridHandle kernel template instantiations
// CpyGridHandleMetaFunctor template instantiations
template Event
launch_cuda_kernel(const Resource& resource,
				   size_t thread_count,
				   const std::tuple<const GridData*, GridHandleMetaData*>& inputs,
				   const std::tuple<>& outputs,
				   const KernelConfig& config,
				   CpyGridHandleMetaFunctor<GridData, GridHandleMetaData>& kernel_func);

// UpdateGridCountFunctor template instantiations
template Event launch_cuda_kernel(const Resource& resource,
								  size_t thread_count,
								  const std::tuple<GridData*, uint32_t, uint32_t, bool*>& inputs,
								  const std::tuple<>& outputs,
								  const KernelConfig& config,
								  UpdateGridCountFunctor<GridData>& kernel_func);

// Stencil kernel template instantiations for common types
// Float gradients with Coord coordinates
template Event launch_cuda_kernel(const Resource& resource,
								  size_t thread_count,
								  const std::tuple<const nanovdb::NanoGrid<float>*,
												   const nanovdb::Coord*,
												   nanovdb::math::Vec3<float>*,
												   size_t>& inputs,
								  const std::tuple<nanovdb::math::Vec3<float>*>& outputs,
								  const KernelConfig& config,
								  GradientStencilFunctor<float, nanovdb::Coord>& kernel_func);

// Float laplacians with Coord coordinates
template Event launch_cuda_kernel(
	const Resource& resource,
	size_t thread_count,
	const std::tuple<const nanovdb::NanoGrid<float>*, const nanovdb::Coord*, float*, size_t>&
		inputs,
	const std::tuple<float*>& outputs,
	const KernelConfig& config,
	LaplacianStencilFunctor<float, nanovdb::Coord>& kernel_func);

// Float interpolation with Vec3 world positions
template Event launch_cuda_kernel(
	const Resource& resource,
	size_t thread_count,
	const std::
		tuple<const nanovdb::NanoGrid<float>*, const nanovdb::math::Vec3<float>*, float*, size_t>&
			inputs,
	const std::tuple<float*>& outputs,
	const KernelConfig& config,
	TrilinearInterpolationFunctor<float, nanovdb::math::Vec3<float>>& kernel_func);

} // namespace ARBD
