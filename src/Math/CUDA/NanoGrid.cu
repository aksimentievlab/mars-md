#include "Backend/CUDA/KernelHelper.cuh"
#include "Math/NanoGridKernels.h"
#include "Math/Types.h"
#include "nanovdb/GridHandle.h"
#include "nanovdb/NanoVDB.h"

// Include necessary headers
#include "Backend/Buffer.h"
#include "Backend/Events.h"
#include "Backend/Kernels.h"
#include "Backend/Resource.h"

namespace ARBD {
using GridData = nanovdb::GridData;
using GridHandleMetaData = nanovdb::GridHandleMetaData;

// GridHandle kernel template instantiations
// CpyGridHandleMetaFunctor template instantiations
template Event launch_cuda_kernel_impl<std::tuple<const GridData*, GridHandleMetaData*>,
									   std::tuple<>,
									   CpyGridHandleMetaFunctor<GridData, GridHandleMetaData>&>(
	const Resource& resource,
	size_t thread_count,
	const std::tuple<const GridData*, GridHandleMetaData*>& inputs,
	const std::tuple<>& outputs,
	const KernelConfig& config,
	CpyGridHandleMetaFunctor<GridData, GridHandleMetaData>& kernel_func);

// UpdateGridCountFunctor template instantiations
template Event launch_cuda_kernel_impl<std::tuple<GridData*, uint32_t, uint32_t, bool*>,
									   std::tuple<>,
									   UpdateGridCountFunctor<GridData>&>(
	const Resource& resource,
	size_t thread_count,
	const std::tuple<GridData*, uint32_t, uint32_t, bool*>& inputs,
	const std::tuple<>& outputs,
	const KernelConfig& config,
	UpdateGridCountFunctor<GridData>& kernel_func);

} // namespace ARBD
