#include "MARSException.h"
#include "DeviceRadix.h"
#include <cuda_runtime.h>
#ifdef Debug
#undef Debug
#include <cub/cub.cuh>
#define Debug(x) static_cast<void>(0)
#else
#include <cub/cub.cuh>
#endif

namespace MARS {
void device_radix_sort_pairs_cub(int device_id,
								 uint32_t* d_keys,
								 uint32_t* d_payloads,
								 uint32_t* d_alt_keys,
								 uint32_t* d_alt_payloads,
								 uint32_t size) {
	// Set the CUDA device
	cudaSetDevice(device_id);

	cudaStream_t stream = nullptr;
	// 1. Wrap pointers in DoubleBuffer
	// This tells CUB it can swap between d_keys <-> d_alt_keys as needed
	cub::DoubleBuffer<uint32_t> d_keys_db(d_keys, d_alt_keys);
	cub::DoubleBuffer<uint32_t> d_values_db(d_payloads, d_alt_payloads);

	// 2. Determine temporary device storage requirements
	void* d_temp_storage = nullptr;
	size_t temp_storage_bytes = 0;

	cudaError_t err = cub::DeviceRadixSort::SortPairs(d_temp_storage,
													  temp_storage_bytes,
													  d_keys_db,
													  d_values_db,
													  size,
													  0,
													  sizeof(uint32_t) * 8,
													  stream);
	if (err != cudaSuccess)
		MARS_Exception(ExceptionType::CUDARuntimeError,
					   "CUB SortPairs size query failed: %s",
					   cudaGetErrorString(err));

	err = cudaMalloc(&d_temp_storage, temp_storage_bytes);
	if (err != cudaSuccess)
		MARS_Exception(ExceptionType::CUDARuntimeError,
					   "cudaMalloc of %zu B radix-sort scratch failed: %s",
					   temp_storage_bytes,
					   cudaGetErrorString(err));

	// 4. Run sorting operation
	err = cub::DeviceRadixSort::SortPairs(d_temp_storage,
										  temp_storage_bytes,
										  d_keys_db,
										  d_values_db,
										  size,
										  0,
										  sizeof(uint32_t) * 8,
										  stream);
	// Catch the immediate return AND an asynchronous launch failure (e.g. a
	// missing kernel image on an untargeted arch), which the return code misses.
	if (err == cudaSuccess)
		err = cudaGetLastError();
	if (err != cudaSuccess) {
		cudaFree(d_temp_storage);
		MARS_Exception(ExceptionType::CUDARuntimeError,
					   "CUB SortPairs failed for %u elements: %s",
					   size,
					   cudaGetErrorString(err));
	}

	cudaFree(d_temp_storage);

	// 5. CUB may leave the result in the alt buffer depending on pass count;
	// copy back so the sorted data always lands in the caller's primary buffers.
	if (d_keys_db.Current() != d_keys) {
		cudaError_t e1 = cudaMemcpy(d_keys,
									d_keys_db.Current(),
									size * sizeof(uint32_t),
									cudaMemcpyDeviceToDevice);
		cudaError_t e2 = cudaMemcpy(d_payloads,
									d_values_db.Current(),
									size * sizeof(uint32_t),
									cudaMemcpyDeviceToDevice);
		if (e1 != cudaSuccess || e2 != cudaSuccess)
			MARS_Exception(ExceptionType::CUDARuntimeError,
						   "radix-sort result copyback failed: keys=%s payloads=%s",
						   cudaGetErrorString(e1),
						   cudaGetErrorString(e2));
	}

	err = cudaDeviceSynchronize();
	if (err != cudaSuccess)
		MARS_Exception(ExceptionType::CUDARuntimeError,
					   "cudaDeviceSynchronize after radix sort failed: %s",
					   cudaGetErrorString(err));
}
} // namespace MARS
