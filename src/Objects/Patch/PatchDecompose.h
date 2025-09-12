#include "Objects/Patch/Patch.h"
#include "SimSystem.h"
#include "System/DecomposeKernels.h"
#include "System/PatchManager.h"
#include "Types/Types.h"

namespace ARBD {
void Patch::decompose_d(Vector3 pos_d[], size_t num) {

	const size_t cells_sz = sizeof(cell_t) * num * numReplicas;
	const size_t numCellRep = numCells * numReplicas;

	if (cells_d == NULL) {
		gpuErrchk(cudaMalloc(&cells_d, cells_sz));
		gpuErrchk(cudaMalloc(&unsorted_cells_d, cells_sz));
		gpuErrchk(cudaMalloc(&ranges_d, sizeof(range_t) * numCellRep));
		unsorted_cells = new cell_t[num * numReplicas];
		cells = new cell_t[num * numReplicas];
		ranges = new range_t[numCellRep];
	}

	// Pair particles with cells.
	size_t nBlocks = (num * numReplicas) / NUM_THREADS + 1;
	decomposeKernel<<<nBlocks, NUM_THREADS>>>(pos_d,
											  cells_d,
											  origin,
											  cutoff,
											  nCells,
											  num,
											  numReplicas);
	gpuErrchk(cudaDeviceSynchronize());
	gpuErrchk(cudaMemcpy(unsorted_cells_d, cells_d, cells_sz, cudaMemcpyDeviceToDevice));
	gpuErrchk(cudaMemcpyAsync(unsorted_cells, unsorted_cells_d, cells_sz, cudaMemcpyDeviceToHost));

	// Sort cells.
	thrust::device_ptr<cell_t> c_d(cells_d);
	thrust::sort(c_d, c_d + num * numReplicas);
	gpuErrchk(cudaMemcpyAsync(cells, cells_d, cells_sz, cudaMemcpyDeviceToHost));
	// Han-Yi Chou
	// gpuErrchk(cudaMemcpy(cells, cells_d, cells_sz, cudaMemcpyDeviceToHost));
	const size_t nMax = std::max(2lu * numCells, num);
	nBlocks = (nMax * numReplicas) / NUM_THREADS + 1;

	// Create ranges for cells.
	int* temp_ranges = NULL;
	gpuErrchk(cudaMalloc(&temp_ranges, 2 * sizeof(int) * numCellRep));
	gpuErrchk(cudaMemset(temp_ranges, -1, 2 * sizeof(int) * numCellRep));
	make_rangesKernel<<<nBlocks, NUM_THREADS>>>(cells_d, temp_ranges, num, numCells, numReplicas);
	gpuErrchk(cudaDeviceSynchronize());

	// Copy temp_ranges to ranges_d
	bind_rangesKernel<<<nBlocks, NUM_THREADS>>>(ranges_d, temp_ranges, numCells, numReplicas);
	gpuErrchk(cudaMemcpy(ranges, ranges_d, numCellRep, cudaMemcpyDeviceToHost));
	gpuErrchk(cudaFree(temp_ranges));
}
} // namespace ARBD
