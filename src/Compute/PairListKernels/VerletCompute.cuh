#pragma once
#include "CudaUtil.cuh"
#include "TabulatedMethods.cuh"
#include <cassert>

#define MAX_CELLS_FOR_CELLNEIGHBORLIST 1 << 25

/* const __device__ int maxPairs = 1 << 14; */

/* __global__ */
/* void pairlistTest(Vector3 pos[], int num, int numReplicas, */
/* 									BaseGrid* sys, CellDecomposition* decomp, */
/* 									const int nCells, const int blocksPerCell, */
/* 									int* g_numPairs, int* g_pairI, int* g_pairJ ) { */
/* 	const int gtid = threadIdx.x + blockIdx.x*blockDim.x; */
/* 	for (int i = gtid; i < gridDim.x*100; i+=blockDim.x) { */
/* 		assert( g_numPairs[i] == 0 ); */
/* 		assert( g_pairI[i] != NULL ); */
/* 		assert( g_pairJ[i] != NULL ); */
/* 	} */
/* } */

__device__ int* exSum;
void initExSum() {
	int tmp = 0;
	int* devPtr;
	cudaMalloc(&devPtr, sizeof(int));
	cudaMemcpyToSymbol(exSum, &devPtr, sizeof(int*));
	cudaMemcpy(devPtr, &tmp, sizeof(int), cudaMemcpyHostToDevice);
}
int getExSum() {
	int tmp;
	int* devPtr;
	cudaMemcpyFromSymbol(&devPtr, exSum, sizeof(int*));
	cudaMemcpy(&tmp, devPtr, sizeof(int), cudaMemcpyDeviceToHost);
	return tmp;
}
//
__device__ int computeCellNeighbor(const int3 cells,
								   const int3 cell_idx,
								   const int dx,
								   const int dy,
								   const int dz) {
	int idx = cell_idx.x;
	int idy = cell_idx.y;
	int idz = cell_idx.z;

	int u = idx + dx;
	int v = idy + dy;
	int w = idz + dz;

	int nID;
	if (cells.x == 1 and u != 0)
		nID = -1;
	else if (cells.y == 1 and v != 0)
		nID = -1;
	else if (cells.z == 1 and w != 0)
		nID = -1;
	else if (cells.x == 2 and (u < 0 || u > 1))
		nID = -1;
	else if (cells.y == 2 and (v < 0 || v > 1))
		nID = -1;
	else if (cells.z == 2 and (w < 0 || w > 1))
		nID = -1;
	else {
		u = (u + cells.x) % cells.x;
		v = (v + cells.y) % cells.y;
		w = (w + cells.z) % cells.z;
		nID = w + cells.z * (v + cells.y * u);
	}

	return nID;
}

__global__ void createNeighborsList(const int3* Cells, int* __restrict__ CellNeighborsList) {
	const int tid = threadIdx.x + blockDim.x * blockIdx.x;
	const int3 cells = Cells[0];
	const int nCells = cells.x * cells.y * cells.z;
	const int Size = blockDim.x * gridDim.x;
	int nID;

	for (int cID = tid; cID < nCells; cID += Size) {

		int idz = cID % cells.z;
		int idy = cID / cells.z % cells.y;
		int idx = cID / (cells.z * cells.y);

		int count = 0;
		for (int dx = -1; dx <= 1; ++dx) {
			for (int dy = -1; dy <= 1; ++dy) {
				for (int dz = -1; dz <= 1; ++dz) {

					nID = computeCellNeighbor(cells, make_int3(idx, idy, idz), dx, dy, dz);
					CellNeighborsList[size_t(count + 27 * cID)] = nID;
					++count;
					//__syncthreads();
				}
			}
		}
	}
}
template<const int BlockSize, const int Size, const int N>
__global__ void createPairlists(Vector3* __restrict__ pos,
								const int num,
								const int numReplicas,
								const BaseGrid* __restrict__ sys,
								const CellDecomposition* __restrict__ decomp,
								const int nCells,
								int* g_numPairs,
								int2* g_pair,
								int numParts,
								const int* __restrict__ type,
								int* __restrict__ g_pairTabPotType,
								const Exclude* __restrict__ excludes,
								const int2* __restrict__ excludeMap,
								const int numExcludes,
								float pairlistdist2,
								cudaTextureObject_t PosTex,
								cudaTextureObject_t NeighborsTex) {
	__shared__ float4 __align__(16) particle[N];
	__shared__ int Index_i[N];

	const int TotalBlocks = gridDim.x * gridDim.y;
	const int cells = TotalBlocks / Size;
	const int cell_start = (blockIdx.x + gridDim.x * blockIdx.y) / Size;
	const int pid_start = ((blockIdx.x + gridDim.x * blockIdx.y) % Size) * N;
	const int tid = threadIdx.x + blockDim.x * threadIdx.y + blockDim.x * blockDim.y * threadIdx.z;
	const int warpLane = tid % WARPSIZE;
	const int nReps = gridDim.z;
	const int idx_ = tid % N;
	const int idx__ = tid / N;
	const int Step1 = Size * N;
	const int Step2 = Size / N;

	const CellDecomposition::cell_t* __restrict__ cellInfo = decomp->getCells();

	for (int repID = blockIdx.z; repID < numReplicas; repID += nReps) {
		for (int cellid_i = cell_start; cellid_i < nCells; cellid_i += cells) {
			CellDecomposition::range_t rangeI = decomp->getRange(cellid_i, repID);
			int Ni = rangeI.last - rangeI.first;

			for (int pid_i = pid_start; pid_i < Ni; pid_i += Step1) {
				__syncthreads();
				if (tid + pid_i < Ni && tid < N) {
					Index_i[tid] = cellInfo[rangeI.first + pid_i + tid].particle;
					particle[tid] = tex1Dfetch<float4>(PosTex, Index_i[tid]);
				}
				__syncthreads();

				if (idx_ + pid_i < Ni) {
					int ai = Index_i[idx_];
					Vector3 A(particle[idx_]);

					int2 ex_pair = make_int2(-1, -1);
					if (numExcludes > 0 && excludeMap != NULL) {
						ex_pair = excludeMap[ai - repID * num];
					}

					// loop over neighbor directions
					for (int idx = 0; idx < 27; ++idx) {

						int currEx = ex_pair.x;
						int nextEx = (ex_pair.x >= 0) ? excludes[currEx].ind2 : -1;

						int neighbor_cell;
						if (nCells < MAX_CELLS_FOR_CELLNEIGHBORLIST) {
							neighbor_cell = tex1Dfetch<int>(NeighborsTex, idx + 27 * cellid_i);
						} else {
							int3 cells = decomp->nCells;
							int3 cell_idx = make_int3(cellid_i % cells.z,
													  cellid_i / cells.z % cells.y,
													  cellid_i / (cells.z * cells.y));

							int dz = (idx % 3) - 1;
							int dy = ((idx / 3) % 3) - 1;
							int dx = ((idx / 9) % 3) - 1;
							neighbor_cell =
								computeCellNeighbor(decomp->nCells, cell_idx, dx, dy, dz);
						}

						if (neighbor_cell < 0) {
							continue;
						}

						CellDecomposition::range_t rangeJ = decomp->getRange(neighbor_cell, repID);
						int Nj = rangeJ.last - rangeJ.first;

						// In each neighbor cell, loop over particles
						for (int pid_j = idx__; pid_j < Nj; pid_j += Step2) {

							int aj = cellInfo[pid_j + rangeJ.first].particle;
							if (aj <= ai) {
								continue;
							}

							while (nextEx >= 0 && nextEx < (aj - repID * num)) {
								nextEx = (currEx < ex_pair.y - 1) ? excludes[++currEx].ind2 : -1;
							}

							if (nextEx == (aj - repID * num)) {
#ifdef DEBUGEXCLUSIONS
								atomicAggInc(exSum, warpLane);
#endif
								nextEx = (currEx < ex_pair.y - 1) ? excludes[++currEx].ind2 : -1;
								continue;
							}

							float4 b = tex1Dfetch<float4>(PosTex, aj);
							Vector3 B(b.x, b.y, b.z);

							float dr = (sys->wrapDiff(A - B)).length2();
							if (dr <= pairlistdist2) {
								int gid = atomicAggInc(g_numPairs, warpLane);
								int pairType = type[ai] + type[aj] * numParts;

								g_pair[gid] = make_int2(ai, aj);
								g_pairTabPotType[gid] = pairType;
							}
						}
					}
				}
			}
		}
	}
}

__global__ void createPairlists_debug(Vector3* __restrict__ pos,
									  const int num,
									  const int numReplicas,
									  const BaseGrid* __restrict__ sys,
									  const CellDecomposition* __restrict__ decomp,
									  const int nCells,
									  int* g_numPairs,
									  int2* g_pair,
									  int numParts,
									  const int* __restrict__ type,
									  int* __restrict__ g_pairTabPotType,
									  const Exclude* __restrict__ excludes,
									  const int2* __restrict__ excludeMap,
									  const int numExcludes,
									  float pairlistdist2) {
	// TODO: loop over all cells with edges within pairlistdist2
	// Loop over threads searching for atom pairs
	//   Each thread has designated values in shared memory as a buffer
	//   A sync operation periodically moves data from shared to global
	const int tid = threadIdx.x;
	const int warpLane = tid % WARPSIZE; /* RBTODO: optimize */
	const int split = 32;				 /* numblocks should be divisible by split */
	/* const int blocksPerCell = gridDim.x/split;  */
	const CellDecomposition::cell_t* __restrict__ cellInfo = decomp->getCells();
	for (int cID = 0 + (blockIdx.x % split); cID < nCells; cID += split) {
		for (int repID = 0; repID < numReplicas; repID++) {
			const CellDecomposition::range_t rangeI = decomp->getRange(cID, repID);
			for (int ci = rangeI.first + blockIdx.x / split; ci < rangeI.last;
				 ci += gridDim.x / split) {
				const int ai = cellInfo[ci].particle;
				const CellDecomposition::cell_t celli = cellInfo[ci];
				const int ex_start =
					(numExcludes > 0 && excludeMap != NULL) ? excludeMap[ai - repID * num].x : -1;
				const int ex_end =
					(numExcludes > 0 && excludeMap != NULL) ? excludeMap[ai - repID * num].y : -1;
				for (int x = -1; x <= 1; ++x) {
					for (int y = -1; y <= 1; ++y) {
						for (int z = -1; z <= 1; ++z) {
							const int nID = decomp->getNeighborID(celli, x, y, z);
							// const int nID = CellNeighborsList[x+27*cID];//elli.id];
							if (nID < 0)
								continue; // Initialize exclusions
							// TODO: optimize exclusion code (and entire kernel)
							int currEx = ex_start;
							int nextEx = (ex_start >= 0) ? excludes[currEx].ind2 : -1;
							// int ajLast = -1; // TODO: remove this sanity check
							const CellDecomposition::range_t range = decomp->getRange(nID, repID);
							for (int n = range.first + tid; n < range.last; n += blockDim.x) {
								const int aj = cellInfo[n].particle;
								if (aj <= ai)
									continue;
								// Skip excludes
								// Implementation requires that aj increases monotonically
								// assert( ajLast < aj ); ajLast = aj; // TODO: remove this sanity
								// check
								while (nextEx >= 0 &&
									   nextEx < (aj - repID * num)) // TODO get rid of this
									nextEx = (currEx < ex_end - 1) ? excludes[++currEx].ind2 : -1;
								if (nextEx == (aj - repID * num)) {
#ifdef DEBUGEXCLUSIONS
									atomicAggInc(exSum, warpLane);
#endif
									nextEx = (currEx < ex_end - 1) ? excludes[++currEx].ind2 : -1;
									continue;
								}
								// TODO: Skip non-interacting types for efficiency
								// Skip ones that are too far away
								const float dr = (sys->wrapDiff(pos[aj] - pos[ai])).length2();
								if (dr > pairlistdist2)
									continue;
								// Add to pairlist
								int gid = atomicAggInc(g_numPairs, warpLane);
								int pairType = type[ai] + type[aj] * numParts;
								g_pair[gid] = make_int2(ai, aj);
								g_pairTabPotType[gid] = pairType;
							}
						}
					}
				}
			}
		}
	}
}

__device__ int pairForceCounter = 0;
__global__ void printPairForceCounter() {
	if (threadIdx.x + blockIdx.x == 0)
		printf("Computed the force for %d pairs\n", pairForceCounter);
}

template<const int BlockSize>
__device__ inline void _computeTabulatedKernel(Vector3* force,
											   const BaseGrid* __restrict__ sys,
											   float cutoff2,
											   const int numPairs,
											   const int2* __restrict__ g_pair,
											   const int* __restrict__ g_pairTabPotType,
											   TabulatedPotential** __restrict__ tablePot,
											   cudaTextureObject_t pairListsTex,
											   cudaTextureObject_t PosTex,
											   cudaTextureObject_t pairTabPotTypeTex) {
	const int tid =
		threadIdx.x + blockDim.x * threadIdx.y + blockDim.x * blockDim.y * threadIdx.z +
		BlockSize * (blockIdx.x + gridDim.x * blockIdx.y + gridDim.x * gridDim.y * blockIdx.z);

	const int TotalThreads = BlockSize * gridDim.x * gridDim.y * gridDim.z;
	for (int i = tid; i < numPairs; i += TotalThreads) {
		// int2 pair = g_pair[i];
		int2 pair = tex1Dfetch<int2>(pairListsTex, i);
		// int  ind  = tex1Dfetch(pairTabPotTypeTex,i);

		int ai = pair.x;
		int aj = pair.y;

		// int ind = g_pairTabPotType[i];

		Vector3 a(tex1Dfetch<float4>(PosTex, ai));
		Vector3 b(tex1Dfetch<float4>(PosTex, aj));
		Vector3 dr = sys->wrapDiff(b - a);

		float d2 = dr.length2();
		int ind = tex1Dfetch<int>(pairTabPotTypeTex, i);
		if (tablePot[ind] != NULL && d2 <= cutoff2) {
			Vector3 f = tablePot[ind]->computef(dr, d2);
			atomicAdd(&force[ai], f);
			atomicAdd(&force[aj], -f);
		}
	}
}

template<const int BlockSize>
__global__ void computeTabulatedKernel(Vector3* force,
									   const BaseGrid* __restrict__ sys,
									   float cutoff2,
									   const int* __restrict__ g_numPairs,
									   const int2* __restrict__ g_pair,
									   const int* __restrict__ g_pairTabPotType,
									   TabulatedPotential** __restrict__ tablePot,
									   cudaTextureObject_t pairListsTex,
									   cudaTextureObject_t PosTex,
									   cudaTextureObject_t pairTabPotTypeTex) {
	_computeTabulatedKernel<BlockSize>(force,
									   sys,
									   cutoff2,
									   *g_numPairs,
									   g_pair,
									   g_pairTabPotType,
									   tablePot,
									   pairListsTex,
									   PosTex,
									   pairTabPotTypeTex);
}

template<const int BlockSize>
__global__ void computeTabulatedKernel(Vector3* force,
									   const BaseGrid* __restrict__ sys,
									   float cutoff2,
									   const int2* __restrict__ g_pair,
									   const int* __restrict__ g_pairTabPotType,
									   TabulatedPotential** __restrict__ tablePot,
									   cudaTextureObject_t pairListsTex,
									   cudaTextureObject_t PosTex,
									   cudaTextureObject_t pairTabPotTypeTex,
									   int start,
									   int numPairs) {
	_computeTabulatedKernel<BlockSize>(force,
									   sys,
									   cutoff2,
									   numPairs,
									   g_pair + start,
									   g_pairTabPotType + start,
									   tablePot,
									   pairListsTex,
									   PosTex,
									   pairTabPotTypeTex);
}

__global__ void clearEnergies(float* __restrict__ g_energies, int num) {
	for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < num; i += blockDim.x * gridDim.x) {
		g_energies[i] = 0.0f;
	}
}

__global__ void computeTabulatedEnergyKernel(Vector3* force,
											 const Vector3* __restrict__ pos,
											 const BaseGrid* __restrict__ sys,
											 float cutoff2,
											 const int* __restrict__ g_numPairs,
											 const int2* __restrict__ g_pair,
											 const int* __restrict__ g_pairTabPotType,
											 TabulatedPotential** __restrict__ tablePot,
											 float* g_energies) {
	const int numPairs = *g_numPairs;
	for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < numPairs; i += blockDim.x * gridDim.x) {
		const int2 pair = g_pair[i];
		const int ai = pair.x;
		const int aj = pair.y;
		const int ind = g_pairTabPotType[i];

		// RBTODO: implement wrapDiff2, returns dr2 (???)
		Vector3 dr = pos[aj] - pos[ai];
		dr = sys->wrapDiff(dr);
		float d2 = dr.length2();
		// RBTODO: order pairs according to distance to reduce divergence // not actually faster

		if (tablePot[ind] != NULL && d2 <= cutoff2) {
			EnergyForce fe = tablePot[ind]->compute(dr, d2);
			atomicAdd(&force[ai], fe.f);
			atomicAdd(&force[aj], -fe.f);
			// RBTODO: reduce energies
			atomicAdd(&(g_energies[ai]), fe.e * 0.5f);
			atomicAdd(&(g_energies[aj]), fe.e * 0.5f);
		}
	}
}
