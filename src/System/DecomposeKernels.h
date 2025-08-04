#pragma once

/*********************************************************************
 * @file  CellDecompositionKernels.h
 *
 * @brief Kernel functors for CellDecomposition operations
 *********************************************************************/

 #include "Math/Types.h"
 
 namespace ARBD {
 
 // Forward declaration
 class CellDecomposition;
 
 /*===============================*\
 |   CELLDECOMPOSITION KERNELS    |
 \===============================*/
 
 /**
  * @brief Kernel to decompose particles into cells
  */
 struct DecomposeKernel {
	 Vector3 origin;
	 float cutoff;
	 Vector3_t<int> nCells;
	 int numReplicas;
 
	 // Cell structure (repeated here for kernel use)
	 struct cell_t {
		 int particle;
		 int repID;
		 int id;
		 Vector3_t<int> pos;
		 
		 HOST DEVICE cell_t() : particle(-1), id(-1) {}
		 HOST DEVICE cell_t(int p, int i, const Vector3_t<int>& r, int rep) :
			 particle(p), repID(rep), id(i), pos(r) {}
	 };
 
	 HOST DEVICE void operator()(size_t idx, 
								const Vector3* pos, 
								cell_t* cells) const {
		 size_t particleIdx = idx / numReplicas;
		 int repID = idx % numReplicas;
		 
		 Vector3 r = pos[particleIdx] - origin;
		 int x = int(floorf(r.x / cutoff));
		 int y = int(floorf(r.y / cutoff));
		 int z = int(floorf(r.z / cutoff));
		 
		 // Wrap coordinates (periodic boundary conditions)
		 x = ((x % nCells.x) + nCells.x) % nCells.x;
		 y = ((y % nCells.y) + nCells.y) % nCells.y;
		 z = ((z % nCells.z) + nCells.z) % nCells.z;
		 
		 int cellID = z + nCells.z * (y + nCells.y * x);
		 
		 cells[idx] = cell_t(particleIdx, cellID, Vector3_t<int>{x, y, z}, repID);
	 }
 };
 
 /**
  * @brief Kernel to create range markers from sorted cells
  */
 struct MakeRangesKernel {
	 int numCells;
	 int numReplicas;
 
	 // Use same cell_t structure
	 using cell_t = DecomposeKernel::cell_t;
 
	 HOST DEVICE void operator()(size_t idx,
								const cell_t* cells,
								int* tmp,
								size_t totalElements) const {
		 if (idx >= totalElements) return;
		 
		 // Initialize tmp array element
		 if (idx < numCells * numReplicas) {
			 tmp[idx] = -1;
		 }
		 
		 // Mark start of new cell/replica group
		 if (idx < totalElements) {
			 bool isStart = (idx == 0) || 
						   (cells[idx].id != cells[idx-1].id) || 
						   (cells[idx].repID != cells[idx-1].repID);
			 
			 if (isStart) {
				 int cellIndex = cells[idx].id + cells[idx].repID * numCells;
				 if (cellIndex < numCells * numReplicas) {
					 tmp[cellIndex] = idx;
				 }
			 }
		 }
	 }
 };
 
 /**
  * @brief Kernel to bind ranges from tmp array
  */
 struct BindRangesKernel {
	 int numCells;
	 int numReplicas;
 
	 // Range structure (repeated here for kernel use)
	 struct range_t {
		 int first, last;
		 HOST DEVICE range_t() : first(-1), last(-1) {}
		 HOST DEVICE range_t(int f, int l) : first(f), last(l) {}
	 };
 
	 // Use same cell_t structure
	 using cell_t = DecomposeKernel::cell_t;
 
	 HOST DEVICE void operator()(size_t idx,
								range_t* ranges,
								const int* tmp,
								const cell_t* cells,
								size_t totalElements) const {
		 if (idx >= numCells * numReplicas) return;
		 
		 ranges[idx] = range_t(); // Initialize to invalid range
		 
		 if (tmp[idx] != -1) {
			 int start = tmp[idx];
			 int end = start;
			 
			 // Find end of range by scanning forward
			 while (end < totalElements - 1) {
				 if (cells[end + 1].id != cells[start].id ||
					 cells[end + 1].repID != cells[start].repID) {
					 break;
				 }
				 end++;
			 }
			 ranges[idx] = range_t(start, end + 1);
		 }
	 }
 };
 
 /**
  * @brief Kernel to validate cell decomposition (for debugging)
  */
 struct ValidateDecompositionKernel {
	 Vector3 origin;
	 float cutoff;
	 Vector3_t<int> nCells;
 
	 using cell_t = DecomposeKernel::cell_t;
 
	 HOST DEVICE void operator()(size_t idx,
								const Vector3* pos,
								const cell_t* cells,
								int* errors) const {
		 const cell_t& cell = cells[idx];
		 
		 if (cell.particle >= 0) {
			 Vector3 r = pos[cell.particle] - origin;
			 
			 // Recalculate expected cell position
			 int expected_x = int(floorf(r.x / cutoff));
			 int expected_y = int(floorf(r.y / cutoff));
			 int expected_z = int(floorf(r.z / cutoff));
			 
			 expected_x = ((expected_x % nCells.x) + nCells.x) % nCells.x;
			 expected_y = ((expected_y % nCells.y) + nCells.y) % nCells.y;
			 expected_z = ((expected_z % nCells.z) + nCells.z) % nCells.z;
			 
			 int expected_id = expected_z + nCells.z * (expected_y + nCells.y * expected_x);
			 
			 // Check for mismatch
			 if (cell.id != expected_id || 
				 cell.pos.x != expected_x ||
				 cell.pos.y != expected_y ||
				 cell.pos.z != expected_z) {
				 atomicAdd(errors, 1);
			 }
		 }
	 }
 };
 
 /**
  * @brief Kernel to count particles per cell (for analysis)
  */
 struct CountParticlesKernel {
	 int numCells;
	 int numReplicas;
 
	 using range_t = BindRangesKernel::range_t;
 
	 HOST DEVICE void operator()(size_t idx,
								const range_t* ranges,
								int* counts) const {
		 if (idx >= numCells * numReplicas) return;
		 
		 const range_t& range = ranges[idx];
		 if (range.first >= 0 && range.last >= 0) {
			 counts[idx] = range.last - range.first;
		 } else {
			 counts[idx] = 0;
		 }
	 }
 };
 
 } // namespace ARBD