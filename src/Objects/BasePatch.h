/*********************************************************************
 * @file  Patch.h
 *
 * @brief Declaration of BasePatchOp class.
 *
 * @details This file contains the declaration of the abstract base
 *          class BasePatchOp, which operates on Patch data. It also
 *          includes headers of derived classes for convenient access
 *          to factory methods. The virtual method
 *          `BasePatchOp::compute(Patch* patch)` is called by
 *          `patch->compute()` by any Patch to which the PatchOp has
 *          been added.
 *********************************************************************/

#pragma once

#include "Math/Types.h"

namespace ARBD {
class Decomposer;
class CellDecomposer;

class BasePatch {
	friend Decomposer;
	friend CellDecomposer;

  public:
	// BasePatch(size_t num, short thread_id, short gpu_id, SimSystem& sys);
	// BasePatch(size_t num, short thread_id, short gpu_id);
	BasePatch() : num(0), capacity(0), patch_idx(++global_patch_idx) {}
	BasePatch(size_t capacity) : num(0), capacity(capacity), patch_idx(++global_patch_idx) {}

	// Copy constructor
	BasePatch(const BasePatch& other)
		: num(other.num), capacity(other.capacity), patch_idx(++global_patch_idx) {
		LOGTRACE("Copy constructing {} @{}",
				 type_name<decltype(*this)>().c_str(),
				 static_cast<void*>(this));
	}
	// Move constructor
	BasePatch(BasePatch&& other)
		: num(std::move(other.num)), capacity(std::move(other.capacity)),
		  patch_idx(std::move(other.patch_idx)) {
		LOGTRACE("Move constructing {} @{}",
				 type_name<decltype(*this)>().c_str(),
				 static_cast<void*>(this));
	}
	// Move assignment operator
	BasePatch& operator=(BasePatch&& other) {
		LOGTRACE("Move assigning {} @{}",
				 type_name<decltype(*this)>().c_str(),
				 static_cast<void*>(this));
		num = std::move(other.num);
		capacity = std::move(other.capacity);
		patch_idx = std::move(other.patch_idx);
		// lower_bound = std::move(other.lower_bound);
		// upper_bound = std::move(other.upper_bound);
		return *this;
	}

	// ~BasePatch();

  protected:
	size_t capacity;
	size_t num;

	// short thread_id;		// MPI
	// short gpu_id;		// -1 if GPU unavailable

	static size_t global_patch_idx; // Unique ID across ranks // TODO: preallocate
									// regions that will be used, or offload this
									// to a parallel singleton class
	size_t patch_idx;	// Unique ID across ranks

	// Q: should we have different kinds of patches? E.g. Spheres? This
	// specialization _could_ go into subclass, or we could have a ptr to a Region
	// class that can have different implementations
	Vector3 lower_bound;
	Vector3 upper_bound;
};

} // namespace ARBD
