/*********************************************************************
 * @file  Patch.h
 *
 * @brief Declaration of PatchManagerOp class.
 *
 * @details This file contains the declaration of the abstract base
 *          class PatchManagerOp, which operates on Patch data. It also
 *          includes headers of derived classes for convenient access
 *          to factory methods. The virtual method
 *          `PatchManagerOp::compute(Patch* patch)` is called by
 *          `patch->compute()` by any Patch to which the PatchOp has
 *          been added.'
 * @todo: Implement neighbour list (v1 cell decomposition)
 *********************************************************************/

#pragma once

#include "Types/Types.h"

namespace ARBD {
class Decomposer;
class CellDecomposer;
class Patch;

class PatchManager {
	friend Decomposer;
	friend CellDecomposer;

  public:
	// PatchManager(idx_t num, short thread_id, short gpu_id, SimSystem& sys);
	// PatchManager(idx_t num, short thread_id, short gpu_id);
	PatchManager() : num(0), capacity(0), patch_idx(++global_patch_idx) {}
	PatchManager(idx_t capacity) : num(0), capacity(capacity), patch_idx(++global_patch_idx) {}

	// Copy constructor
	PatchManager(const PatchManager& other)
		: num(other.num), capacity(other.capacity), patch_idx(++global_patch_idx) {
		LOGTRACE("Copy constructing {} @{}",
				 type_name<decltype(*this)>().c_str(),
				 static_cast<void*>(this));
	}
	// Move constructor
	PatchManager(PatchManager&& other)
		: num(std::move(other.num)), capacity(std::move(other.capacity)),
		  patch_idx(std::move(other.patch_idx)) {
		LOGTRACE("Move constructing {} @{}",
				 type_name<decltype(*this)>().c_str(),
				 static_cast<void*>(this));
	}
	// Move assignment operator
	PatchManager& operator=(PatchManager&& other) {
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

	~PatchManager() {
		LOGTRACE("Destroying {} @{}",
				 type_name<decltype(*this)>().c_str(),
				 static_cast<void*>(this));
	}
	/**
	 * @brief Metadata for each patch.
	 * @details This is used to store the metadata for each patch.
	 */
	struct Metadata {
		Metadata() : num(0), capacity(0), min(Vector3(0.0f)), max(Vector3(0.0f)){};
		Metadata(const PatchManager& p)
			: num(p.num), capacity(p.capacity), min(p.lower_bound), max(p.upper_bound){};
		Metadata(const Metadata& other)
			: num(other.num), capacity(other.capacity), min(other.min), max(other.max){};
		idx_t num;
		idx_t capacity;
		Vector3 min;
		Vector3 max;
	};

	const Vector3 get_min() const {
		Vector3 min(Vector3::highest());
		for (auto& p : patches) {
			if (min.x > p.metadata->min.x)
				min.x = p.metadata->min.x;
			if (min.y > p.metadata->min.y)
				min.y = p.metadata->min.y;
			if (min.z > p.metadata->min.z)
				min.z = p.metadata->min.z;
			if (min.w > p.metadata->min.w)
				min.w = p.metadata->min.w;
		}
		return min;
	}

	const Vector3 get_max() const {
		Vector3 max(Vector3::lowest());
		for (auto& p : patches) {
			if (max.x < p.metadata->max.x)
				max.x = p.metadata->max.x;
			if (max.y < p.metadata->max.y)
				max.y = p.metadata->max.y;
			if (max.z < p.metadata->max.z)
				max.z = p.metadata->max.z;
			if (max.w < p.metadata->max.w)
				max.w = p.metadata->max.w;
		}
		return max;
	}

	// ~PatchManager();

  protected:
	idx_t capacity;
	idx_t num;

	Metadata metadata;
	// short thread_id;		// MPI
	// short gpu_id;		// -1 if GPU unavailable

	static idx_t global_patch_idx; // Unique ID across ranks // TODO: preallocate
								   // regions that will be used, or offload this
								   // to a parallel singleton class
	idx_t patch_idx;			   // Unique ID across ranks

	// Q: should we have different kinds of patches? E.g. Spheres? This
	// specialization _could_ go into subclass, or we could have a ptr to a Region
	// class that can have different implementations
};

} // namespace ARBD
