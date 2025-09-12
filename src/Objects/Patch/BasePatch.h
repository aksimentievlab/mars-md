/*********************************************************************
 * @file  Patch.h
 *
 * @brief Declaration of BasePatchOp class.
 *
 * @details This file contains the declaration of the abstract base
 *          class BasePatchOp, which operates on Patch data. It also
 *          includes headers of derived classes for convenient access
 *          to factory methods.
 *********************************************************************/

#pragma once

#include "Types/Types.h"

namespace ARBD {

class BasePatch {
  public:
	BasePatch() : num(0), capacity(0), patch_idx(++global_patch_idx) {}
	BasePatch(idx_t capacity) : num(0), capacity(capacity), patch_idx(++global_patch_idx) {}
	BasePatch(idx_t num, short thread_id, short gpu_id);
	// BasePatch(idx_t num, short thread_id, short gpu_id);
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
		return *this;
	}

	~BasePatch() {
		LOGTRACE("Destroying {} @{}",
				 type_name<decltype(*this)>().c_str(),
				 static_cast<void*>(this));
	}
	/**
	 * @brief Metadata for this patch.
	 * @details This is used to store the metadata for this patch.
	 */
	struct Metadata {
		idx_t num;
		idx_t capacity;
		Vector3 min;
		Vector3 max;
	};

  protected:
	idx_t capacity;
	idx_t num;
	short thread_id;			   // MPI
	short gpu_id;				   // -1 if GPU unavailable
	static idx_t global_patch_idx; // Unique ID across ranks
	idx_t patch_idx;			   // Unique ID across each node
	Metadata metadata;
};

} // namespace ARBD
