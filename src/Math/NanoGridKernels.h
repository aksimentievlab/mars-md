#pragma once
#include "Backend/Header.h"
#include "nanovdb/GridHandle.h"

namespace ARBD {

// --- Functor for GridHandle Meta Copy ---
template<typename GridDataT, typename GridHandleMetaDataT>
struct CpyGridHandleMetaFunctor {
	const GridDataT* d_data;
	GridHandleMetaDataT* d_meta;

	HOST DEVICE void
	operator()(size_t i, const GridDataT* d_data, GridHandleMetaDataT* d_meta) const {
		if (i == 0) { // Only one thread needs to do this
			nanovdb::cpyGridHandleMeta(d_data, d_meta);
		}
	}
};

// --- Functor for Grid Count Update ---
template<typename GridDataT>
struct UpdateGridCountFunctor {
	GridDataT* d_data;
	uint32_t gridIndex;
	uint32_t gridCount;
	bool* d_dirty;

	HOST DEVICE void operator()(size_t i,
								GridDataT* d_data,
								uint32_t gridIndex,
								uint32_t gridCount,
								bool* d_dirty) const {
		if (i == 0) { // Only one thread needs to do this
			*d_dirty = (d_data->mGridIndex != gridIndex) || (d_data->mGridCount != gridCount);
			if (*d_dirty) {
				d_data->mGridIndex = gridIndex;
				d_data->mGridCount = gridCount;
				if (d_data->mChecksum.isEmpty()) {
					*d_dirty = false; // no need to update checksum if it didn't already exist
				}
			}
		}
	}
};

} // namespace ARBD
