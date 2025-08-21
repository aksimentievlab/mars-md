#pragma once

#include "ARBDException.h"
#include "ARBDLogger.h"
#include "Backend/Resource.h"
#include "Math/NanoGridHandle.h"
#include <nanovdb/io/IO.h>
#include <string>
#include <vector>

namespace ARBD {
namespace IO {
/**
 * @brief Load NanoVDB grid from file into ARBD buffer system
 */
template<typename ValueT = float>
NanoGridAdapter<ValueT> load_nanovdb_grid(const std::string& filename,
										  const std::string& grid_name = "",
										  const Resource& resource = Resource{ResourceType::CPU,
																			  0}) {

	try {
		LOGINFO("Loading NanoVDB grid from: {}", filename);
		return NanoGridAdapter<ValueT>::from_file(filename, grid_name, resource);
	} catch (const std::exception& e) {
		ARBD_Exception(ExceptionType::RuntimeError,
					   "Failed to load NanoVDB grid from {}: {}",
					   filename,
					   e.what());
	}
}

/**
 * @brief Load multiple grids from a NanoVDB file
 */
template<typename ValueT = float>
std::vector<NanoGridAdapter<ValueT>>
load_nanovdb_grids(const std::string& filename,
				   const Resource& resource = Resource{ResourceType::CPU, 0}) {

	std::vector<NanoGridAdapter<ValueT>> result;

	try {
		auto handles = nanovdb::io::readGrids(filename);
		result.reserve(handles.size());

		for (auto& handle : handles) {
			result.emplace_back(std::move(handle), resource);
		}

		LOGINFO("Loaded {} NanoVDB grids from: {}", result.size(), filename);
		return result;

	} catch (const std::exception& e) {
		ARBD_Exception(ExceptionType::RuntimeError,
					   "Failed to load NanoVDB grids from {}: {}",
					   filename,
					   e.what());
	}
}

/**
 * @brief Query available grids in a NanoVDB file
 */
inline std::vector<std::string> query_nanovdb_grids(const std::string& filename) {
	try {
		auto metadata = nanovdb::io::readGridMetaData(filename);
		std::vector<std::string> grid_names;
		grid_names.reserve(metadata.size());

		for (const auto& meta : metadata) {
			grid_names.push_back(meta.gridName);
		}

		LOGINFO("Found {} grids in {}", grid_names.size(), filename);
		return grid_names;

	} catch (const std::exception& e) {
		ARBD_Exception(ExceptionType::RuntimeError,
					   "Failed to query NanoVDB file {}: {}",
					   filename,
					   e.what());
	}
}

/**
 * @brief Check if a grid exists in a NanoVDB file
 */
inline bool has_nanovdb_grid(const std::string& filename, const std::string& grid_name) {
	try {
		return nanovdb::io::hasGrid(filename, grid_name);
	} catch (const std::exception& e) {
		LOGWARN("Error checking for grid '{}' in {}: {}", grid_name, filename, e.what());
		return false;
	}
}

} // namespace IO
} // namespace ARBD
