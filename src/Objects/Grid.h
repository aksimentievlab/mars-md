#pragma once
#include "ARBDLogger.h"
#include "Backend/Buffer.h"
#include "Backend/Resource.h"
#include "Types/BaseGrid.h"
#include "Types/BaseGridDevice.h"
#include "Types/GridTerm.h"
// #include "Types/NanoGridHandle.h"
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace ARBD {

// What a grid is used for - host-side bookkeeping only, so it stays here.
// GridFormat and InterpolationOrder moved to Types/GridTerm.h: device code
// (and, soon, the sparse backend) needs them, and can't include this header.
enum class GridType { Potential, Diffusion, PMF, Force, Density };
/**
 * @brief Lightweight grid identifier with metadata
 * @details Used for configuration and lookup. Does NOT own grid data.
 */
struct GridKey {
	std::string name;
	GridFormat format;
	int grid_id = -1; ///< Index into GridManager's unified grid system
	InterpolationOrder interpolation_order = InterpolationOrder::Linear;

	GridKey() = default;
	GridKey(const std::string& name, const GridFormat& format) : name(name), format(format) {}

	bool operator==(const GridKey& o) const {
		return name == o.name && format == o.format;
	}

	bool is_valid() const {
		return grid_id >= 0;
	}
};

/**
 * @brief Unified grid management system
 * @details Manages both dense (BaseGrid) and sparse (NanoVDB) grids,
 *          providing a unified interface for device-side access across multiple resources.
 *
 * Design:
 * - Host-side: Separate storage for dense/sparse grids (single copy)
 * - Device-side: Replicated on ALL resources for multi-GPU support
 * - GridKey.grid_id maps to unified device arrays
 * - Each resource gets its own device arrays (grid_ptrs, configs)
 */
class GridManager {
  public:
	GridManager() = default;

	/**
	 * @brief Initialize with resources from SimSystem
	 * @param resources Reference to SimSystem's resource vector
	 * @note GridManager does NOT own resources, just references them
	 */
	void set_resources(const std::vector<Resource>* resources) {
		resources_ = resources;
		if (resources_) {
			LOGINFO("GridManager: Configured for {} resources", resources_->size());
		}
	}
	GridKey add_grid(const std::string& name) {
		if (name.find(".dx") != std::string::npos) {
			GridFormat format = GridFormat::Dense;
			return add_dense_grid(name);
		} else {
			GridFormat format = GridFormat::Sparse;
			return add_sparse_grid(name);
		}
	};

	/**
	 * @brief Add a dense grid from file
	 * @param filename Path to .dx file
	 * @param type Grid type (PMF, Diffusion, etc.)
	 * @return GridKey with assigned grid_id
	 */
	GridKey add_dense_grid(const std::string& filename) {
		// Check if already loaded
		auto it = fname_to_gridkey_.find(filename);
		if (it != fname_to_gridkey_.end()) {
			LOGINFO("GridManager: Grid '{}' already loaded (grid_id={})",
					filename,
					it->second.grid_id);
			return it->second;
		}

		// Load grid from file
		BaseGrid<float> grid = DXReader::read_from_file<float>(filename);

		// Assign unified grid_id
		int grid_id = next_grid_id_++;
		dense_grids_.push_back(std::move(grid));
		int dense_idx = dense_grids_.size() - 1;

		// Create GridKey
		GridKey key(filename, GridFormat::Dense);
		key.grid_id = grid_id;

		// Store mappings
		fname_to_gridkey_[filename] = key;
		grid_keys_.push_back(key);
		grid_id_to_dense_idx_[grid_id] = dense_idx;

		LOGINFO("GridManager: Loaded dense grid '{}' → grid_id={}", filename, grid_id);
		return key;
	}

	/**
	 * @brief Add a sparse grid from file
	 * @param filename Path to .nvdb file
	 * @param type Grid type
	 * @return GridKey with assigned grid_id
	 * @note Sparse grids will be replicated to all resources during build_device_arrays()
	 */

	GridKey add_sparse_grid(const std::string& filename) {
		throw_not_implemented("Sparse grids are not supported yet");

		return GridKey(filename, GridFormat::Sparse);
		/*
		auto it = fname_to_gridkey_.find(filename);
		if (it != fname_to_gridkey_.end()) {
			LOGINFO("GridManager: Grid '{}' already loaded (grid_id={})",
					filename,
					it->second.grid_id);
			return it->second;
		}

		// Load sparse grid on first resource(will be replicated later)
		Resource first_resource =
			(resources_ && !resources_->empty()) ? (*resources_)[0] : Resource{};
		auto adapter = NanoGridAdapter<>::from_file(filename, first_resource);

		// Assign unified grid_id
		int grid_id = next_grid_id_++;
		sparse_grids_.push_back(NanoGridAdapter<float>(std::move(adapter), first_resource));
		int sparse_idx = sparse_grids_.size() - 1;
		grid_id_to_sparse_idx_[grid_id] = sparse_idx;

		LOGINFO("GridManager: Loaded sparse grid '{}' → grid_id={}", filename, grid_id);
		return GridKey(filename, GridFormat::Sparse);
		*/
	}
	/**
	 * @brief Get GridKey by filename
	 */
	GridKey get_grid_key(const std::string& filename) const {
		auto it = fname_to_gridkey_.find(filename);
		if (it != fname_to_gridkey_.end()) {
			return it->second;
		}
		return GridKey{}; // Invalid key (grid_id = -1)
	}

	/**
	 * @brief Check if grid is loaded
	 */
	bool has_grid(const std::string& filename) const {
		return fname_to_gridkey_.count(filename) > 0;
	}

	/**
	 * @brief Storage format of a previously-assigned grid_id
	 * @details Used by RigidBodyForcePairList (Phase 3) to enforce the
	 *          format-uniformity rule on grid-grid pairs.
	 */
	GridFormat get_grid_format(int grid_id) const {
		if (grid_id_to_dense_idx_.count(grid_id))
			return GridFormat::Dense;
		if (grid_id_to_sparse_idx_.count(grid_id))
			return GridFormat::Sparse;
		throw_value_error("GridManager: Invalid grid_id: %d", grid_id);
	}

	/**
	 * @brief Get total number of grids
	 */
	size_t num_grids() const {
		return dense_grids_.size(); // + sparse_grids_.size();
	}

	/**
	 * @brief Build unified device arrays for ALL resources
	 * @details Replicates grids to all GPU resources for multi-GPU support.
	 *          Creates flat arrays of grid pointers and configs that kernels can index directly.
	 */
	void build_device_arrays() {
		if (!resources_ || resources_->empty()) {
			LOGWARN("GridManager: No resources configured, using default resource");
			// Create temporary default resource
			std::vector<Resource> default_resources = {Resource{}};
			build_device_arrays_impl(default_resources);
			return;
		}

		build_device_arrays_impl(*resources_);
	}

  private:
	void build_device_arrays_impl(const std::vector<Resource>& resources) {
		LOGINFO("GridManager: Building device arrays for {} grids across {} resources",
				num_grids(),
				resources.size());

		// Clear existing device data
		device_grid_ptrs_per_resource_.clear();
		device_grid_configs_per_resource_.clear();
		device_grid_views_per_resource_.clear();

		// Build device arrays for each resource
		for (size_t res_idx = 0; res_idx < resources.size(); ++res_idx) {
			const auto& resource = resources[res_idx];

			LOGINFO("GridManager: Transferring grids to resource {}", res_idx);

			// Initialize arrays for this resource
			std::vector<float*> grid_ptrs(next_grid_id_, nullptr);
			std::vector<BaseGrid<float>::Config> grid_configs(next_grid_id_);
			std::vector<BaseGridView<float>> grid_views(next_grid_id_);

			for (const auto& [grid_id, dense_idx] : grid_id_to_dense_idx_) {
				auto& grid = dense_grids_[dense_idx];
				grid.sync_to_device(resource);
				grid_ptrs[grid_id] = grid.get_device_pointer(resource);
				grid_configs[grid_id] = grid.config();
				grid_views[grid_id] = grid.get_device_view(resource);
				grid_views[grid_id].grid_id = grid_id;
			}

			/** Transfer sparse grids to this resource
			for (const auto& [grid_id, sparse_idx] : grid_id_to_sparse_idx_) {
				// TODO: Replicate sparse grids to multiple resources
				// For now, use the first resource's sparse grid
				auto& adapter = sparse_grids_[sparse_idx];
				grid_ptrs[grid_id] = reinterpret_cast<float*>(adapter.device_grid<float>());
				// TODO: Extract config from NanoVDB grid
				grid_configs[grid_id] = BaseGrid<float>::Config{};
			}
			*/

			// Store device arrays for this resource
			device_grid_ptrs_per_resource_.push_back(std::move(grid_ptrs));
			device_grid_configs_per_resource_.push_back(std::move(grid_configs));

			DeviceBuffer<BaseGridView<float>> device_views(grid_views.size(), resource);
			if (!grid_views.empty()) {
				device_views.copy_from_host(grid_views.data(), grid_views.size());
			}
			device_grid_views_per_resource_.push_back(std::move(device_views));
		}

		LOGINFO("GridManager: Device arrays built successfully for all resources");
	}

  public:
	/**
	 * @brief Get device grid pointer array for a specific resource
	 * @param resource_idx Index into resources vector
	 */
	const std::vector<float*>& get_device_grid_ptrs(size_t resource_idx = 0) const {
		if (resource_idx >= device_grid_ptrs_per_resource_.size()) {
			throw std::runtime_error("GridManager: Invalid resource index: " +
									 std::to_string(resource_idx));
		}
		return device_grid_ptrs_per_resource_[resource_idx];
	}

	/**
	 * @brief Get device grid config array for a specific resource
	 * @param resource_idx Index into resources vector
	 */
	const std::vector<BaseGrid<float>::Config>&
	get_device_grid_configs(size_t resource_idx = 0) const {
		if (resource_idx >= device_grid_configs_per_resource_.size()) {
			throw std::runtime_error("GridManager: Invalid resource index: " +
									 std::to_string(resource_idx));
		}
		return device_grid_configs_per_resource_[resource_idx];
	}

	/**
	 * @brief Get device BaseGridView array for kernel indexing by grid_id
	 */
	const DeviceBuffer<BaseGridView<float>>& get_device_grid_views(size_t resource_idx = 0) const {
		if (resource_idx >= device_grid_views_per_resource_.size()) {
			throw std::runtime_error("GridManager: Invalid resource index: " +
									 std::to_string(resource_idx));
		}
		return device_grid_views_per_resource_[resource_idx];
	}

	/**
	 * @brief Get all GridKeys
	 */
	const std::vector<GridKey>& get_grid_keys() const {
		return grid_keys_;
	}

	/**
	 * @brief Access dense grid by grid_id
	 */
	BaseGrid<float>& get_dense_grid(int grid_id) {
		auto it = grid_id_to_dense_idx_.find(grid_id);
		if (it == grid_id_to_dense_idx_.end()) {
			throw std::runtime_error("GridManager: Invalid dense grid_id: " +
									 std::to_string(grid_id));
		}
		return dense_grids_[it->second];
	}

	/**
	 * @brief Access sparse grid by grid_id

	NanoGridAdapter<>& get_sparse_grid(int grid_id) {
		auto it = grid_id_to_sparse_idx_.find(grid_id);
		if (it == grid_id_to_sparse_idx_.end()) {
			throw std::runtime_error("GridManager: Invalid sparse grid_id: " +
									 std::to_string(grid_id));
		}
		return sparse_grids_[it->second];
	}
	 */
  private:
	// Host-side storage (separate by format, single copy)
	std::vector<BaseGrid<float>> dense_grids_;
	// std::vector<NanoGridAdapter<float>> sparse_grids_;

	// Metadata and lookup
	std::vector<GridKey> grid_keys_;
	std::unordered_map<std::string, GridKey> fname_to_gridkey_;

	// Mapping from unified grid_id to storage indices
	std::unordered_map<int, int> grid_id_to_dense_idx_;
	std::unordered_map<int, int> grid_id_to_sparse_idx_;

	// Resources (pointer to SimSystem's resource vector - NOT owned)
	const std::vector<Resource>* resources_ = nullptr;

	// Device arrays per resource (outer index = resource_idx, inner index = grid_id)
	std::vector<std::vector<float*>> device_grid_ptrs_per_resource_;
	std::vector<std::vector<BaseGrid<float>::Config>> device_grid_configs_per_resource_;
	std::vector<DeviceBuffer<BaseGridView<float>>> device_grid_views_per_resource_;

	// ID counter
	int next_grid_id_ = 0;
};

} // namespace ARBD
