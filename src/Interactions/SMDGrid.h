#pragma once

#include "MARSException.h"
#include "MARSLogger.h"
#include "Types/BaseGrid.h"
#include "Types/Types.h"
#include <memory>
#include <vector>

namespace MARS {

/**
 * @brief SMD Grid functionality for dynamic grid scaling
 *
 * This class provides SMD grid capabilities including:
 * - Dynamic grid scaling during simulation
 * - Time-dependent grid transformations
 * - SMD frequency control
 */
class SMDGrid {
  public:
	/**
	 * @brief Constructor for SMD grid
	 * @param base_grid Base grid to apply SMD scaling to
	 * @param scale_slope Rate of change of grid scale
	 * @param smd_freq Frequency of SMD operations
	 */
	SMDGrid(std::shared_ptr<BaseGrid<mars_real>> base_grid,
			float scale_slope = 0.0f,
			float smd_freq = 0.0f)
		: base_grid_(base_grid), scale_slope_(scale_slope), smd_freq_(smd_freq),
		  current_scale_(1.0f), enabled_(false) {

		if (!base_grid_) {
			throw Exception(ExceptionType::ValueError,
							SourceLocation(),
							"SMDGrid: Invalid base grid provided");
		}

		LOGINFO("SMDGrid: Initialized with scale_slope=%f, smd_freq=%f", scale_slope_, smd_freq_);
	}

	/**
	 * @brief Destructor
	 */
	~SMDGrid() = default;

	// Non-copyable but movable
	SMDGrid(const SMDGrid&) = delete;
	SMDGrid& operator=(const SMDGrid&) = delete;

	SMDGrid(SMDGrid&&) = default;
	SMDGrid& operator=(SMDGrid&&) = default;

	/**
	 * @brief Update grid scale based on current timestep
	 * @param timestep Current timestep
	 */
	void updateScale(int timestep) {
		if (!enabled_)
			return;

		// Update scale based on slope and timestep
		current_scale_ = 1.0f + scale_slope_ * timestep;

		// Ensure scale doesn't become negative or too large
		if (current_scale_ <= 0.0f) {
			current_scale_ = 0.1f; // Minimum scale
			LOGWARN("SMDGrid: Scale became negative, clamping to 0.1");
		} else if (current_scale_ > 10.0f) {
			current_scale_ = 10.0f; // Maximum scale
			LOGWARN("SMDGrid: Scale became too large, clamping to 10.0");
		}

		LOGDEBUG("SMDGrid: Updated scale to %f at timestep %d", current_scale_, timestep);
	}

	/**
	 * @brief Get current grid scale
	 * @return Current scale factor
	 */
	float getCurrentScale() const {
		return current_scale_;
	}

	/**
	 * @brief Get scale slope
	 * @return Scale slope parameter
	 */
	float getScaleSlope() const {
		return scale_slope_;
	}

	/**
	 * @brief Get SMD frequency
	 * @return SMD frequency parameter
	 */
	float getSMDFrequency() const {
		return smd_freq_;
	}

	/**
	 * @brief Set scale slope
	 * @param slope New scale slope
	 */
	void setScaleSlope(float slope) {
		scale_slope_ = slope;
	}

	/**
	 * @brief Set SMD frequency
	 * @param freq New SMD frequency
	 */
	void setSMDFrequency(float freq) {
		smd_freq_ = freq;
	}

	/**
	 * @brief Enable or disable SMD grid
	 * @param enable Whether to enable SMD
	 */
	void setEnabled(bool enable) {
		enabled_ = enable;
	}

	/**
	 * @brief Check if SMD grid is enabled
	 * @return true if enabled
	 */
	bool isEnabled() const {
		return enabled_;
	}

	/**
	 * @brief Get the base grid
	 * @return Shared pointer to base grid
	 */
	std::shared_ptr<BaseGrid<mars_real>> getBaseGrid() const {
		return base_grid_;
	}

	/**
	 * @brief Create SMD grid from existing BaseGrid
	 * @param base_grid Existing BaseGrid to wrap
	 * @param scale_slope Rate of change of grid scale
	 * @param smd_freq Frequency of SMD operations
	 * @return Shared pointer to new SMDGrid
	 */
	static std::shared_ptr<SMDGrid> createFromBaseGrid(std::shared_ptr<BaseGrid<mars_real>> base_grid,
													   float scale_slope = 0.0f,
													   float smd_freq = 0.0f) {
		return std::make_shared<SMDGrid>(base_grid, scale_slope, smd_freq);
	}

	/**
	 * @brief Interpolate force at given position with SMD scaling
	 * @param position World position
	 * @param timestep Current timestep
	 * @return Force at position
	 */
	Vector3 interpolateForce(const Vector3& position, int timestep) {
		if (!enabled_ || !base_grid_) {
			return Vector3(0.0f);
		}

		// Update scale if needed
		updateScale(timestep);

		// Apply SMD scaling to position
		Vector3 scaled_position = applySMDScaling(position);

		// Interpolate force from base grid
		// This would need to be implemented based on the specific grid type
		// For now, return zero force
		LOGDEBUG("SMDGrid: Interpolating force at position (%f, %f, %f) with scale %f",
				 position.x,
				 position.y,
				 position.z,
				 current_scale_);

		return Vector3(0.0f); // Placeholder
	}

	/**
	 * @brief Interpolate potential at given position with SMD scaling
	 * @param position World position
	 * @param timestep Current timestep
	 * @return Potential at position
	 */
	float interpolatePotential(const Vector3& position, int timestep) {
		if (!enabled_ || !base_grid_) {
			return 0.0f;
		}

		// Update scale if needed
		updateScale(timestep);

		// Apply SMD scaling to position
		Vector3 scaled_position = applySMDScaling(position);

		// Interpolate potential from base grid
		// This would need to be implemented based on the specific grid type
		// For now, return zero potential
		LOGDEBUG("SMDGrid: Interpolating potential at position (%f, %f, %f) with scale %f",
				 position.x,
				 position.y,
				 position.z,
				 current_scale_);

		return 0.0f; // Placeholder
	}

  private:
	std::shared_ptr<BaseGrid<mars_real>> base_grid_;
	float scale_slope_;
	float smd_freq_;
	float current_scale_;
	bool enabled_;

	/**
	 * @brief Apply SMD scaling to a position
	 * @param position Original position
	 * @return Scaled position
	 */
	Vector3 applySMDScaling(const Vector3& position) {
		if (!base_grid_)
			return position;

		// Get grid center
		Vector3 grid_center = base_grid_->get_center();

		// Apply scaling relative to grid center
		Vector3 relative_pos = position - grid_center;
		Vector3 scaled_pos = relative_pos * current_scale_;

		return grid_center + scaled_pos;
	}
};

/**
 * @brief Manager for multiple SMD grids
 *
 * This class manages multiple SMD grids in the system
 */
class SMDGridManager {
  public:
	/**
	 * @brief Constructor
	 */
	SMDGridManager() : enabled_(false) {
		LOGINFO("SMDGridManager: Initialized");
	}

	/**
	 * @brief Destructor
	 */
	~SMDGridManager() = default;

	// Non-copyable but movable
	SMDGridManager(const SMDGridManager&) = delete;
	SMDGridManager& operator=(const SMDGridManager&) = delete;

	SMDGridManager(SMDGridManager&&) = default;
	SMDGridManager& operator=(SMDGridManager&&) = default;

	/**
	 * @brief Add an SMD grid
	 * @param smd_grid SMD grid to add
	 */
	void addGrid(std::shared_ptr<SMDGrid> smd_grid) {
		if (!smd_grid) {
			throw Exception(ExceptionType::ValueError,
							SourceLocation(),
							"SMDGridManager: Invalid SMD grid provided");
		}

		smd_grids_.push_back(smd_grid);
		LOGINFO("SMDGridManager: Added SMD grid (total: %zu)", smd_grids_.size());
	}

	/**
	 * @brief Get number of SMD grids
	 * @return Number of SMD grids
	 */
	size_t getNumGrids() const {
		return smd_grids_.size();
	}

	/**
	 * @brief Enable or disable SMD grids
	 * @param enable Whether to enable SMD grids
	 */
	void setEnabled(bool enable) {
		enabled_ = enable;
	}

	/**
	 * @brief Check if SMD grids are enabled
	 * @return true if enabled
	 */
	bool isEnabled() const {
		return enabled_;
	}

	/**
	 * @brief Update all SMD grids
	 * @param timestep Current timestep
	 */
	void updateAllGrids(int timestep) {
		if (!enabled_)
			return;

		for (auto& grid : smd_grids_) {
			if (grid && grid->isEnabled()) {
				grid->updateScale(timestep);
			}
		}
	}

	/**
	 * @brief Get SMD grid by index
	 * @param index Grid index
	 * @return SMD grid at index, or nullptr if invalid
	 */
	std::shared_ptr<SMDGrid> getGrid(size_t index) const {
		if (index >= smd_grids_.size()) {
			return nullptr;
		}
		return smd_grids_[index];
	}

  private:
	std::vector<std::shared_ptr<SMDGrid>> smd_grids_;
	bool enabled_;
};

} // namespace MARS
