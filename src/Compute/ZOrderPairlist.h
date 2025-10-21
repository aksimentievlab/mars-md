#pragma once

/*********************************************************************
 * @file  ZOrderPairlist.h
 *
 * @brief Z-order (Morton) based pairlist for improved cache locality
 *
 * This class implements neighbor finding using Z-order spatial sorting
 * to improve memory access patterns and cache performance compared to
 * traditional cell-based approaches.
 *********************************************************************/

#include "Compute/Pairlist.h"
#include "System/ZOrderKernels.h"
#include "System/ZOrderSort.h"
#include "Types/Types.h"

namespace ARBD {

/**
 * @brief Z-order based pairlist implementation
 *
 * Uses Morton code sorting to organize particles spatially, then leverages
 * the spatial locality property of Z-order curves to efficiently find
 * neighbors. This approach often provides better cache performance than
 * cell decomposition, especially for irregular particle distributions.
 */
class ZOrderPairlist : public Pairlist {
  public:
	/**
	 * @brief Constructor
	 * @param resource Computing resource for this pairlist
	 * @param max_particles Maximum number of particles to handle
	 * @param max_pairs Maximum number of particle pairs
	 */
	ZOrderPairlist(const Resource& resource, size_t max_particles, size_t max_pairs);

	/**
	 * @brief Destructor
	 */
	~ZOrderPairlist() override = default;

	/**
	 * @brief Build pairlist using Z-order sorting
	 * @param positions Particle positions
	 * @param num_particles Number of particles
	 * @param cutoff Interaction cutoff distance
	 */
	void build_pairlist(const DeviceBuffer<Vector3>& positions,
						size_t num_particles,
						float cutoff) override;

	/**
	 * @brief Update pairlist (rebuilds with current sort if particles haven't moved much)
	 * @param positions Updated particle positions
	 * @param num_particles Number of particles
	 */
	void update_pairlist(const DeviceBuffer<Vector3>& positions, size_t num_particles) override;

	/**
	 * @brief Check if pairlist needs updating based on particle displacement
	 * @param positions Current particle positions
	 * @param old_positions Previous particle positions
	 * @param num_particles Number of particles
	 * @param skin_distance Skin distance for update criterion
	 * @return True if update is needed
	 */
	bool needs_update(const DeviceBuffer<Vector3>& positions,
					  const DeviceBuffer<Vector3>& old_positions,
					  size_t num_particles,
					  float skin_distance) const override;

	/**
	 * @brief Get pairlist type
	 */
	PairlistType get_type() const override {
		return PairlistType::ZOrder;
	}

	/**
	 * @brief Get implementation name
	 */
	const char* get_name() const override {
		return "Z-Order Pairlist";
	}

	/**
	 * @brief Resize buffers
	 * @param new_max_particles New maximum particle count
	 * @param new_max_pairs New maximum pair count
	 */
	void resize(size_t new_max_particles, size_t new_max_pairs) override;

	/**
	 * @brief Get detailed statistics including Z-order specific metrics
	 */
	Statistics get_statistics() const override;

	/**
	 * @brief Get access to the underlying Z-order sorter
	 * Useful for other algorithms that want to reuse the sorted data
	 */
	const ZOrderSort& get_sorter() const {
		return sorter_;
	}

	/**
	 * @brief Get sorted particle positions
	 * These positions are ordered by Morton code for cache-friendly access
	 */
	const DeviceBuffer<Vector3>& get_sorted_positions() const {
		return sorted_positions_;
	}

	/**
	 * @brief Set search range for neighbor finding
	 * @param range Number of positions to search in each direction in sorted order
	 *
	 * Larger values find more neighbors but are slower. Typical values: 32-128.
	 * The optimal value depends on particle density and cutoff distance.
	 */
	void set_search_range(size_t range) {
		search_range_ = range;
	}

	/**
	 * @brief Get current search range
	 */
	size_t get_search_range() const {
		return search_range_;
	}

	/**
	 * @brief Enable/disable automatic bounding box computation
	 * @param enable If true, compute bounding box from particles
	 * @param box_min Manual minimum bounds (used if enable=false)
	 * @param box_max Manual maximum bounds (used if enable=false)
	 */
	void set_bounding_box_mode(bool enable,
							   const Vector3& box_min = Vector3(0.0f),
							   const Vector3& box_max = Vector3(1.0f)) {
		auto_bbox_ = enable;
		manual_box_min_ = box_min;
		manual_box_max_ = box_max;
	}

  private:
	ZOrderSort sorter_;						 ///< Z-order sorting utility
	DeviceBuffer<Vector3> sorted_positions_; ///< Positions sorted by Morton code
	DeviceBuffer<Vector3> old_positions_;	 ///< Previous positions for update detection

	// Configuration parameters
	size_t search_range_;	 ///< Search range in sorted order
	bool auto_bbox_;		 ///< Whether to auto-compute bounding box
	Vector3 manual_box_min_; ///< Manual bounding box minimum
	Vector3 manual_box_max_; ///< Manual bounding box maximum

	// Timing and statistics
	mutable double last_build_time_ms_;
	mutable size_t last_max_neighbors_;

	/**
	 * @brief Find neighbors using Z-order locality
	 * @param num_particles Number of particles to process
	 */
	void find_neighbors_zorder(size_t num_particles);

	/**
	 * @brief Compute or use manual bounding box
	 * @param positions Particle positions
	 * @param num_particles Number of particles
	 * @param box_min Output minimum bounds
	 * @param box_max Output maximum bounds
	 */
	void get_bounding_box(const DeviceBuffer<Vector3>& positions,
						  size_t num_particles,
						  Vector3& box_min,
						  Vector3& box_max) const;
};

} // namespace ARBD
