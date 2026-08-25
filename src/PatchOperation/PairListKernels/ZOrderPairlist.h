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

#include "../ZOrderKernels/ZOrderSort.h"
#include "PatchOperation/Pairlist.h"
#include "Types/Types.h"

namespace MARS {

/**
 * @brief Z-order based pairlist implementation
 * @todo Make it fused.
 *
 * Uses Morton code sorting to organize particles spatially, then leverages
 * the spatial locality property of Z-order curves to efficiently find
 * neighbors.
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
	 * @param pairlist_cutoff Pairlist search radius (force cutoff + skin)
	 */
	void build_pairlist(const DeviceBuffer<Vector3>& positions,
						size_t num_particles,
						float pairlist_cutoff) override;

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
	PairlistBuilderType get_type() const override {
		return PairlistBuilderType::ZOrder;
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
	 * @brief Set displacement thresholds for intelligent updates
	 * @param validation_threshold Threshold for Morton code validation
	 * @param update_threshold Threshold for full rebuild
	 */
	void set_displacement_thresholds(float validation_threshold, float update_threshold) {
		sorter_.set_displacement_thresholds(validation_threshold, update_threshold);
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

	// Persistent buffers for bounding box computation (avoid recreation)
	mutable DeviceBuffer<Vector3> persistent_bbox_min_; ///< Persistent bounding box minimum buffer
	mutable DeviceBuffer<Vector3> persistent_bbox_max_; ///< Persistent bounding box maximum buffer

	/// Max coarse-cell bits/dim; caps cell arrays at 8^7 entries. See dev_notes.md.
	static constexpr int kMaxCoarseBits = 7;
	/// Initial allocation for the cell index, grown on demand.
	static constexpr size_t kInitialCoarseCells = 4096;

	/// Coarse-cell index over the Morton-sorted array, used by the exact
	/// 27-cell neighbor search. Sized 8^coarse_bits_ and rebuilt each pass.
	DeviceBuffer<uint32_t> cell_begin_;
	DeviceBuffer<uint32_t> cell_end_;
	/// Per-cell neighbor table [num_cells * MAX_NEIGHBORS]. Topology depends only on
	/// the grid, so it is rebuilt only when coarse_bits_/periodicity change (~patch init).
	DeviceBuffer<uint32_t> cell_neighbors_;
	int cell_neighbors_bits_ = -1;	  ///< coarse_bits_ the table was built for (-1 = unbuilt)
	int cell_neighbors_permask_ = -1; ///< periodicity mask the table was built for
	int coarse_bits_ = 0;			  ///< Coarse cells per dimension = 2^coarse_bits_
	Vector3 box_len_{0.0f};			  ///< Per-axis periodic length; <= 0 marks an open axis
	Vector3 box_origin_{0.0f};		  ///< Lower corner of the periodic box
	Vector3 last_box_extent_{0.0f};	  ///< Extent of the box used for the last Morton encoding

  public:
	/**
	 * @brief Declare which axes of the simulation box are periodic.
	 *
	 * Per axis via box_len: a positive component wraps that axis (minimum image)
	 * so cross-boundary pairs are enumerated; zero leaves it open. On a periodic
	 * axis the Morton box is forced to [origin, origin+length), not the bbox.
	 *
	 * @param box_len Per-axis periodic lengths; a zero component marks that
	 *        axis open, and a zero vector disables periodicity entirely.
	 * @param box_origin Lower corner of the periodic box (PeriodicBox::get_origin()).
	 */
	void set_periodic_box(const Vector3& box_len, const Vector3& box_origin = Vector3(0.0f)) {
		box_len_ = box_len;
		box_origin_ = box_origin;
	}

  private:
	// Configuration parameters
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

} // namespace MARS
