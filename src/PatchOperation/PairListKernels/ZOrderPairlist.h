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

  private:
	ZOrderSort sorter_;						 ///< Z-order sorting utility
	DeviceBuffer<Vector3> sorted_positions_; ///< Positions sorted by Morton code

	// Persistent buffers for the particle-extent reduction (avoid recreation)
	mutable DeviceBuffer<Vector3> persistent_bbox_min_;
	mutable DeviceBuffer<Vector3> persistent_bbox_max_;

	/// Particle extent, used as the Morton domain on open axes only. A tight
	/// domain packs the cells better; measured worth far more than the
	/// reduction costs. See dev_notes.md.
	void compute_particle_extent(const DeviceBuffer<Vector3>& positions,
								 size_t num_particles,
								 Vector3& box_min,
								 Vector3& box_max) const;

	/// Per-axis periodic lengths in the form the cell kernels expect: a positive
	/// component wraps that axis, zero leaves it open.
	Vector3 periodic_lengths() const {
		const Vector3& bs = box_.get_box_size();
		return Vector3(box_.is_periodic(0) ? bs.x : 0.0f,
					   box_.is_periodic(1) ? bs.y : 0.0f,
					   box_.is_periodic(2) ? bs.z : 0.0f);
	}

	/// Initial allocation for the cell index, grown on demand.
	static constexpr size_t kInitialCellGridCells = 4096;

	/// Cell index over the Morton-sorted array, used by the exact
	/// stencil neighbor search. Sized 8^cell_grid_bits_ and rebuilt each pass.
	DeviceBuffer<uint32_t> cell_begin_;
	DeviceBuffer<uint32_t> cell_end_;
	/// Per-cell neighbor table [num_cells * neighbors_per_cell_]. Topology depends only
	/// on the grid, so it is rebuilt only when the grid changes (~patch init).
	DeviceBuffer<uint32_t> cell_neighbors_;
	int cell_neighbors_bits_ = -1;	  ///< cell_grid_bits_ the table was built for (-1 = unbuilt)
	int cell_neighbors_permask_ = -1; ///< periodicity mask the table was built for
	int3 cell_neighbors_radii_{-1, -1, -1}; ///< stencil radii the table was built for
	int cell_grid_bits_ = 0;				///< Cells per dimension = 2^cell_grid_bits_
	/// Stencil half-width in cells per axis. Morton cells inherit the box's aspect
	/// ratio, so a single radius would oversize the search on anisotropic boxes.
	int3 cell_radii_{1, 1, 1};
	int neighbors_per_cell_ = MAX_NEIGHBORS; ///< row stride of cell_neighbors_
	Vector3 last_box_extent_{0.0f}; ///< Extent of the domain used for the last Morton encoding

	/**
	 * @brief Pick cell_grid_bits_/cell_radii_/neighbors_per_cell_.
	 *
	 * Minimises `scanned_volume * slots` under two parameter-free bounds:
	 * - **at most one cell per particle** — past that the walk pays per-slot cost to
	 *   scan empty cells, which is what makes it saturate;
	 * - the grid's own storage may not exceed the pair buffer it indexes, an
	 *   absurdity guard that should never bind.
	 *
	 * See dev_notes.md.
	 */
	void select_cell_grid(float cutoff, size_t num_particles);

	// Timing and statistics
	mutable double last_build_time_ms_;
	mutable size_t last_max_neighbors_;

	/**
	 * @brief Find neighbors using Z-order locality
	 * @param num_particles Number of particles to process
	 */
	void find_neighbors_zorder(size_t num_particles);
};

} // namespace MARS
