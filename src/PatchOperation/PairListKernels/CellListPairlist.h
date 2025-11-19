#pragma once

/*********************************************************************
 * @file  CellListPairlist.h
 *
 * @brief Traditional cell-based pairlist implementation
 *
 * This class implements neighbor finding using cell decomposition,
 * reusing the existing DecomposeKernel infrastructure for optimal
 * performance with uniform particle distributions.
 *********************************************************************/

#include "PatchOperation/DecomposeKernels.h"
#include "PatchOperation/Pairlist.h"
#include "Types/Types.h"

#ifdef USE_CUDA
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#endif

namespace ARBD {

/**
 * @brief Cell-based pairlist implementation
 *
 * Uses traditional cell decomposition to organize particles spatially.
 * This approach is well-suited for uniform particle distributions and
 * provides good performance for most MD simulations.
 */
class CellListPairlist : public Pairlist {
  public:
	/**
	 * @brief Constructor
	 * @param resource Computing resource for this pairlist
	 * @param max_particles Maximum number of particles to handle
	 * @param max_pairs Maximum number of particle pairs
	 * @param num_replicas Number of replica systems (default: 1)
	 */
	CellListPairlist(const Resource& resource,
					 size_t max_particles,
					 size_t max_pairs,
					 int num_replicas = 1);

	/**
	 * @brief Destructor
	 */
	~CellListPairlist() override = default;

	/**
	 * @brief Build pairlist using cell decomposition
	 * @param positions Particle positions
	 * @param num_particles Number of particles
	 * @param cutoff Interaction cutoff distance
	 */
	void build_pairlist(const DeviceBuffer<Vector3>& positions,
						size_t num_particles,
						float cutoff) override;

	/**
	 * @brief Update pairlist (rebuilds cell decomposition)
	 * @param positions Updated particle positions
	 * @param num_particles Number of particles
	 */
	void update_pairlist(const DeviceBuffer<Vector3>& positions, size_t num_particles) override;

	/**
	 * @brief Check if pairlist needs updating
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
		return PairlistBuilderType::CellList;
	}

	/**
	 * @brief Get implementation name
	 */
	const char* get_name() const override {
		return "Cell List Pairlist";
	}

	/**
	 * @brief Resize buffers
	 * @param new_max_particles New maximum particle count
	 * @param new_max_pairs New maximum pair count
	 */
	void resize(size_t new_max_particles, size_t new_max_pairs) override;

	/**
	 * @brief Get detailed statistics
	 */
	Statistics get_statistics() const override;

	/**
	 * @brief Set cell decomposition parameters
	 * @param origin Origin point for the cell grid
	 * @param box_size Size of the simulation box
	 */
	void set_decomposition_params(const Vector3& origin, const Vector3& box_size) {
		origin_ = origin;
		box_size_ = box_size;
	}

	/**
	 * @brief Get current cell grid dimensions
	 */
	Vector3_t<int> get_cell_dimensions() const {
		return n_cells_;
	}

	/**
	 * @brief Get cell size
	 */
	float get_cell_size() const {
		return cell_size_;
	}

	/**
	 * @brief Set number of replicas
	 * @param num_replicas Number of replica systems
	 */
	void set_num_replicas(int num_replicas) {
		num_replicas_ = num_replicas;
	}

  private:
	int num_replicas_;		 ///< Number of replica systems
	Vector3 origin_;		 ///< Origin of cell grid
	Vector3 box_size_;		 ///< Size of simulation box
	Vector3_t<int> n_cells_; ///< Number of cells in each dimension
	float cell_size_;		 ///< Size of each cell
	size_t total_cells_;	 ///< Total number of cells

	// Cell decomposition data
	DeviceBuffer<DecomposeKernel::cell_t> cells_;		  ///< Cell assignments for particles
	DeviceBuffer<BindRangesKernel::range_t> cell_ranges_; ///< Start/end ranges for each cell
	DeviceBuffer<int> temp_ranges_;						  ///< Temporary array for range computation

	// Timing and statistics
	mutable double last_build_time_ms_;
	mutable size_t cells_used_;
	mutable float average_particles_per_cell_;

	/**
	 * @brief Compute optimal cell grid dimensions
	 * @param cutoff Interaction cutoff distance
	 */
	void compute_cell_grid(float cutoff);

	/**
	 * @brief Decompose particles into cells
	 * @param positions Particle positions
	 * @param num_particles Number of particles
	 */
	void decompose_particles(const DeviceBuffer<Vector3>& positions, size_t num_particles);

	/**
	 * @brief Sort particles by cell ID
	 */
	void sort_particles_by_cell();

	/**
	 * @brief Create cell ranges after sorting
	 */
	void create_cell_ranges();

	/**
	 * @brief Find neighbors using cell decomposition
	 * @param positions Particle positions
	 * @param num_particles Number of particles
	 */
	void find_neighbors_celllist(const DeviceBuffer<Vector3>& positions, size_t num_particles);
};

} // namespace ARBD
