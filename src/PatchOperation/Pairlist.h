#pragma once

/*********************************************************************
 * @file  Pairlist.h
 *
 * @brief Abstract base class for different pairlist strategies
 *
 * This file contains the abstract interface for pairlist implementations
 * including cell-based, Z-order, and other spatial data structures.
 *********************************************************************/

#include "Backend/Buffer.h"
#include "Backend/Resource.h"
#include "Types/Types.h"
#include "Types/Vector3.h"

namespace ARBD {

/**
 * @brief Enumeration of available pairlist strategies
 */
enum class PairlistBuilderType {
	ZOrder,		 ///< Z-order (Morton) based spatial sorting
	CellList,	 ///< Traditional cell-based neighbor lists
	VerletList,	 ///< Verlet neighbor lists
	Hierarchical ///< Hierarchical spatial structures (future)
};

/**
 * @brief Abstract base class for pairlist implementations
 *
 * Defines the interface that all pairlist strategies must implement.
 * Each strategy optimizes neighbor finding for different scenarios:
 * - CellList: Good for uniform distributions
 * - ZOrder: Better cache locality, good for multi-device
 * - VerletList: Reduced updates for stable systems
 */
class Pairlist {
  public:
	/**
	 * @brief Constructor
	 * @param resource Computing resource for this pairlist
	 * @param max_particles Maximum number of particles to handle
	 * @param max_pairs Maximum number of particle pairs
	 */
	Pairlist(const Resource& resource, size_t max_particles, size_t max_pairs)
		: resource_(resource), max_particles_(max_particles), max_pairs_(max_pairs),
		  num_particles_(0), num_pairs_(0), cutoff_(0.0f), cutoff_squared_(0.0f),
		  neighbor_pairs_(max_pairs, resource), pair_count_(1, resource) {
		uint32_t zero = 0;
		pair_count_.copy_from_host(&zero, 1, true);
	}

	/**
	 * @brief Virtual destructor
	 */
	virtual ~Pairlist() = default;

	/**
	 * @brief Build the pairlist for given particle positions
	 * @param positions Particle positions
	 * @param num_particles Number of particles
	 * @param cutoff Interaction cutoff distance
	 */
	virtual void
	build_pairlist(const DeviceBuffer<Vector3>& positions, size_t num_particles, float cutoff) = 0;

	/**
	 * @brief Update pairlist (may be more efficient than full rebuild)
	 * @param positions Updated particle positions
	 * @param num_particles Number of particles
	 */
	virtual void update_pairlist(const DeviceBuffer<Vector3>& positions, size_t num_particles) {
		// Default implementation: full rebuild
		build_pairlist(positions, num_particles, cutoff_);
	}

	/**
	 * @brief Check if pairlist needs updating based on particle displacement
	 * @param positions Current particle positions
	 * @param old_positions Previous particle positions
	 * @param num_particles Number of particles
	 * @param skin_distance Skin distance for update criterion
	 * @return True if update is needed
	 */
	virtual bool needs_update(const DeviceBuffer<Vector3>& positions,
							  const DeviceBuffer<Vector3>& old_positions,
							  size_t num_particles,
							  float skin_distance) const {
		// Default implementation: always update
		// Derived classes can implement smarter update criteria
		return true;
	}

	/**
	 * @brief Get the neighbor pairs
	 */
	const DeviceBuffer<int2>& get_neighbor_pairs() const {
		return neighbor_pairs_;
	}

	/**
	 * @brief Get current number of pairs
	 */
	size_t get_num_pairs() const {
		return num_pairs_;
	}

	/**
	 * @brief Get current cutoff distance
	 */
	float get_cutoff() const {
		return cutoff_;
	}

	/**
	 * @brief Get pairlist type identifier
	 */
	virtual PairlistBuilderType get_type() const = 0;

	/**
	 * @brief Get implementation name
	 */
	virtual const char* get_name() const = 0;

	/**
	 * @brief Resize buffers for different particle/pair counts
	 * @param new_max_particles New maximum particle count
	 * @param new_max_pairs New maximum pair count
	 */
	virtual void resize(size_t new_max_particles, size_t new_max_pairs) {
		max_particles_ = new_max_particles;
		max_pairs_ = new_max_pairs;
		neighbor_pairs_.resize(max_pairs_);
	}

	/**
	 * @brief Get statistics about the pairlist
	 */
	struct Statistics {
		size_t num_particles;
		size_t num_pairs;
		float cutoff;
		float average_neighbors_per_particle;
		size_t max_neighbors_per_particle;
		double build_time_ms;
	};

	/**
	 * @brief Get pairlist statistics
	 */
	virtual Statistics get_statistics() const {
		Statistics stats;
		stats.num_particles = num_particles_;
		stats.num_pairs = num_pairs_;
		stats.cutoff = cutoff_;
		stats.average_neighbors_per_particle =
			num_particles_ > 0 ? static_cast<float>(num_pairs_) / num_particles_ : 0.0f;
		stats.max_neighbors_per_particle = 0; // Derived classes can override
		stats.build_time_ms = 0.0;			  // Derived classes can track timing
		return stats;
	}

  protected:
	Resource resource_;
	uint32_t max_particles_;
	uint32_t max_pairs_;
	uint32_t num_particles_;
	uint32_t num_pairs_;
	float cutoff_;
	float cutoff_squared_;

	// Core pairlist data.
	//
	// There is deliberately no per-pair distance array. One used to be
	// allocated alongside neighbor_pairs_ and was never written or read by any
	// builder or consumer; at the default capacity that is 4 bytes x 307M pairs
	// = 1.2 GB of device memory reserved for nothing. Recomputing a distance in
	// the force kernel is cheaper than the bandwidth to fetch a stored one, so
	// if a consumer ever needs it, it should recompute rather than reinstate
	// this.
	DeviceBuffer<int2> neighbor_pairs_;	///< Neighbor particle pairs (i, j)
	DeviceBuffer<uint32_t> pair_count_;	///< Atomic counter for pair generation

	/**
	 * @brief Update internal state after pairlist build
	 * @param num_particles Number of particles processed
	 * @param cutoff Cutoff distance used
	 */
	void update_state(size_t num_particles, float cutoff) {
		num_particles_ = num_particles;
		cutoff_ = cutoff;
		cutoff_squared_ = cutoff * cutoff;

		// Get actual number of pairs generated
		pair_count_.copy_to_host_sync(reinterpret_cast<uint32_t*>(&num_pairs_), 1);
	}

	/**
	 * @brief Reset pair counter for new build
	 */
	void reset_pair_count() {
		uint32_t zero = 0;
		pair_count_.copy_from_host(&zero, 1, true);
	}
};

/**
 * @brief Factory function to create pairlist instances
 * @param type Type of pairlist to create
 * @param resource Computing resource
 * @param max_particles Maximum number of particles
 * @param max_pairs Maximum number of pairs
 * @return Unique pointer to the created pairlist
 */
std::unique_ptr<Pairlist> create_pairlist(PairlistBuilderType type,
										  const Resource& resource,
										  size_t max_particles,
										  size_t max_pairs);

} // namespace ARBD
