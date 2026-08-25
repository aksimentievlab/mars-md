#pragma once
/**
 * @file System/PatchDecomposer.h
 * @author Pin-Yi Li <pinyili2@illinois.edu>
 * @brief Base patch decomposer interfaces for DeviceParticle system
 * @version 2.0
 * @date 2025-09-09
 *
 * Contains abstract base classes for patch decomposition algorithms.
 * Concrete implementations are in Decomposers.h/cpp to avoid circular dependencies.
 *
 * @copyright Copyright (c) 2025
 */

#include "MARSException.h"
#include "MARSLogger.h"
#include "Backend/Resource.h"
#include "Objects/ParticleProperties.h"
#include "SimParam.h"
#include "System/PeriodicBox.h"
#include "Types/Types.h"
#include <array>
#include <memory>
#include <string>
#include <vector>

namespace MARS {

// Forward declarations
class SimSystem;
class SystemState;

/**
 * @brief Statistics from decomposition process
 */
struct DecompositionStats {
	float decomposition_time_ms = 0.0f;		 ///< Time taken for decomposition
	float load_balance_factor = 1.0f;		 ///< Load imbalance metric
	float communication_volume = 0.0f;		 ///< Estimated communication overhead
	size_t total_particles = 0;				 ///< Total particles decomposed
	std::vector<size_t> particles_per_patch; ///< Particle distribution
};

/**
 * @brief Result of patch decomposition with DeviceParticle integration
 *
 * This structure encapsulates spatial decomposition results and provides
 * all information needed by PatchManager to create patches with proper
 * resource binding and particle distribution.
 */
struct DecompositionPlan {
	// Grid structure
	std::array<int, 3> grid_dimensions = {1, 1, 1};		  ///< Number of patches (nx, ny, nz)
	std::array<bool, 3> periodicity = {true, true, true}; ///< Periodic boundary flags

	// Patch boundaries (one per patch in the grid)
	std::vector<Vector3> patch_min_bounds; ///< Minimum bounds for each patch
	std::vector<Vector3> patch_max_bounds; ///< Maximum bounds for each patch
	std::vector<PeriodicBox> patch_boxes;  ///< Periodic boxes for each patch

	// Resource assignment
	std::vector<Resource> patch_resources; ///< Resource assigned to each patch

	// System-wide information
	Vector3 system_min = Vector3(0.0f); ///< Global system minimum bounds
	Vector3 system_max = Vector3(0.0f); ///< Global system maximum bounds
	PeriodicBox system_box;				///< System periodic box probalby needs a const

	// Particle assignment (for non-regular decompositions)
	std::vector<std::vector<idx_t>> patch_particle_indices; ///< Particle indices per patch

	// Performance and validation
	DecompositionStats statistics; ///< Decomposition performance metrics

	/**
	 * @brief Get total number of patches
	 */
	size_t total_patches() const {
		return static_cast<size_t>(grid_dimensions[0]) * static_cast<size_t>(grid_dimensions[1]) *
			   static_cast<size_t>(grid_dimensions[2]);
	}
	/**
	 * @brief Populate per-patch bounds and periodic boxes from the grid.
	 */
	void set_periodic_box() {
		const Vector3 origin = system_box.get_origin();
		const auto& basis = system_box.get_basis();
		const bool* sys_per = system_box.get_periodicity();

		const int nx = grid_dimensions[0] > 0 ? grid_dimensions[0] : 1;
		const int ny = grid_dimensions[1] > 0 ? grid_dimensions[1] : 1;
		const int nz = grid_dimensions[2] > 0 ? grid_dimensions[2] : 1;
		const size_t n = total_patches();

		// System bounds span the full basis from the origin.
		// NOTE: For triclinic boxes, this is the opposite vertex of the parallelipiped,
		// NOT necessarily the maximum Cartesian bounding box extent!
		system_min = origin;
		system_max = origin + basis.ex() + basis.ey() + basis.ez();

		// A split axis is non-periodic per patch.
		const bool per_x = sys_per[0] && nx == 1;
		const bool per_y = sys_per[1] && ny == 1;
		const bool per_z = sys_per[2] && nz == 1;

		// Fill regular-grid bounds only if a decomposer left them empty
		const bool have_bounds = patch_min_bounds.size() == n && patch_max_bounds.size() == n;
		if (!have_bounds) {
			patch_min_bounds.clear();
			patch_max_bounds.clear();
			patch_min_bounds.reserve(n);
			patch_max_bounds.reserve(n);
			for (int k = 0; k < nz; ++k)
				for (int j = 0; j < ny; ++j)
					for (int i = 0; i < nx; ++i) {
						patch_min_bounds.push_back(origin +
												   basis.ex() * (static_cast<mars_real>(i) / nx) +
												   basis.ey() * (static_cast<mars_real>(j) / ny) +
												   basis.ez() * (static_cast<mars_real>(k) / nz));
						patch_max_bounds.push_back(
							origin + basis.ex() * (static_cast<mars_real>(i + 1) / nx) +
							basis.ey() * (static_cast<mars_real>(j + 1) / ny) +
							basis.ez() * (static_cast<mars_real>(k + 1) / nz));
					}
		}

		patch_boxes.clear();
		patch_boxes.reserve(n);
		if (system_box.is_triclinic()) {
			// Uniform per-patch basis for the regular grid (constant across patches).
			const Vector3 pv1 = basis.ex() / static_cast<mars_real>(nx);
			const Vector3 pv2 = basis.ey() / static_cast<mars_real>(ny);
			const Vector3 pv3 = basis.ez() / static_cast<mars_real>(nz);
			for (size_t p = 0; p < n; ++p) {
				PeriodicBox pbox;
				pbox.set_origin(patch_min_bounds[p]);

				if (!have_bounds) {
					pbox.set_basis(pv1, pv2, pv3);
				} else {
					// Irregular cuts: solve h * s = diff for the fractional spans s.
					// Columns are upper-triangular, so back-substitute z -> y -> x.
					const Vector3 e1 = basis.ex();
					const Vector3 e2 = basis.ey();
					const Vector3 e3 = basis.ez();
					const Vector3 diff = patch_max_bounds[p] - patch_min_bounds[p];
					const mars_real sz = (e3.z != mars_real(0.0)) ? diff.z / e3.z : mars_real(0.0);
					const mars_real sy =
						(e2.y != mars_real(0.0)) ? (diff.y - sz * e3.y) / e2.y : mars_real(0.0);
					const mars_real sx = (e1.x != mars_real(0.0))
											 ? (diff.x - sy * e2.x - sz * e3.x) / e1.x
											 : mars_real(0.0);
					pbox.set_basis(e1 * sx, e2 * sy, e3 * sz);
				}

				pbox.set_periodicity(per_x, per_y, per_z);
				patch_boxes.push_back(pbox);
			}
		} else {
			// Orthogonal fast-path: extent is the per-axis span (regular or irregular).
			for (size_t p = 0; p < n; ++p) {
				PeriodicBox pbox;
				pbox.set_box_size(patch_max_bounds[p] - patch_min_bounds[p]);
				pbox.set_origin(patch_min_bounds[p]);
				pbox.set_periodicity(per_x, per_y, per_z);
				patch_boxes.push_back(pbox);
			}
		}
	}
	/**
	 * @brief Validate that the decomposition plan is consistent
	 */
	bool is_valid() const {
		size_t expected_patches = total_patches();
		return expected_patches > 0 && patch_min_bounds.size() == expected_patches &&
			   patch_max_bounds.size() == expected_patches &&
			   patch_resources.size() == expected_patches;
	}
};

/**
 * @brief Abstract base class for patch decomposition strategies
 *
 * Patch decomposers are responsible for dividing the simulation domain
 * into patches and assigning them to available computational resources
 * (GPUs, MPI ranks, etc.)
 */
class PatchDecomposer {
  public:
	virtual ~PatchDecomposer() = default;

	/**
	 * @brief Get decomposer type identifier
	 */
	DecomposerType get_type() const {
		return type_;
	}

	/**
	 * @brief Perform patch decomposition with DeviceParticle integration
	 *
	 * This method computes spatial decomposition based on:
	 * - Global particle data from SystemState
	 * - System configuration from SimSystem
	 * - Available computational resources
	 * - Optimization criteria (load balance, communication, etc.)
	 *
	 * @param system Simulation system with configuration and resources
	 * @param state System state with global particle data
	 * @return DecompositionPlan for PatchManager initialization
	 */
	virtual DecompositionPlan decompose(SimSystem& system, SystemState& state) = 0;

	/**
	 * @brief Get human-readable decomposer name
	 */
	virtual std::string get_name() const {
		switch (type_) {
		case DecomposerType::Spatial:
			return "Spatial Decomposer";
		case DecomposerType::RecursiveBisection:
			return "Recursive Bisection Decomposer";
		case DecomposerType::Geometric:
			return "Geometric Decomposer";
		case DecomposerType::ZOrder:
			return "Z-Order Decomposer";
		default:
			throw Exception(ExceptionType::ValueError,
							SourceLocation(),
							"Unsupported decomposer type");
		}
	}

	/**
	 * @brief Get decomposition statistics (if available)
	 */
	virtual DecompositionStats get_statistics() const {
		return {}; // Default empty stats
	}

  protected:
	DecomposerType type_;

	bool validate_decomposition_plan(const DecompositionPlan& plan) const {
		return plan.is_valid() && plan.patch_min_bounds.size() == plan.patch_max_bounds.size() &&
			   plan.patch_min_bounds.size() == plan.patch_resources.size();
	}

	std::vector<std::vector<idx_t>>
	assign_particles_to_patches(const HostParticleData& particles,
								const std::vector<Vector3>& patch_min_bounds,
								const std::vector<Vector3>& patch_max_bounds) const;
};

} // namespace MARS
