// RigidBodyForcePairs.h (2026)
// Phase 3 of the rigid-body suite: host-side force-pair construction.
// Port of legacy RigidBodyController::initializeForcePairs.
#pragma once
#include "ARBDException.h"
#include "Objects/RigidBodyProperties.h"
#include "Types/GridTerm.h"
#include <functional>
#include <vector>

namespace ARBD {

/**
 * @brief One matched (density, potential) grid pair between two rigid body
 *        types - or a type and one of its own PMF grids (is_pmf).
 * @details Type-level, not RB-instance-level: which pairs of RB instances
 *          actually interact is decided by Phase 4's on-device cull kernel,
 *          which looks up matches here per (type_id_a, type_id_b) rather than
 *          re-deriving the grid-key match every step.
 */
struct RigidBodyGridPair {
	int type_i = -1;	  ///< owns the density grid
	int type_j = -1;	  ///< owns the potential grid (== type_i when is_pmf)
	int grid_id_rho = -1; ///< type_i's density grid
	int grid_id_u = -1;   ///< type_j's potential grid, or type_i's pmf grid when is_pmf
	bool is_pmf = false;  ///< true for a density/PMF match (not a true pairwise RB interaction)
	int update_period = 1; ///< legacy: rigidBodyGridGridPeriod - steps between force evaluations
};

/**
 * @brief Matches density/potential (and density/pmf) grid keys by name across
 *        every pair of rigid body types.
 *
 * Grid identity here is the grid_id GridManager already assigned when `types`
 * was populated (typically at config-parse time) - this class does no grid
 * loading itself. `grid_format` maps a grid_id to its storage format so the
 * per-pair format-uniformity rule (architecture decision #6) can be enforced
 * at construction, before Phase 4 ever sees the pair. Production callers pass
 * `[&](int id){ return grid_manager.get_grid_format(id); }`; tests can supply
 * a plain lookup with no grid loading at all.
 *
 * Segregates matches by format so Phase 4's cull kernel never re-inspects
 * format per item - today only Dense is populated (GridManager doesn't load
 * sparse grids yet); a sparse_pairs() list is deferred behind the same
 * GridFormat template seam as Phase 4.1.
 */
class RigidBodyForcePairList {
  public:
	/**
	 * @brief (Re)build the pair list from a fresh set of types
	 * @param types RigidBodyTypes with grid ids/keys already resolved
	 * @param grid_format grid_id -> GridFormat lookup
	 * @param update_period legacy: rigidBodyGridGridPeriod, applied to every emitted pair
	 * @throws Exception(ValueError) if a matched pair's two grids don't share a format
	 */
	void build(const std::vector<RigidBodyType>& types,
			   const std::function<GridFormat(int)>& grid_format,
			   int update_period = 1) {
		dense_pairs_.clear();

		// Type-type: density(t1) matched against potential(t2) by key name,
		// for every unordered pair with ti <= tj (legacy loops tj from ti, so
		// a type pair is only checked in this one direction - reproduced
		// faithfully here rather than "fixed" to be symmetric).
		for (size_t ti = 0; ti < types.size(); ++ti) {
			const RigidBodyType& t1 = types[ti];
			for (size_t tj = ti; tj < types.size(); ++tj) {
				const RigidBodyType& t2 = types[tj];
				match_and_emit(static_cast<int>(ti),
							   static_cast<int>(tj),
							   t1.density_grid_keys,
							   t1.density_grid_ids,
							   t2.potential_grid_keys,
							   t2.potential_grid_ids,
							   /*is_pmf=*/false,
							   update_period,
							   grid_format);
			}
		}

		// Type-PMF: density(t1) matched against that same type's own pmf
		// grids by key name - not a true pairwise RB interaction (legacy
		// comment survives the port).
		for (size_t ti = 0; ti < types.size(); ++ti) {
			const RigidBodyType& t1 = types[ti];
			match_and_emit(static_cast<int>(ti),
						   static_cast<int>(ti),
						   t1.density_grid_keys,
						   t1.density_grid_ids,
						   t1.pmf_keys,
						   t1.pmf_grid_ids,
						   /*is_pmf=*/true,
						   update_period,
						   grid_format);
		}
	}

	/// Today's only populated list - see class doc for why sparse is deferred.
	const std::vector<RigidBodyGridPair>& dense_pairs() const {
		return dense_pairs_;
	}

	size_t size() const {
		return dense_pairs_.size();
	}

  private:
	void match_and_emit(int type_i,
						 int type_j,
						 const std::vector<std::string>& keys1,
						 const std::vector<uint32_t>& ids1,
						 const std::vector<std::string>& keys2,
						 const std::vector<uint32_t>& ids2,
						 bool is_pmf,
						 int update_period,
						 const std::function<GridFormat(int)>& grid_format) {
		for (size_t k1 = 0; k1 < keys1.size(); ++k1) {
			for (size_t k2 = 0; k2 < keys2.size(); ++k2) {
				if (keys1[k1] != keys2[k2])
					continue;
				emit(type_i,
					 type_j,
					 static_cast<int>(ids1[k1]),
					 static_cast<int>(ids2[k2]),
					 is_pmf,
					 update_period,
					 grid_format);
			}
		}
	}

	void emit(int type_i,
			  int type_j,
			  int grid_id_rho,
			  int grid_id_u,
			  bool is_pmf,
			  int update_period,
			  const std::function<GridFormat(int)>& grid_format) {
		const GridFormat fmt_rho = grid_format(grid_id_rho);
		const GridFormat fmt_u = grid_format(grid_id_u);
		if (fmt_rho != fmt_u) {
			throw_value_error("RigidBodyForcePairList: format mismatch between grid_id %d "
							  "(density, type %d) and grid_id %d (potential/pmf, type %d)",
							  grid_id_rho,
							  type_i,
							  grid_id_u,
							  type_j);
		}

		if (fmt_rho == GridFormat::Sparse) {
			throw_not_implemented(
				"RigidBodyForcePairList: sparse grid-grid pairs (grid_id %d / %d)",
				grid_id_rho,
				grid_id_u);
		}

		RigidBodyGridPair pair;
		pair.type_i = type_i;
		pair.type_j = type_j;
		pair.grid_id_rho = grid_id_rho;
		pair.grid_id_u = grid_id_u;
		pair.is_pmf = is_pmf;
		pair.update_period = update_period;
		dense_pairs_.push_back(pair);
	}

	std::vector<RigidBodyGridPair> dense_pairs_;
};

} // namespace ARBD
