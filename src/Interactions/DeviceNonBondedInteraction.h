/**
 * @file DeviceNonBondedInteractions.h
 * @brief Device-side nonbonded interaction data structures
 * @author Pin-Yi Li <pinyili2@illinois.edu>
 * @date 2025-01-17
 */

#pragma once
#include "Backend/Buffer.h"
#include "Backend/Resource.h"
#include "Interactions/NonBondedInteraction.h"
#include "Interactions/TabulatedPotential.h"
#include "Objects/Tables.h"
#include "Types/Types.h"

namespace ARBD {

/**
 * @brief Device-side nonbonded interaction data (SoA format)
 *
 * Handles pairwise nonbonded interactions with O(1) type-pair lookup.
 * Uses bulk copy pattern for efficient device memory transfers.
 *
 * Design:
 * - Type-pair matrix for O(1) lookup: matrix[type_i * num_types + type_j] → table_index
 * - Separate form matrix to store interaction forms (Tabulated/Analytical)
 * - Bulk copy initialization (no individual device copies)
 * - Reuses TablesRegistry device buffers (no data duplication)
 */
class DeviceNonBondedInteractions {
  public:
	//========================================================================
	// CONSTRUCTION & INITIALIZATION
	//========================================================================

	/**
	 * @brief Constructor - allocates device buffers but does not initialize
	 * @param num_particle_types Number of particle types in system
	 * @param resource Computational resource (GPU/CPU)
	 *
	 * Call copy_pairwise_from_host() and link_pairwise_tables() after construction
	 */
	DeviceNonBondedInteractions(idx_t num_particle_types, const Resource& resource)
		: resource_(resource), num_particle_types_(num_particle_types),
		  pairwise_table_matrix_(num_particle_types * num_particle_types, resource),
		  pairwise_form_matrix_(num_particle_types * num_particle_types, resource) {
		// Initialize all pairs to -1 (no interaction) with ONE bulk copy
		std::vector<int> init_tables(num_particle_types * num_particle_types, -1);
		std::vector<int> init_forms(num_particle_types * num_particle_types,
									static_cast<int>(InteractionForm::Tabulated));

		pairwise_table_matrix_.copy_from_host(init_tables.data(), init_tables.size());
		pairwise_form_matrix_.copy_from_host(init_forms.data(), init_forms.size());

		LOGDEBUG("DeviceNonBondedInteractions: Allocated for {} particle types ({} pairs)",
				 num_particle_types,
				 num_particle_types * num_particle_types);
	}

	/**
	 * @brief Copy pairwise interactions from host (BULK COPY - call once)
	 *
	 * Builds type-pair matrix from PairNonBonded list with ONE device copy.
	 * Matrix layout: [type1 * num_types + type2] → (table_index, form)
	 *
	 * @param host_pairs Vector of PairNonBonded from TablesRegistry
	 *
	 * Example:
	 *   const auto& pairs = tables_registry.get_pair_nonbonded_types();
	 *   device_nb->copy_pairwise_from_host(pairs);
	 */
	void copy_pairwise_from_host(const std::vector<PairNonBonded>& host_pairs) {
		// Build entire matrix on host
		std::vector<int> table_matrix(num_particle_types_ * num_particle_types_, -1);
		std::vector<int> form_matrix(num_particle_types_ * num_particle_types_,
									 static_cast<int>(InteractionForm::Tabulated));

		for (const auto& pair : host_pairs) {
			// PairNonBonded already ensures type_id_1 <= type_id_2
			int type1 = pair.type_id_1;
			int type2 = pair.type_id_2;

			// Validate types
			if (type1 < 0 || type2 < 0 || type1 >= static_cast<int>(num_particle_types_) ||
				type2 >= static_cast<int>(num_particle_types_)) {
				LOGWARN("DeviceNonBondedInteractions: Invalid type pair ({}, {}), skipping",
						type1,
						type2);
				continue;
			}

			// Set both (i,j) and (j,i) for symmetric lookup in HOST memory
			idx_t idx1 = type1 * num_particle_types_ + type2;
			idx_t idx2 = type2 * num_particle_types_ + type1;

			table_matrix[idx1] = pair.function_index;
			table_matrix[idx2] = pair.function_index; // Symmetric

			form_matrix[idx1] = static_cast<int>(pair.form);
			form_matrix[idx2] = static_cast<int>(pair.form);
		}

		// ONE bulk copy to device for each matrix
		pairwise_table_matrix_.copy_from_host(table_matrix.data(), table_matrix.size());
		pairwise_form_matrix_.copy_from_host(form_matrix.data(), form_matrix.size());

		LOGINFO("DeviceNonBondedInteractions: Copied {} pairwise interactions to device",
				host_pairs.size());
	}

	/**
	 * @brief Link pairwise tabulated potential tables from TablesRegistry
	 *
	 * TablesRegistry already has device buffers for nonbonded tables.
	 * We create TabulatedPotential structs pointing to those buffers.
	 *
	 * @param tables_registry Global tables registry
	 * @param resource_idx Index of this resource in TablesRegistry's device arrays
	 *
	 * Example:
	 *   device_nb->link_pairwise_tables(tables_registry, patch_resource_idx);
	 */
	void link_pairwise_tables(const TablesRegistry& tables_registry, idx_t resource_idx) {
		const auto& nonbonded_tables = tables_registry.get_nonbonded();

		if (nonbonded_tables.empty()) {
			LOGDEBUG("DeviceNonBondedInteractions: No nonbonded tables to link");
			return;
		}

		// Build array of TabulatedPotential structs on HOST
		std::vector<TabulatedPotential> nb_pots;
		nb_pots.reserve(nonbonded_tables.size());

		for (size_t i = 0; i < nonbonded_tables.size(); ++i) {
			const auto& table = nonbonded_tables[i];
			const auto& device_buffer =
				tables_registry.get_device_nonbonded_buffer(resource_idx, i);

			// Validate buffer
			if (device_buffer.empty()) {
				LOGWARN("DeviceNonBondedInteractions: Empty device buffer for table {}, skipping",
						i);
				continue;
			}

			// Create metadata struct pointing to TablesRegistry's device buffer
			TabulatedPotential pot;
			pot.pot = const_cast<arbd_real*>(device_buffer.data()); // Device pointer to Y values
			pot.step_inv = 1.0f / static_cast<arbd_real>(table.step_size);
			pot.size = static_cast<unsigned int>(table.Y.size());
			pot.start = static_cast<arbd_real>(table.start);
			pot.is_periodic = false; // Pairwise potentials are not periodic

			nb_pots.push_back(pot);
		}

		if (!nb_pots.empty()) {
			// Allocate device buffer and copy metadata structs
			nonbonded_potentials_ = DeviceBuffer<TabulatedPotential>(nb_pots.size(), resource_);
			nonbonded_potentials_.copy_from_host(nb_pots.data(), nb_pots.size());

			LOGINFO("DeviceNonBondedInteractions: Linked {} nonbonded potential tables",
					nb_pots.size());
		}
	}

	//========================================================================
	// ACCESSORS FOR KERNELS
	//========================================================================

	/**
	 * @brief Get pairwise table index matrix (for kernel use)
	 * @return Device pointer to flattened matrix [type_i * num_types + type_j]
	 */
	DEVICE_PTR(const int) pairwise_table_matrix() const {
		return pairwise_table_matrix_.data();
	}

	/**
	 * @brief Get pairwise interaction form matrix (for kernel use)
	 * @return Device pointer to form matrix (InteractionForm enum values)
	 */
	DEVICE_PTR(const int) pairwise_form_matrix() const {
		return pairwise_form_matrix_.data();
	}

	/**
	 * @brief Get array of nonbonded potential structs (for kernel use)
	 * @return Device pointer to TabulatedPotential array
	 */
	DEVICE_PTR(const TabulatedPotential) nonbonded_potentials() const {
		return nonbonded_potentials_.data();
	}

	/**
	 * @brief Get number of particle types
	 */
	idx_t num_particle_types() const {
		return num_particle_types_;
	}

	/**
	 * @brief Get number of nonbonded potential tables
	 */
	idx_t num_potentials() const {
		return nonbonded_potentials_.size();
	}

	//========================================================================
	// DEVICE-SIDE LOOKUP HELPERS
	//========================================================================

	/**
	 * @brief Get pairwise table index for type pair (i, j)
	 * @return Index into nonbonded_potentials_, or -1 if no interaction
	 *
	 * Usage in kernel:
	 *   int table_idx = device_nb->get_pairwise_table_index(type_i, type_j);
	 *   if (table_idx >= 0) {
	 *       const TabulatedPotential& pot = device_nb->nonbonded_potentials()[table_idx];
	 *       // Compute force...
	 *   }
	 */
	DEVICE int get_pairwise_table_index(int type_i, int type_j) const {
		idx_t idx = type_i * num_particle_types_ + type_j;
		return pairwise_table_matrix_.data()[idx];
	}

	/**
	 * @brief Get pairwise interaction form for type pair
	 * @return InteractionForm enum value
	 */
	DEVICE int get_pairwise_form(int type_i, int type_j) const {
		idx_t idx = type_i * num_particle_types_ + type_j;
		return pairwise_form_matrix_.data()[idx];
	}

  private:
	Resource resource_;
	idx_t num_particle_types_;

	// === PAIRWISE INTERACTIONS ===
	// Flattened matrices: [type1 * num_types + type2] → value
	DeviceBuffer<int>
		pairwise_table_matrix_; // Index into nonbonded_potentials_ (-1 = no interaction)
	DeviceBuffer<int> pairwise_form_matrix_; // InteractionForm enum value

	// Array of potential structs (indexed by pairwise_table_matrix_)
	// Each struct contains pointer to TablesRegistry's device Y-value buffer
	DeviceBuffer<TabulatedPotential> nonbonded_potentials_;
};

} // namespace ARBD

// SYCL device copyable traits
#ifdef USE_SYCL
#include <sycl/sycl.hpp>
template<>
struct sycl::is_device_copyable<ARBD::DeviceNonBondedInteractions> : std::true_type {};
#endif
