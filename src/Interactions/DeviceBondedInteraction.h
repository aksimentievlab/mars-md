// File: src/Objects/DeviceBondedInteractions.h

#pragma once
#include "Backend/Buffer.h"
#include "Backend/Kernels.h"
#include "Backend/Resource.h"
#include "Interactions/Bonded/BondComputer.h"
#include "Interactions/BondedInteraction.h"
#include "Interactions/TabulatedPotential.h"
#include "Objects/Tables.h"
#include "System/PeriodicBox.h"
#include "Types/Types.h"
#include "Types/Vector3.h"
#include <algorithm>
#include <utility>

namespace MARS {

/**
 * @brief Device-side bonded interaction data (SoA format)
 *
 * Stores bonded topology and potential tables on GPU.
 * Each patch gets its own DeviceBondedInteractions.
 */
class DeviceBondedInteractions {
  public:
	//========================================================================
	// CONSTRUCTION & INITIALIZATION
	//========================================================================

	DeviceBondedInteractions(const Resource& resource) : resource_(resource) {}

	/**
	 * @brief Copy bonded topology from host
	 *
	 * Extracts particle indices and function indices from host structures
	 */
	void copy_from_host(const BondedInteractions& host_bonded) {
		// === BONDS ===
		const auto& bonds = host_bonded.get_bonds();
		num_bonds_ = bonds.size();

		if (num_bonds_ > 0) {
			std::vector<int2> bond_idx(num_bonds_);
			std::vector<int> bond_fn_idx(num_bonds_);
			std::vector<int> bond_form(num_bonds_);

			for (size_t i = 0; i < num_bonds_; ++i) {
				bond_idx[i] = int2{bonds[i].ind1, bonds[i].ind2};
				bond_fn_idx[i] = bonds[i].function_index;
				bond_form[i] = static_cast<int>(bonds[i].form);
			}

			bond_indices_ = DeviceBuffer<int2>(num_bonds_, resource_);
			bond_indices_.copy_from_host(bond_idx.data(), num_bonds_);
			bond_table_indices_ = DeviceBuffer<int>(num_bonds_, resource_);
			bond_table_indices_.copy_from_host(bond_fn_idx.data(), num_bonds_);
			bond_forms_ = DeviceBuffer<int>(num_bonds_, resource_);
			bond_forms_.copy_from_host(bond_form.data(), num_bonds_);
		}

		// === ANGLES ===
		const auto& angles = host_bonded.get_angles();
		num_angles_ = angles.size();

		if (num_angles_ > 0) {
			std::vector<int3> angle_idx(num_angles_);
			std::vector<int> angle_fn_idx(num_angles_);
			std::vector<int> angle_form(num_angles_);

			for (size_t i = 0; i < num_angles_; ++i) {
				angle_idx[i] = int3{angles[i].ind1, angles[i].ind2, angles[i].ind3};
				angle_fn_idx[i] = angles[i].function_index;
				angle_form[i] = static_cast<int>(angles[i].form);
			}

			angle_indices_ = DeviceBuffer<int3>(num_angles_, resource_);
			angle_indices_.copy_from_host(angle_idx.data(), num_angles_);
			angle_table_indices_ = DeviceBuffer<int>(num_angles_, resource_);
			angle_table_indices_.copy_from_host(angle_fn_idx.data(), num_angles_);
			angle_forms_ = DeviceBuffer<int>(num_angles_, resource_);
			angle_forms_.copy_from_host(angle_form.data(), num_angles_);
		}

		// === DIHEDRALS ===
		const auto& dihedrals = host_bonded.get_dihedrals();
		num_dihedrals_ = dihedrals.size();

		if (num_dihedrals_ > 0) {
			std::vector<int4> dihedral_idx(num_dihedrals_);
			std::vector<int> dihedral_fn_idx(num_dihedrals_);
			std::vector<int> dihedral_form(num_dihedrals_);

			for (size_t i = 0; i < num_dihedrals_; ++i) {
				dihedral_idx[i] = int4{dihedrals[i].ind1,
									   dihedrals[i].ind2,
									   dihedrals[i].ind3,
									   dihedrals[i].ind4};
				dihedral_fn_idx[i] = dihedrals[i].function_index;
				dihedral_form[i] = static_cast<int>(dihedrals[i].form);
			}

			dihedral_indices_ = DeviceBuffer<int4>(num_dihedrals_, resource_);
			dihedral_indices_.copy_from_host(dihedral_idx.data(), num_dihedrals_);
			dihedral_table_indices_ = DeviceBuffer<int>(num_dihedrals_, resource_);
			dihedral_table_indices_.copy_from_host(dihedral_fn_idx.data(), num_dihedrals_);
			dihedral_forms_ = DeviceBuffer<int>(num_dihedrals_, resource_);
			dihedral_forms_.copy_from_host(dihedral_form.data(), num_dihedrals_);
		}

		// === EXCLUSIONS ===
		const auto& exclusions = host_bonded.get_exclusions();
		num_exclusions_ = exclusions.size();

		if (num_exclusions_ > 0) {
			// Keep the flat (canonicalized, sorted) pair list for any consumer that
			// still wants it, but the nonbonded kernel uses the per-particle CSR
			// lists built below.
			std::vector<int2> excl_pairs(num_exclusions_);
			int max_idx = 0;
			for (size_t i = 0; i < num_exclusions_; ++i) {
				int a = exclusions[i].ind1;
				int b = exclusions[i].ind2;
				if (a > b) {
					std::swap(a, b);
				}
				excl_pairs[i] = int2{a, b};
				max_idx = std::max(max_idx, b);
			}
			std::sort(excl_pairs.begin(), excl_pairs.end(), [](const int2& p, const int2& q) {
				return p.x != q.x ? p.x < q.x : p.y < q.y;
			});
			exclusion_pairs_ = DeviceBuffer<int2>(num_exclusions_, resource_);
			exclusion_pairs_.copy_from_host(excl_pairs.data(), num_exclusions_);
#ifdef ENABLE_ZORDER_REORDER
			excl_pairs_host_ = excl_pairs; // retained for reorder CSR rebuild
#endif

			// Build per-particle CSR exclusion lists. Each exclusion (a,b) is
			// recorded in both a's and b's segment, so the nonbonded kernel can
			// test a candidate pair (i,j) by scanning only i's excluded partners
			// (~degree(i), typically 1-2 for a linear polymer). This replaces the
			// O(log num_exclusions) binary search (and the original O(num_exclusions)
			// scan) in the per-pair hot loop.
			num_excl_particles_ = static_cast<idx_t>(max_idx) + 1;
			std::vector<int> offsets(num_excl_particles_ + 1, 0);
			for (size_t i = 0; i < num_exclusions_; ++i) {
				offsets[excl_pairs[i].x + 1]++;
				offsets[excl_pairs[i].y + 1]++;
			}
			for (idx_t p = 0; p < num_excl_particles_; ++p) {
				offsets[p + 1] += offsets[p];
			}
			const size_t total = static_cast<size_t>(offsets[num_excl_particles_]);
			std::vector<int> neighbors(total);
			std::vector<int> cursor(offsets.begin(), offsets.end() - 1);
			for (size_t i = 0; i < num_exclusions_; ++i) {
				const int a = excl_pairs[i].x;
				const int b = excl_pairs[i].y;
				neighbors[cursor[a]++] = b;
				neighbors[cursor[b]++] = a;
			}
			excl_offsets_ = DeviceBuffer<int>(num_excl_particles_ + 1, resource_);
			excl_offsets_.copy_from_host(offsets.data(), num_excl_particles_ + 1);
			excl_neighbors_ = DeviceBuffer<int>(total, resource_);
			excl_neighbors_.copy_from_host(neighbors.data(), total);
		}

		// === RESTRAINTS ===
		const auto& restraints = host_bonded.get_restraints();
		num_restraints_ = restraints.size();

		if (num_restraints_ > 0) {
			std::vector<int> rest_ids(num_restraints_);
			std::vector<Vector3> rest_pos(num_restraints_);
			std::vector<float> rest_k(num_restraints_);

			for (size_t i = 0; i < num_restraints_; ++i) {
				rest_ids[i] = restraints[i].ind;
				rest_pos[i] = restraints[i].r0;
				rest_k[i] = restraints[i].k;
			}

			restraint_particle_ids_ = DeviceBuffer<int>(num_restraints_, resource_);
			restraint_particle_ids_.copy_from_host(rest_ids.data(), num_restraints_);
			restraint_positions_ = DeviceBuffer<Vector3>(num_restraints_, resource_);
			restraint_positions_.copy_from_host(rest_pos.data(), num_restraints_);
			restraint_spring_constants_ = DeviceBuffer<mars_real>(num_restraints_, resource_);
			restraint_spring_constants_.copy_from_host(rest_k.data(), num_restraints_);
		}
	}
	/**
	 * @brief Link device potential tables from TablesRegistry
	 *
	 * TablesRegistry already has device buffers. We just need pointers to them.
	 */
	void link_tables(const TablesRegistry& tables_registry, idx_t resource_idx) {
		// TablesRegistry has device_bond_y_buffers_[resource_idx][table_idx]
		// We need to create TabulatedPotential structs pointing to these

		const auto& bond_tables = tables_registry.get_bond_functions();
		const auto& angle_tables = tables_registry.get_angle_functions();
		const auto& dihedral_tables = tables_registry.get_dihedral_functions();

		// Build TabulatedPotential structs on HOST
		std::vector<TabulatedPotential> bond_pots;
		for (size_t i = 0; i < bond_tables.size(); ++i) {
			const auto& table = bond_tables[i];
			auto& device_buffer = tables_registry.get_device_bond_buffer(resource_idx, i);

			TabulatedPotential pot;
			pot.pot = const_cast<mars_real*>(device_buffer.data()); // Device pointer
			pot.step_inv = table.step_size > 0.0 ? static_cast<mars_real>(1.0 / table.step_size)
												 : mars_real(0);
			pot.size = static_cast<unsigned int>(table.Y.size());
			pot.start = table.start;
			pot.is_periodic = false; // Bonds are not periodic

			bond_pots.push_back(pot);
		}

		// Similar for angles and dihedrals...
		std::vector<TabulatedPotential> angle_pots;
		for (size_t i = 0; i < angle_tables.size(); ++i) {
			const auto& table = angle_tables[i];
			const auto& device_buffer = tables_registry.get_device_angle_buffer(resource_idx, i);

			TabulatedPotential pot;
			pot.pot = const_cast<mars_real*>(device_buffer.data());
			pot.step_inv = table.step_size > 0.0 ? static_cast<mars_real>(1.0 / table.step_size)
												 : mars_real(0);
			pot.size = static_cast<unsigned int>(table.Y.size());
			pot.start = table.start;
			pot.is_periodic = true;

			angle_pots.push_back(pot);
		}

		std::vector<TabulatedPotential> dihedral_pots;
		for (size_t i = 0; i < dihedral_tables.size(); ++i) {
			const auto& table = dihedral_tables[i];
			const auto& device_buffer = tables_registry.get_device_dihedral_buffer(resource_idx, i);

			TabulatedPotential pot;
			pot.pot = const_cast<mars_real*>(device_buffer.data());
			pot.step_inv = table.step_size > 0.0 ? static_cast<mars_real>(1.0 / table.step_size)
												 : mars_real(0);
			pot.size = static_cast<unsigned int>(table.Y.size());
			pot.start = table.start;
			pot.is_periodic = true; // Dihedrals ARE periodic

			dihedral_pots.push_back(pot);
		}

		// Allocate device buffers and copy
		if (!bond_pots.empty()) {
			bond_potentials_ = DeviceBuffer<TabulatedPotential>(bond_pots.size(), resource_);
			bond_potentials_.copy_from_host(bond_pots.data(), bond_pots.size());
		}

		if (!angle_pots.empty()) {
			angle_potentials_ = DeviceBuffer<TabulatedPotential>(angle_pots.size(), resource_);
			angle_potentials_.copy_from_host(angle_pots.data(), angle_pots.size());
		}

		if (!dihedral_pots.empty()) {
			dihedral_potentials_ =
				DeviceBuffer<TabulatedPotential>(dihedral_pots.size(), resource_);
			dihedral_potentials_.copy_from_host(dihedral_pots.data(), dihedral_pots.size());
		}
	}

	//========================================================================
	// ACCESSORS FOR KERNELS
	//========================================================================

	// Bonds
	DEVICE_PTR(const int2) bond_indices() const {
		return bond_indices_.data();
	}
	DEVICE_PTR(const int) bond_table_indices() const {
		return bond_table_indices_.data();
	}
	DEVICE_PTR(const int) bond_forms() const {
		return bond_forms_.data();
	}
	DEVICE_PTR(const TabulatedPotential) bond_potentials() const {
		return bond_potentials_.data();
	}
	idx_t num_bonds() const {
		return num_bonds_;
	}

	// Angles
	DEVICE_PTR(const int3) angle_indices() const {
		return angle_indices_.data();
	}
	DEVICE_PTR(const int) angle_table_indices() const {
		return angle_table_indices_.data();
	}
	DEVICE_PTR(const int) angle_forms() const {
		return angle_forms_.data();
	}
	DEVICE_PTR(const TabulatedPotential) angle_potentials() const {
		return angle_potentials_.data();
	}
	idx_t num_angles() const {
		return num_angles_;
	}

	// Dihedrals
	DEVICE_PTR(const int4) dihedral_indices() const {
		return dihedral_indices_.data();
	}
	DEVICE_PTR(const int) dihedral_table_indices() const {
		return dihedral_table_indices_.data();
	}
	DEVICE_PTR(const int) dihedral_forms() const {
		return dihedral_forms_.data();
	}
	DEVICE_PTR(const TabulatedPotential) dihedral_potentials() const {
		return dihedral_potentials_.data();
	}
	idx_t num_dihedrals() const {
		return num_dihedrals_;
	}

	// Exclusions
	DEVICE_PTR(const int2) exclusion_pairs() const {
		return exclusion_pairs_.data();
	}
	idx_t num_exclusions() const {
		return num_exclusions_;
	}
	// Per-particle (CSR) exclusion view; see member declarations below.
	DEVICE_PTR(const int) exclusion_offsets() const {
		return excl_offsets_.data();
	}
	DEVICE_PTR(const int) exclusion_neighbors() const {
		return excl_neighbors_.data();
	}
	idx_t num_excl_particles() const {
		return num_excl_particles_;
	}

	// Restraints
	DEVICE_PTR(const int) restraint_particle_ids() const {
		return restraint_particle_ids_.data();
	}
	DEVICE_PTR(const Vector3) restraint_positions() const {
		return restraint_positions_.data();
	}
	DEVICE_PTR(const mars_real) restraint_spring_constants() const {
		return restraint_spring_constants_.data();
	}
	idx_t num_restraints() const {
		return num_restraints_;
	}

#ifdef ENABLE_ZORDER_REORDER
	/// Remap all particle-slot references after a Morton reorder. Bonds/angles/
	/// dihedrals/restraints in place on device; exclusion CSR rebuilt. See dev_notes.
	template<class Sorter>
	void remap_particle_indices(Sorter& sorter) {
		if (num_bonds_ > 0)
			sorter.remap_indices(reinterpret_cast<int*>(bond_indices_.data()),
								 static_cast<size_t>(num_bonds_) * 2);
		if (num_angles_ > 0)
			sorter.remap_indices(reinterpret_cast<int*>(angle_indices_.data()),
								 static_cast<size_t>(num_angles_) * 3);
		if (num_dihedrals_ > 0)
			sorter.remap_indices(reinterpret_cast<int*>(dihedral_indices_.data()),
								 static_cast<size_t>(num_dihedrals_) * 4);
		if (num_restraints_ > 0)
			sorter.remap_indices(restraint_particle_ids_.data(),
								 static_cast<size_t>(num_restraints_));
		if (num_exclusions_ > 0)
			rebuild_exclusions_after_reorder(sorter);
	}

	/// Remap exclusion edge endpoints through the inverse map, then rebuild the
	/// per-particle CSR from the remapped edges (host; reorder cadence only).
	template<class Sorter>
	void rebuild_exclusions_after_reorder(Sorter& sorter) {
		const size_t nparts = sorter.get_num_particles();
		std::vector<uint32_t> inv(nparts);
		sorter.get_inverse_indices().copy_to_host(inv.data(), nparts);

		for (auto& e : excl_pairs_host_) {
			int a = static_cast<int>(inv[e.x]);
			int b = static_cast<int>(inv[e.y]);
			if (a > b) {
				std::swap(a, b);
			}
			e = int2{a, b};
		}
		std::sort(excl_pairs_host_.begin(), excl_pairs_host_.end(),
				  [](const int2& p, const int2& q) { return p.x != q.x ? p.x < q.x : p.y < q.y; });

		int max_idx = 0;
		for (const auto& e : excl_pairs_host_) {
			max_idx = std::max(max_idx, e.y);
		}
		num_excl_particles_ = static_cast<idx_t>(max_idx) + 1;
		std::vector<int> offsets(num_excl_particles_ + 1, 0);
		for (const auto& e : excl_pairs_host_) {
			offsets[e.x + 1]++;
			offsets[e.y + 1]++;
		}
		for (idx_t p = 0; p < num_excl_particles_; ++p) {
			offsets[p + 1] += offsets[p];
		}
		const size_t total = static_cast<size_t>(offsets[num_excl_particles_]);
		std::vector<int> neighbors(total);
		std::vector<int> cursor(offsets.begin(), offsets.end() - 1);
		for (const auto& e : excl_pairs_host_) {
			neighbors[cursor[e.x]++] = e.y;
			neighbors[cursor[e.y]++] = e.x;
		}

		exclusion_pairs_.copy_from_host(excl_pairs_host_.data(), num_exclusions_);
		// Grow-only reuse: reorder runs every few steps, so never realloc these
		// per reorder (excl_neighbors_ is a constant 2*num_exclusions_; excl_offsets_
		// only grows toward num_particles+1). Reallocating each time churns device
		// memory over a long run. The kernel bounds its reads by num_excl_particles_,
		// so trailing stale entries in an oversized buffer are never touched.
		if (excl_offsets_.size() < static_cast<size_t>(num_excl_particles_ + 1)) {
			excl_offsets_ = DeviceBuffer<int>(num_excl_particles_ + 1, resource_);
		}
		excl_offsets_.copy_from_host(offsets.data(), num_excl_particles_ + 1);
		if (excl_neighbors_.size() < total) {
			excl_neighbors_ = DeviceBuffer<int>(total, resource_);
		}
		excl_neighbors_.copy_from_host(neighbors.data(), total);
	}
#endif

  private:
	Resource resource_;

	// === BONDS ===
	DeviceBuffer<int2> bond_indices_;	   // Particle pairs (i, j)
	DeviceBuffer<int> bond_table_indices_; // Which table to use (index into bond_potentials_)
	DeviceBuffer<int> bond_forms_;		   // InteractionForm: Grid=0, Tabulated=1, Analytical=2
	DeviceBuffer<TabulatedPotential> bond_potentials_; // Array of potential structs
	idx_t num_bonds_{0};

	// === ANGLES ===
	DeviceBuffer<int3> angle_indices_; // Particle triplets (i, j, k)
	DeviceBuffer<int> angle_table_indices_;
	DeviceBuffer<int> angle_forms_;
	DeviceBuffer<TabulatedPotential> angle_potentials_;
	idx_t num_angles_{0};

	// === DIHEDRALS ===
	DeviceBuffer<int4> dihedral_indices_; // Particle quartets (i, j, k, l)
	DeviceBuffer<int> dihedral_table_indices_;
	DeviceBuffer<int> dihedral_forms_;
	DeviceBuffer<TabulatedPotential> dihedral_potentials_;
	idx_t num_dihedrals_{0};

	// === EXCLUSIONS ===
	DeviceBuffer<int2> exclusion_pairs_; // Pairs to exclude from nonbonded
	idx_t num_exclusions_{0};
	// Per-particle (CSR) view of the same exclusions: excl_neighbors_ holds, for
	// each particle p, its excluded partners in [excl_offsets_[p], excl_offsets_[p+1]).
	// Each exclusion (a,b) is stored in both a's and b's segment. The nonbonded
	// kernel uses this to test a candidate pair in O(degree(p)) (~2 for a linear
	// chain) instead of scanning/searching all exclusions.
	DeviceBuffer<int> excl_offsets_;   // size num_excl_particles_ + 1
	DeviceBuffer<int> excl_neighbors_; // size 2 * num_exclusions_
	idx_t num_excl_particles_{0};	   // number of particles covered by excl_offsets_
#ifdef ENABLE_ZORDER_REORDER
	// Canonical exclusion edges in current slot order, retained so a reorder can
	// remap endpoints and rebuild the CSR without re-reading host topology.
	std::vector<int2> excl_pairs_host_;
#endif

	// === RESTRAINTS ===
	DeviceBuffer<int> restraint_particle_ids_;
	DeviceBuffer<Vector3> restraint_positions_;
	DeviceBuffer<mars_real> restraint_spring_constants_;
	idx_t num_restraints_{0};
};

// ============================================================================
// LAUNCH HELPER FUNCTIONS (similar to launch_BD, launch_BAOAB)
// ============================================================================

/**
 * @brief Launch tabulated bond force computation
 */
inline Event launch_tabulated_bonds(const Resource& resource,
									DEVICE_PTR(const int2) particle_indices,
									DEVICE_PTR(Vector3) positions,
									DEVICE_PTR(Vector3) force_energy,
									DEVICE_PTR(const TabulatedPotential) tables,
									DEVICE_PTR(const int) table_indices,
									DEVICE_PTR(const int) forms,
									const PeriodicBox* pbox,
									bool get_energy,
									idx_t num_bonds) {
	KernelConfig config = KernelConfig::for_1d(num_bonds, resource);

	TabulatedBondComputer computer(particle_indices,
								   positions,
								   force_energy,
								   tables,
								   table_indices,
								   forms,
								   pbox,
								   get_energy,
								   num_bonds);

	return launch_kernel(resource, config, computer);
}

/**
 * @brief Launch tabulated angle force computation
 */
inline Event launch_tabulated_angles(const Resource& resource,
									 DEVICE_PTR(const int3) particle_indices,
									 DEVICE_PTR(Vector3) positions,
									 DEVICE_PTR(Vector3) force_energy,
									 DEVICE_PTR(const TabulatedPotential) tables,
									 DEVICE_PTR(const int) table_indices,
									 DEVICE_PTR(const int) forms,
									 const PeriodicBox* pbox,
									 bool get_energy,
									 idx_t num_angles) {
	KernelConfig config = KernelConfig::for_1d(num_angles, resource);

	TabulatedAngleComputer computer(particle_indices,
									positions,
									force_energy,
									tables,
									table_indices,
									forms,
									pbox,
									get_energy,
									num_angles);

	return launch_kernel(resource, config, computer);
}

/**
 * @brief Launch tabulated dihedral force computation
 */
inline Event launch_tabulated_dihedrals(const Resource& resource,
										DEVICE_PTR(const int4) particle_indices,
										DEVICE_PTR(Vector3) positions,
										DEVICE_PTR(Vector3) force_energy,
										DEVICE_PTR(const TabulatedPotential) tables,
										DEVICE_PTR(const int) table_indices,
										DEVICE_PTR(const int) forms,
										const PeriodicBox* pbox,
										bool get_energy,
										idx_t num_dihedrals) {
	KernelConfig config = KernelConfig::for_1d(num_dihedrals, resource);

	TabulatedDihedralComputer computer(particle_indices,
									   positions,
									   force_energy,
									   tables,
									   table_indices,
									   forms,
									   pbox,
									   get_energy,
									   num_dihedrals);

	return launch_kernel(resource, config, computer);
}

/**
 * @brief Launch analytical bond force computation (templated by bond type)
 */
template<int BondTypeId>
inline Event launch_analytical_bonds(const Resource& resource,
									 DEVICE_PTR(const int2) particle_indices,
									 DEVICE_PTR(Vector3) positions,
									 DEVICE_PTR(Vector3) force_energy,
									 DEVICE_PTR(const mars_real) params,
									 const PeriodicBox* pbox,
									 bool get_energy,
									 idx_t num_bonds) {
	KernelConfig config = KernelConfig::for_1d(num_bonds, resource);

	AnalyticalBondComputer<BondTypeId>
		computer(particle_indices, positions, force_energy, params, pbox, get_energy, num_bonds);

	return launch_kernel(resource, config, computer);
}

/**
 * @brief Launch harmonic restraint force computation
 */
inline Event launch_harmonic_restraints(const Resource& resource,
										DEVICE_PTR(const int) particle_ids,
										DEVICE_PTR(Vector3) positions,
										DEVICE_PTR(Vector3) force_energy,
										DEVICE_PTR(const Vector3) anchors,
										DEVICE_PTR(const mars_real) spring_constants,
										const PeriodicBox* pbox,
										bool get_energy,
										idx_t num_restraints) {
	KernelConfig config = KernelConfig::for_1d(num_restraints, resource);

	HarmonicRestraintComputer computer(particle_ids,
									   positions,
									   force_energy,
									   anchors,
									   spring_constants,
									   pbox,
									   get_energy,
									   num_restraints);

	return launch_kernel(resource, config, computer);
}

} // namespace MARS
