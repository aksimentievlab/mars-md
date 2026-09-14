#pragma once
/**
 * @file NonBondedInteraction.h
 * @brief Host-side nonbonded pair list: which type pairs interact, and how
 */

#include "Backend/Kernels.h"
#include "BondedInteraction.h"
#include "Header.h"
#include "IO/Reader.h"
#include "Interactions/Nonbonded/GridGridKernels.h"
#include "Interactions/Nonbonded/Pairwise.h"
#include "Objects/DeviceParticle.h"
#include "Objects/ParticleProperties.h"
#include "SimParam.h"
#include "Types/BaseGrid.h"

namespace MARS {

/// @brief The analytical pair term a name denotes, or PAIR_TERM_NONE
inline uint32_t pair_term_from_name(const std::string& function_name) {
	if (function_name == "coulomb" || function_name == "columb")
		return PAIR_TERM_COULOMB;
	if (function_name == "debye_huckel")
		return PAIR_TERM_DEBYE_HUCKEL;
	if (function_name == "onck")
		return PAIR_TERM_ONCK;
	if (function_name == "gaussian")
		return PAIR_TERM_GAUSSIAN;
	if (function_name == "softcore")
		return PAIR_TERM_SOFTCORE;
	return PAIR_TERM_NONE;
}

/**
 * @brief One type pair and its interaction: analytical by name, or tabulated by
 *        index into TablesRegistry::get_nonbonded()
 */
struct PairNonBonded {
	int type_id_1{-1};
	int type_id_2{-1};
	std::string type_name_1{};
	std::string type_name_2{};
	std::string function_name{""};
	InteractionForm form{InteractionForm::Tabulated};
	int function_index{-1};

	/// Analytical pair by type id
	PairNonBonded(int type_id_1, int type_id_2, const std::string& function_name)
		: PairNonBonded(type_id_1, type_id_2, function_name, -1) {}

	/// Tabulated pair by type id; @p table_index from TablesRegistry
	PairNonBonded(int type_id_1, int type_id_2, const std::string& function_name, int table_index) {
		this->type_id_1 = std::min(type_id_1, type_id_2);
		this->type_id_2 = std::max(type_id_1, type_id_2);
		set_function(function_name, table_index);
	}

	/// Analytical pair by type name
	PairNonBonded(const std::string& type_name_1,
				  const std::string& type_name_2,
				  const std::string& function_name)
		: PairNonBonded(type_name_1, type_name_2, function_name, -1) {}

	/// Tabulated pair by type name; @p table_index from TablesRegistry
	PairNonBonded(const std::string& type_name_1,
				  const std::string& type_name_2,
				  const std::string& function_name,
				  int table_index)
		: type_name_1(type_name_1), type_name_2(type_name_2) {
		set_function(function_name, table_index);
	}

	bool is_analytical() const {
		return form == InteractionForm::Analytical;
	}

	/// The term bits this pair contributes to the enabled mask
	uint32_t term_bits() const {
		return is_analytical() ? pair_term_from_name(function_name) : PAIR_TERM_TABULATED;
	}

	/**
	 * @brief Resolve type_name_1/type_name_2 into type_id_1/type_id_2.
	 * @param name_to_id Callable mapping a type name to its assigned id.
	 * @note No-op when the names are empty (ids were supplied directly).
	 *       Templated on the lookup so this header stays independent of SimSystem.
	 */
	template<typename NameToId>
	void resolve_type_names(NameToId&& name_to_id) {
		if (type_name_1.empty() || type_name_2.empty()) {
			return;
		}
		const int a = name_to_id(type_name_1);
		const int b = name_to_id(type_name_2);
		type_id_1 = std::min(a, b);
		type_id_2 = std::max(a, b);
	}

  private:
	void set_function(const std::string& function_name, int table_index) {
		this->function_name = function_name;
		if (table_index >= 0) {
			form = InteractionForm::Tabulated;
			function_index = table_index;
			return;
		}
		if (pair_term_from_name(function_name) == PAIR_TERM_NONE) {
			throw_value_error("PairNonBonded: '%s' is not an analytical pair term and no table "
							  "index was given",
							  function_name.c_str());
		}
		form = InteractionForm::Analytical;
		function_index = -1;
	}
};

/**
 * @brief Solvent constants of the screened electrostatic pair terms
 * @note One solvent per system, so these are global; defaults equal the
 *       functor defaults in Columb.h
 */
struct SolventParams {
	mars_real debye_length{10}; ///< Debye-Huckel screening length, Angstrom
	mars_real dielectric{80};	///< Debye-Huckel relative dielectric
	mars_real onck_kappa{0.1};	///< Onck inverse screening length, 1/Angstrom
	mars_real onck_sz{80};		///< Onck bulk dielectric plateau
	mars_real onck_z{6.86};		///< Onck sigmoid width, Angstrom
};

/// @brief The system's nonbonded pair list, owned by SimSystem
class NonBondedInteractions {
  public:
	NonBondedInteractions() = default;
	explicit NonBondedInteractions(std::vector<PairNonBonded> pair_nonbonded)
		: pair_nonbonded_(std::move(pair_nonbonded)) {}

	void set_solvent_params(const SolventParams& params) {
		solvent_ = params;
	}
	const SolventParams& get_solvent_params() const {
		return solvent_;
	}

	/// @brief Append a pair; a pair already declared (by id or by name) is kept as is
	void add_pair_nonbonded(const PairNonBonded& pair) {
		for (const auto& existing : pair_nonbonded_) {
			if (same_pair(existing, pair)) {
				LOGWARN("NonBondedInteractions: pair ({}, {}) already declared as '{}'; "
						"ignoring '{}'",
						existing.type_name_1.empty() ? std::to_string(existing.type_id_1)
													 : existing.type_name_1,
						existing.type_name_2.empty() ? std::to_string(existing.type_id_2)
													 : existing.type_name_2,
						existing.function_name,
						pair.function_name);
				return;
			}
		}
		pair_nonbonded_.push_back(pair);
	}

	const std::vector<PairNonBonded>& get_pair_nonbonded() const {
		return pair_nonbonded_;
	}
	size_t get_num_pair_nonbonded() const {
		return pair_nonbonded_.size();
	}

	/// @brief Union of every declared pair's term bits; the kernel's enabled mask
	uint32_t enabled_terms() const {
		uint32_t terms = PAIR_TERM_NONE;
		for (const auto& pair : pair_nonbonded_) {
			terms |= pair.term_bits();
		}
		return terms;
	}

	/// @brief Resolve every name-declared pair into type ids
	template<typename NameToId>
	void resolve_type_names(NameToId&& name_to_id) {
		for (auto& pair : pair_nonbonded_) {
			pair.resolve_type_names(name_to_id);
		}
	}

  private:
	static bool same_pair(const PairNonBonded& a, const PairNonBonded& b) {
		const bool by_name = !a.type_name_1.empty() && !b.type_name_1.empty();
		if (by_name) {
			return (a.type_name_1 == b.type_name_1 && a.type_name_2 == b.type_name_2) ||
				   (a.type_name_1 == b.type_name_2 && a.type_name_2 == b.type_name_1);
		}
		return a.type_id_1 >= 0 && a.type_id_1 == b.type_id_1 && a.type_id_2 == b.type_id_2;
	}

	std::vector<PairNonBonded> pair_nonbonded_{};
	SolventParams solvent_{};
};

/**
 * @brief Resolve per-pair table tags; run once per pairlist rebuild.
 * @note Exclusions are applied by the pairlist builder, not here.
 */
inline Event launch_resolve_pair_tables(const Resource& resource,
										DEVICE_PTR(const int2) particle_indices,
										DEVICE_PTR(const int) type_ids,
										DEVICE_PTR(const uint32_t) pairwise_term_matrix,
										idx_t num_particle_types,
										DEVICE_PTR(uint32_t) pair_tag,
										idx_t num_pairs) {
	if (num_pairs == 0)
		return Event(nullptr, resource);
	KernelConfig config = KernelConfig::for_1d(num_pairs, resource);
	ResolvePairTableKernel resolver{particle_indices,
									type_ids,
									pairwise_term_matrix,
									num_particle_types,
									pair_tag,
									num_pairs};
	return launch_kernel(resource, config, resolver);
}

/**
 * @brief Launch pairwise tabulated nonbonded force computation.
 * @note `pair_tag` must be filled by launch_resolve_pair_tables after each rebuild.
 */
inline Event launch_tabulated_nonbonded(const Resource& resource,
										DEVICE_PTR(const int2) particle_indices,
										DEVICE_PTR(Vector3) positions,
										DEVICE_PTR(Vector3) force_energy,
										DEVICE_PTR(const uint32_t) pair_tag,
										DEVICE_PTR(const TabulatedPotential) tables,
										const PeriodicBox* pbox,
										bool get_energy,
										idx_t num_pairs,
										float cutoff_squared) {
	if (num_pairs == 0)
		return Event(nullptr, resource);

	KernelConfig config = KernelConfig::for_1d(num_pairs, resource);

	TabulatedNonBondedComputer computer(particle_indices,
										positions,
										force_energy,
										pair_tag,
										tables,
										pbox,
										get_energy,
										num_pairs,
										cutoff_squared);

	return launch_kernel(resource, config, computer);
}
/**
 * @brief Launch every enabled nonbonded pair term in one pass over the pairlist.
 * @todo fix the PeriodicBox to DEVICE_PTR(const PeriodicBox) in the kernel;  `enabled_terms` gates
 * the run; the per-pair tag narrows it further, so a pair contributes a term only when both agree.
 * @param solvent Screening constants set on the electrostatic functors before launch
 * @note `pair_tag` must be filled by launch_resolve_pair_tables after each rebuild.
 */
inline Event launch_pairwise_nonbonded(const Resource& resource,
									   DEVICE_PTR(const int2) particle_indices,
									   ParticleView particles,
									   DEVICE_PTR(const uint32_t) pair_tag,
									   DEVICE_PTR(const TabulatedPotential) tables,
									   ParticleTypeView types,
									   const PeriodicBox* pbox,
									   bool get_energy,
									   idx_t num_pairs,
									   float cutoff_squared,
									   uint32_t enabled_terms,
									   const SolventParams& solvent) {
	if (num_pairs == 0 || enabled_terms == PAIR_TERM_NONE)
		return Event(nullptr, resource);

	KernelConfig config = KernelConfig::for_1d(num_pairs, resource);

	PairNonbondedComputer computer(particle_indices,
								   particles,
								   pair_tag,
								   tables,
								   types,
								   pbox,
								   get_energy,
								   num_pairs,
								   cutoff_squared,
								   enabled_terms);
	computer.debye_huckel.screen_length = solvent.debye_length;
	computer.debye_huckel.epsilon = solvent.dielectric;
	computer.onck.kappa = solvent.onck_kappa;
	computer.onck.sz = solvent.onck_sz;
	computer.onck.z = solvent.onck_z;

	return launch_kernel(resource, config, computer);
}
/// How a convolution's output is normalized.
enum class ConvolutionNormalization {
	VoxelSum, ///< sum(rho*K); matches the reference gen_pot tool
	Volume	  ///< sum(rho*K) * density cell volume
};

/**
 * @brief Convolve a density grid with a kernel grid on the device
 * @param density Source grid; its boundary condition governs the stencil taps
 * @param kernel Convolution kernel, centered on voxel (n-1)/2 per axis
 * @param resource Device to run on
 * @param norm Output normalization
 * @return New grid with density's geometry, holding the convolution
 * @note Kernel dimensions must not exceed density's. See NonBondedInteraction.md.
 */
template<typename T>
BaseGrid<T> convolve_grids(const BaseGrid<T>& density,
						   const BaseGrid<T>& kernel,
						   const Resource& resource = Resource{},
						   ConvolutionNormalization norm = ConvolutionNormalization::VoxelSum) {
	if (kernel.nx() > density.nx() || kernel.ny() > density.ny() || kernel.nz() > density.nz()) {
		throw_value_error("convolve_grids: kernel (%zu,%zu,%zu) exceeds density (%zu,%zu,%zu)",
						  kernel.nx(),
						  kernel.ny(),
						  kernel.nz(),
						  density.nx(),
						  density.ny(),
						  density.nz());
	}

	BaseGrid<T> out(density.basis(), density.origin(), density.nx(), density.ny(), density.nz());
	const T scale = (norm == ConvolutionNormalization::Volume) ? density.get_cell_volume() : T{1};

	ConvolveGridKernel<T> k{density.get_device_view(resource),
							kernel.get_device_view(resource),
							out.get_mutable_device_view(resource),
							scale};

	KernelConfig config = KernelConfig::for_1d(out.size(), resource);
	config.sync = true;
	launch_kernel(resource, config, k);

	out.sync_from_device(resource);
	return out;
}

} // namespace MARS
