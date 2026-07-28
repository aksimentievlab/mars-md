#pragma once
#include "Analytical.h"
#include "Backend/Kernels.h"
#include "BondGeometry.h"
#include "Interactions/BondedInteraction.h"
#include "Interactions/TabulatedPotential.h"
#include "System/PeriodicBox.h"
#include "Types/Types.h"

namespace ARBD {

// ============================================================================
// BOND COMPUTERS - Functor pattern for launch_kernel
// ============================================================================

/**
 * @brief Analytical bond force computer (Harmonic, Morse, FENE, etc.)
 * Template parameter specifies which analytical bond type
 */
template<int BondTypeId>
struct AnalyticalBondComputer {
	// Members - stored by value for device compatibility
	DEVICE_PTR(const int2) particle_indices;
	DEVICE_PTR(Vector3) positions;
	DEVICE_PTR(Vector3) force_energy;
	DEVICE_PTR(const float) params;
	const PeriodicBox* pbox;
	bool get_energy;
	idx_t num_bonds;

	// Constructor for initialization
	AnalyticalBondComputer(DEVICE_PTR(const int2) indices,
						   DEVICE_PTR(Vector3) pos,
						   DEVICE_PTR(Vector3) fe,
						   DEVICE_PTR(const float) p,
						   const PeriodicBox* box,
						   bool energy,
						   idx_t n)
		: particle_indices(indices), positions(pos), force_energy(fe), params(p), pbox(box),
		  get_energy(energy), num_bonds(n) {}

	// Kernel operator - called by launch_kernel
	DEVICE void operator()(idx_t i) const {
		if (i >= num_bonds)
			return;

		const int2& indices = particle_indices[i];

		// Phase 1: Compute geometry
		BondGeometry geom = BondGeometry::compute(positions, indices, pbox);
		if (geom.distance < 1e-6f)
			return;

		// Phase 2: Compute force using analytical formula
		const ScalarForceEnergy fe = AnalyticalForceComputer<BondTypeId>::compute(
			geom.distance,
			params + (i * AnalyticalForceComputer<BondTypeId>::NUM_PARAMS));

		// Phase 3: Apply forces
		const Vector3 force = geom.unit_vector * fe.force_magnitude;
		const float energy = fe.energy * 0.5f;

		atomic_add(&force_energy[indices.x], -force);
		atomic_add(&force_energy[indices.y], force);

		if (get_energy) {
			atomic_add(&force_energy[indices.x].t, energy);
			atomic_add(&force_energy[indices.y].t, energy);
		}
	}
};

/**
 * @brief Tabulated bond force computer
 *
 * `tables` holds one entry per distinct bond *type* (as loaded by
 * TablesRegistry), not one per bond instance - table_indices[i] says which
 * entry bond i uses (see DeviceBondedInteractions::bond_table_indices()).
 * forms[i] is an InteractionForm (Grid=0, Tabulated=1, Analytical=2); this
 * computer only processes Tabulated entries and skips the rest (Analytical
 * bonds are handled by AnalyticalBondComputer instead, via a separate launch).
 */
struct TabulatedBondComputer {
	// Members
	DEVICE_PTR(const int2) particle_indices;
	DEVICE_PTR(Vector3) positions;
	DEVICE_PTR(Vector3) force_energy;
	DEVICE_PTR(const TabulatedPotential) tables;
	DEVICE_PTR(const int) table_indices;
	DEVICE_PTR(const int) forms;
	const PeriodicBox* pbox;
	bool get_energy;
	idx_t num_bonds;

	// Constructor
	TabulatedBondComputer(DEVICE_PTR(const int2) indices,
						  DEVICE_PTR(Vector3) pos,
						  DEVICE_PTR(Vector3) fe,
						  DEVICE_PTR(const TabulatedPotential) tabs,
						  DEVICE_PTR(const int) tab_indices,
						  DEVICE_PTR(const int) bond_forms,
						  const PeriodicBox* box,
						  bool energy,
						  idx_t n)
		: particle_indices(indices), positions(pos), force_energy(fe), tables(tabs),
		  table_indices(tab_indices), forms(bond_forms), pbox(box), get_energy(energy),
		  num_bonds(n) {}

	// Kernel operator
	DEVICE void operator()(idx_t i) const {
		if (i >= num_bonds)
			return;
		if (static_cast<InteractionForm>(forms[i]) != InteractionForm::Tabulated)
			return;

		const int2& indices = particle_indices[i];

		// Phase 1: Compute geometry
		BondGeometry geom = BondGeometry::compute(positions, indices, pbox);
		if (geom.distance < 1e-6f)
			return;

		// Phase 2: Lookup force from tabulated potential
		const ScalarForceEnergy fe =
			TabulatedPotential::compute(geom.distance, &tables[table_indices[i]]);

		// Phase 3: Apply forces
		const Vector3 force = geom.unit_vector * fe.force_magnitude;
		const float energy = fe.energy * 0.5f;

		atomic_add(&force_energy[indices.x], -force);
		atomic_add(&force_energy[indices.y], force);

		if (get_energy) {
			atomic_add(&force_energy[indices.x].t, energy);
			atomic_add(&force_energy[indices.y].t, energy);
		}
	}
};

// ============================================================================
// ANGLE COMPUTERS
// ============================================================================

/**
 * @brief Tabulated angle force computer
 *
 * Same table-sharing/form-filtering scheme as TabulatedBondComputer: see its
 * comment for what table_indices/forms mean.
 */
struct TabulatedAngleComputer {
	// Members
	DEVICE_PTR(const int3) particle_indices;
	DEVICE_PTR(Vector3) positions;
	DEVICE_PTR(Vector3) force_energy;
	DEVICE_PTR(const TabulatedPotential) tables;
	DEVICE_PTR(const int) table_indices;
	DEVICE_PTR(const int) forms;
	const PeriodicBox* pbox;
	bool get_energy;
	idx_t num_angles;

	// Constructor
	TabulatedAngleComputer(DEVICE_PTR(const int3) indices,
						   DEVICE_PTR(Vector3) pos,
						   DEVICE_PTR(Vector3) fe,
						   DEVICE_PTR(const TabulatedPotential) tabs,
						   DEVICE_PTR(const int) tab_indices,
						   DEVICE_PTR(const int) angle_forms,
						   const PeriodicBox* box,
						   bool energy,
						   idx_t n)
		: particle_indices(indices), positions(pos), force_energy(fe), tables(tabs),
		  table_indices(tab_indices), forms(angle_forms), pbox(box), get_energy(energy),
		  num_angles(n) {}

	// Kernel operator
	DEVICE void operator()(idx_t i) const {
		if (i >= num_angles)
			return;
		if (static_cast<InteractionForm>(forms[i]) != InteractionForm::Tabulated)
			return;

		const int3& indices = particle_indices[i];

		// Phase 1: Compute geometry
		AngleGeometry geom = AngleGeometry::compute(positions, indices, pbox);

		// Phase 2: Lookup force from tabulated potential
		const ScalarForceEnergy fe =
			TabulatedPotential::compute(geom.angle, &tables[table_indices[i]]);

		// Phase 3: Apply forces
		const Vector3 force1 = geom.ab.cross(geom.bc) * fe.force_magnitude;
		const Vector3 force3 = geom.bc.cross(geom.ab) * fe.force_magnitude;
		const float energy = fe.energy * 0.3333333333f;

		atomic_add(&force_energy[indices.x], -force1);
		atomic_add(&force_energy[indices.y], force1 + force3);
		atomic_add(&force_energy[indices.z], -force3);

		if (get_energy) {
			atomic_add(&force_energy[indices.x].t, energy);
			atomic_add(&force_energy[indices.y].t, energy);
			atomic_add(&force_energy[indices.z].t, energy);
		}
	}
};

// ============================================================================
// DIHEDRAL COMPUTERS
// ============================================================================

/**
 * @brief Tabulated dihedral force computer
 *
 * Same table-sharing/form-filtering scheme as TabulatedBondComputer: see its
 * comment for what table_indices/forms mean.
 */
struct TabulatedDihedralComputer {
	// Members
	DEVICE_PTR(const int4) particle_indices;
	DEVICE_PTR(Vector3) positions;
	DEVICE_PTR(Vector3) force_energy;
	DEVICE_PTR(const TabulatedPotential) tables;
	DEVICE_PTR(const int) table_indices;
	DEVICE_PTR(const int) forms;
	const PeriodicBox* pbox;
	bool get_energy;
	idx_t num_dihedrals;

	// Constructor
	TabulatedDihedralComputer(DEVICE_PTR(const int4) indices,
							  DEVICE_PTR(Vector3) pos,
							  DEVICE_PTR(Vector3) fe,
							  DEVICE_PTR(const TabulatedPotential) tabs,
							  DEVICE_PTR(const int) tab_indices,
							  DEVICE_PTR(const int) dihedral_forms,
							  const PeriodicBox* box,
							  bool energy,
							  idx_t n)
		: particle_indices(indices), positions(pos), force_energy(fe), tables(tabs),
		  table_indices(tab_indices), forms(dihedral_forms), pbox(box), get_energy(energy),
		  num_dihedrals(n) {}

	// Kernel operator
	DEVICE void operator()(idx_t i) const {
		if (i >= num_dihedrals)
			return;
		if (static_cast<InteractionForm>(forms[i]) != InteractionForm::Tabulated)
			return;

		const int4& indices = particle_indices[i];

		// Phase 1: Compute geometry
		DihedralGeometry geom = DihedralGeometry::compute(positions, indices, pbox);

		// Phase 2: Lookup force from tabulated potential
		const ScalarForceEnergy fe =
			TabulatedPotential::compute(geom.dihedral_angle + BD_PI, &tables[table_indices[i]]);

		// Phase 3: Apply forces
		const Vector3 f1 = geom.f1 * fe.force_magnitude;
		const Vector3 f2 = geom.f2 * fe.force_magnitude;
		const Vector3 f3 = geom.f3 * fe.force_magnitude;
		const float energy = fe.energy * 0.25f;

		atomic_add(&force_energy[indices.x], -f1);
		atomic_add(&force_energy[indices.y], f2 - f1);
		atomic_add(&force_energy[indices.z], f3 - f2);
		atomic_add(&force_energy[indices.t], -f3);

		if (get_energy) {
			atomic_add(&force_energy[indices.x].t, energy);
			atomic_add(&force_energy[indices.y].t, energy);
			atomic_add(&force_energy[indices.z].t, energy);
			atomic_add(&force_energy[indices.t].t, energy);
		}
	}
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
									 DEVICE_PTR(const float) params,
									 const PeriodicBox* pbox,
									 bool get_energy,
									 idx_t num_bonds) {
	KernelConfig config = KernelConfig::for_1d(num_bonds, resource);

	AnalyticalBondComputer<BondTypeId>
		computer(particle_indices, positions, force_energy, params, pbox, get_energy, num_bonds);

	return launch_kernel(resource, config, computer);
}

} // namespace ARBD

// Explicit template instantiation declarations to prevent host instantiation
#ifdef USE_CUDA
#include "Backend/CUDA/KernelHelper.cuh"
namespace ARBD {
extern template struct AnalyticalBondComputer<0>;
extern template struct AnalyticalBondComputer<1>;
// Also declare launch_cuda_kernel instantiations to prevent host stub instantiation
extern template Event launch_cuda_kernel(const Resource& resource,
										 const KernelConfig& config,
										 AnalyticalBondComputer<0> kernel_func);
extern template Event launch_cuda_kernel(const Resource& resource,
										 const KernelConfig& config,
										 AnalyticalBondComputer<1> kernel_func);
// Tabulated bond/angle/dihedral computers are concrete (non-template) types,
// but still need their launch_cuda_kernel instantiation forced into a real
// CUDA compilation unit (see Bonded/BondedInstantiations.cu) - otherwise any
// host-only .cpp that calls launch_tabulated_bonds/_angles/_dihedrals (or
// Patch::calculate_bonded_forces, which calls them) implicitly instantiates
// the non-CUDA stub in KernelHelper.cuh and throws NotImplementedError.
extern template Event launch_cuda_kernel(const Resource& resource,
										 const KernelConfig& config,
										 TabulatedBondComputer kernel_func);
extern template Event launch_cuda_kernel(const Resource& resource,
										 const KernelConfig& config,
										 TabulatedAngleComputer kernel_func);
extern template Event launch_cuda_kernel(const Resource& resource,
										 const KernelConfig& config,
										 TabulatedDihedralComputer kernel_func);
} // namespace ARBD
#endif

#ifdef USE_SYCL
#include <sycl/sycl.hpp>
template<int T>
struct sycl::is_device_copyable<ARBD::AnalyticalBondComputer<T>> : std::true_type {};

template<>
struct sycl::is_device_copyable<ARBD::TabulatedBondComputer> : std::true_type {};

template<>
struct sycl::is_device_copyable<ARBD::TabulatedAngleComputer> : std::true_type {};

template<>
struct sycl::is_device_copyable<ARBD::TabulatedDihedralComputer> : std::true_type {};
#endif
