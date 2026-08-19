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
	DEVICE_PTR(const int2) __restrict__ particle_indices;
	DEVICE_PTR(const Vector3) __restrict__ positions;
	DEVICE_PTR(Vector3) force_energy;
	DEVICE_PTR(const arbd_real) __restrict__ params;
	const PeriodicBox* __restrict__ pbox;
	bool get_energy;
	idx_t num_bonds;

	// Constructor for initialization
	AnalyticalBondComputer(DEVICE_PTR(const int2) indices,
						   DEVICE_PTR(Vector3) pos,
						   DEVICE_PTR(Vector3) fe,
						   DEVICE_PTR(const arbd_real) p,
						   const PeriodicBox* box,
						   bool energy,
						   idx_t n)
		: particle_indices(indices), positions(pos), force_energy(fe), params(p), pbox(box),
		  get_energy(energy), num_bonds(n) {}

	// Kernel operator - called by launch_kernel
	KERNEL_FUNC void operator()(idx_t i) const {
		if (i >= num_bonds)
			return;

		const int2& indices = particle_indices[i];

		// Phase 1: Compute geometry
		BondGeometry geom = BondGeometry::compute(positions, indices, pbox);
		if (geom.distance < arbd_real(1e-6))
			return;

		// Phase 2: Compute force using analytical formula
		const ScalarForceEnergy fe = AnalyticalForceComputer<BondTypeId>::compute(
			geom.distance,
			params + (i * AnalyticalForceComputer<BondTypeId>::NUM_PARAMS));

		// Phase 3: Apply forces
		const Vector3 force = geom.unit_vector * fe.force_magnitude;
		const arbd_real energy = fe.energy * arbd_real(0.5);

		atomic_add(&force_energy[indices.x], -force);
		atomic_add(&force_energy[indices.y], force);

		if (get_energy) {
			atomic_add(&force_energy[indices.x].t, energy);
			atomic_add(&force_energy[indices.y].t, energy);
		}
	}
};

/**
 * @brief Harmonic restraint pinning one particle to a fixed point.
 *
 * @f$\vec{f} = -k\,\mathrm{wrapDiff}(\vec{r} - \vec{r}_0)@f$, one restraint per
 * work item. See dev_notes.md.
 */
struct HarmonicRestraintComputer {
	DEVICE_PTR(const int) __restrict__ particle_ids;
	DEVICE_PTR(const Vector3) __restrict__ positions;
	DEVICE_PTR(Vector3) force_energy;
	DEVICE_PTR(const Vector3) __restrict__ anchors;
	DEVICE_PTR(const arbd_real) __restrict__ spring_constants;
	const PeriodicBox* __restrict__ pbox;
	bool get_energy;
	idx_t num_restraints;

	HarmonicRestraintComputer(DEVICE_PTR(const int) ids,
							  DEVICE_PTR(Vector3) pos,
							  DEVICE_PTR(Vector3) fe,
							  DEVICE_PTR(const Vector3) r0,
							  DEVICE_PTR(const arbd_real) k,
							  const PeriodicBox* box,
							  bool energy,
							  idx_t n)
		: particle_ids(ids), positions(pos), force_energy(fe), anchors(r0), spring_constants(k),
		  pbox(box), get_energy(energy), num_restraints(n) {}

	KERNEL_FUNC void operator()(idx_t i) const {
		if (i >= num_restraints)
			return;

		const int id = particle_ids[i];
		if (id < 0)
			return;

		const CalcDistance geom = CalcDistance::compute(anchors[i], positions[id], pbox);

		const arbd_real params[AnalyticalForceComputer<0>::NUM_PARAMS] = {spring_constants[i],
																		  arbd_real(0)};
		const ScalarForceEnergy fe = AnalyticalForceComputer<0>::compute(geom.distance, params);

		atomic_add(&force_energy[id], geom.unit_vector * fe.force_magnitude);

		if (get_energy) {
			atomic_add(&force_energy[id].t, fe.energy);
		}
	}
};

/**
 * @brief Tabulated bond force computer. See dev_notes.md.
 */
struct TabulatedBondComputer {
	// Members
	DEVICE_PTR(const int2) __restrict__ particle_indices;
	DEVICE_PTR(const Vector3) __restrict__ positions;
	DEVICE_PTR(Vector3) force_energy;
	DEVICE_PTR(const TabulatedPotential) __restrict__ tables;
	DEVICE_PTR(const int) __restrict__ table_indices;
	DEVICE_PTR(const int) __restrict__ forms;
	const PeriodicBox* __restrict__ pbox;
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
	KERNEL_FUNC void operator()(idx_t i) const {
		if (i >= num_bonds)
			return;
		if (static_cast<InteractionForm>(forms[i]) != InteractionForm::Tabulated)
			return;

		const int2& indices = particle_indices[i];

		// Phase 1: Compute geometry
		BondGeometry geom = BondGeometry::compute(positions, indices, pbox);
		if (geom.distance < arbd_real(1e-6))
			return;

		// Phase 2: Lookup force from tabulated potential
		const ScalarForceEnergy fe =
			TabulatedPotential::compute(geom.distance, &tables[table_indices[i]]);

		// Phase 3: Apply forces
		const Vector3 force = geom.unit_vector * fe.force_magnitude;
		const arbd_real energy = fe.energy * arbd_real(0.5);

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
	DEVICE_PTR(const int3) __restrict__ particle_indices;
	DEVICE_PTR(const Vector3) __restrict__ positions;
	DEVICE_PTR(Vector3) force_energy;
	DEVICE_PTR(const TabulatedPotential) __restrict__ tables;
	DEVICE_PTR(const int) __restrict__ table_indices;
	DEVICE_PTR(const int) __restrict__ forms;
	const PeriodicBox* __restrict__ pbox;
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
	KERNEL_FUNC void operator()(idx_t i) const {
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

		// Phase 3: Apply forces (force1 on i, force3 on k; see BondComputer.md)
		constexpr arbd_real sin_floor = arbd_real(1e-3);
		const arbd_real sin_angle = geom.sin_angle > sin_floor ? geom.sin_angle : sin_floor;
		const arbd_real dUdtheta = -fe.force_magnitude / sin_angle;
		const arbd_real inv_ab = arbd_real(1) / math::sqrt(geom.ab.length2());
		const arbd_real inv_bc = arbd_real(1) / math::sqrt(geom.bc.length2());

		const Vector3 force1 =
			(dUdtheta * inv_ab) * (geom.ab * (geom.cos_angle * inv_ab) + geom.bc * inv_bc);
		const Vector3 force3 =
			-(dUdtheta * inv_bc) * (geom.bc * (geom.cos_angle * inv_bc) + geom.ab * inv_ab);
		const arbd_real energy = fe.energy * arbd_real(1.0 / 3.0);

		atomic_add(&force_energy[indices.x], force1);
		atomic_add(&force_energy[indices.y], -(force1 + force3));
		atomic_add(&force_energy[indices.z], force3);

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
	DEVICE_PTR(const int4) __restrict__ particle_indices;
	DEVICE_PTR(const Vector3) __restrict__ positions;
	DEVICE_PTR(Vector3) force_energy;
	DEVICE_PTR(const TabulatedPotential) __restrict__ tables;
	DEVICE_PTR(const int) __restrict__ table_indices;
	DEVICE_PTR(const int) __restrict__ forms;
	const PeriodicBox* __restrict__ pbox;
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
	KERNEL_FUNC void operator()(idx_t i) const {
		if (i >= num_dihedrals)
			return;
		if (static_cast<InteractionForm>(forms[i]) != InteractionForm::Tabulated)
			return;

		const int4& indices = particle_indices[i];

		// Phase 1: Compute geometry
		DihedralGeometry geom = DihedralGeometry::compute(positions, indices, pbox);

		// Phase 2: Lookup force from tabulated potential (see BondComputer.md)
		const ScalarForceEnergy fe =
			TabulatedPotential::compute(geom.dihedral_angle, &tables[table_indices[i]]);

		// Phase 3: Apply forces (see BondComputer.md)
		const Vector3 f1 = geom.f1 * fe.force_magnitude;
		const Vector3 f2 = geom.f2 * fe.force_magnitude;
		const Vector3 f3 = geom.f3 * fe.force_magnitude;
		const arbd_real energy = fe.energy * arbd_real(0.25);

		atomic_add(&force_energy[indices.x], f1);
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

} // namespace ARBD

// Explicit template instantiation declarations to prevent host instantiation
#ifdef USE_CUDA
#include "Backend/CUDA/KernelHelper.cuh"
namespace ARBD {
extern template struct AnalyticalBondComputer<0>;
extern template struct AnalyticalBondComputer<1>;
extern template Event launch_cuda_kernel(const Resource& resource,
										 const KernelConfig& config,
										 AnalyticalBondComputer<0> kernel_func);
extern template Event launch_cuda_kernel(const Resource& resource,
										 const KernelConfig& config,
										 AnalyticalBondComputer<1> kernel_func);
extern template Event launch_cuda_kernel(const Resource& resource,
										 const KernelConfig& config,
										 TabulatedBondComputer kernel_func);
extern template Event launch_cuda_kernel(const Resource& resource,
										 const KernelConfig& config,
										 TabulatedAngleComputer kernel_func);
extern template Event launch_cuda_kernel(const Resource& resource,
										 const KernelConfig& config,
										 TabulatedDihedralComputer kernel_func);
extern template Event launch_cuda_kernel(const Resource& resource,
										 const KernelConfig& config,
										 HarmonicRestraintComputer kernel_func);
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

template<>
struct sycl::is_device_copyable<ARBD::HarmonicRestraintComputer> : std::true_type {};
#endif
