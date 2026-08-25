#pragma once
#include "Interactions/Interactions.h"
#include "System/PeriodicBox.h"
#include "Types/Math.h"
#include "Types/Types.h"
#include "Types/Vector3.h"
namespace MARS {

using BondGeometry = CalcDistance;
// CalcDistance is already defined in Interactions.h since it is also used in non-bonded
// interactions

struct AngleGeometry {
	Vector3 ab, bc, ac;	 ///< Vectors
	mars_real angle;	 ///< Computed angle
	mars_real cos_angle; ///< Cosine of angle
	mars_real sin_angle; ///< Sine of angle

	DEVICE static AngleGeometry
	compute(const Vector3* positions, const int4& particle_indices, const PeriodicBox* pbox) {
		AngleGeometry geom;
		geom.ab = pbox->wrap_diff(positions[particle_indices.y] - positions[particle_indices.x]);
		geom.bc = pbox->wrap_diff(positions[particle_indices.z] - positions[particle_indices.y]);
		geom.ac = pbox->wrap_diff(positions[particle_indices.z] - positions[particle_indices.x]);

		// Compute angle using law of cosines
		mars_real distab2 = geom.ab.length2();
		mars_real distbc2 = geom.bc.length2();
		mars_real distac2 = geom.ac.length2();

		geom.cos_angle = (distab2 + distbc2 - distac2) * mars_real(0.5) /
						 (math::sqrt(distab2) * math::sqrt(distbc2));

		// Clamp cosine to valid range
		if (geom.cos_angle < mars_real(-1.0))
			geom.cos_angle = mars_real(-1.0);
		if (geom.cos_angle > mars_real(1.0))
			geom.cos_angle = mars_real(1.0);

		geom.angle = acos(geom.cos_angle);
		geom.sin_angle = math::sqrt(mars_real(1.0) - geom.cos_angle * geom.cos_angle);

		return geom;
	}
};

/**
 * @brief Dihedral angle (IUPAC sign convention) and its Cartesian gradients
 *
 * @var DihedralGeometry::f1 @f$\partial\phi/\partial r_i@f$
 * @var DihedralGeometry::f2 gradient combination for particle @f$j@f$
 * @var DihedralGeometry::f3 @f$-\partial\phi/\partial r_l@f$
 *
 * @see BondGeometry.md
 */
struct DihedralGeometry {
	Vector3 ab, bc, cd;		  // Vectors
	mars_real dihedral_angle; // Computed dihedral angle
	Vector3 f1, f2, f3;		  // force directions

	DEVICE static DihedralGeometry
	compute(const Vector3* positions, const int4& particle_indices, const PeriodicBox* pbox) {
		DihedralGeometry geom;
		geom.ab = pbox->wrap_diff(positions[particle_indices.y] - positions[particle_indices.x]);
		geom.bc = pbox->wrap_diff(positions[particle_indices.z] - positions[particle_indices.y]);
		geom.cd = pbox->wrap_diff(positions[particle_indices.t] - positions[particle_indices.z]);

		Vector3 crossABC = geom.ab.cross(geom.bc);
		Vector3 crossBCD = geom.bc.cross(geom.cd);
		Vector3 crossX = geom.bc.cross(crossABC);

		mars_real cos_phi = crossABC.dot(crossBCD) / (crossABC.length() * crossBCD.length());
		mars_real sin_phi = crossX.dot(crossBCD) / (crossX.length() * crossBCD.length());

		geom.dihedral_angle = atan2(sin_phi, cos_phi);

		geom.f1 = -geom.bc.length() * crossABC.rLength2() * crossABC;
		geom.f3 = -geom.bc.length() * crossBCD.rLength2() * crossBCD;
		geom.f2 = -(geom.ab.dot(geom.bc) * geom.bc.rLength2()) * geom.f1 -
				  (geom.bc.dot(geom.cd) * geom.bc.rLength2()) * geom.f3;

		return geom;
	}
};

/**
 * @brief Geometry for product potentials (coupled bonded interactions)
 *
 * This structure contains the geometry for complex bonded interactions
 * that couple multiple terms (e.g., BondAngle: 2 angles + 1 bond)
 */
struct ProductPotentialGeometry {
	// For BondAngle type: angle1 (i-j-k), bond (j-k), angle2 (j-k-l)
	AngleGeometry angle1;
	BondGeometry bond;
	AngleGeometry angle2;

	// For AngleAngle type: two angles sharing a bond
	AngleGeometry angle_a;
	AngleGeometry angle_b;

	// For BondBond type: two bonds sharing an atom
	BondGeometry bond_a;
	BondGeometry bond_b;

	// Combined metrics (for coupled potentials)
	mars_real combined_metric;
	bool is_singular;
	ProductPotentialGeometry() = default;
	ProductPotentialGeometry(const AngleGeometry& angle1,
							 const BondGeometry& bond,
							 const AngleGeometry& angle2)
		: angle1(angle1), bond(bond), angle2(angle2) {}
	ProductPotentialGeometry(const AngleGeometry& angle_a, const AngleGeometry& angle_b)
		: angle_a(angle_a), angle_b(angle_b) {}
	ProductPotentialGeometry(const BondGeometry& bond_a, const BondGeometry& bond_b)
		: bond_a(bond_a), bond_b(bond_b) {}
	ProductPotentialGeometry(const mars_real combined_metric, const bool is_singular)
		: combined_metric(combined_metric), is_singular(is_singular) {}

	/**
	 * @brief Compute geometry for BondAngle product potential
	 * @param positions Particle positions
	 * @param indices 4 particle indices (i, j, k, l)
	 * @param pbox Periodic boundary conditions
	 * @return BondAngle geometry
	 */
	DEVICE static ProductPotentialGeometry compute_bond_angle(DEVICE_PTR(Vector3) positions,
															  int4 indices, // i, j, k, l
															  const PeriodicBox* pbox) {
		ProductPotentialGeometry geom;

		// Compute angle 1 (i-j-k)
		geom.angle1 =
			AngleGeometry::compute(positions, int3(indices.x, indices.y, indices.z), pbox);

		// Compute bond (j-k)
		geom.bond = BondGeometry::compute(positions, int2(indices.y, indices.z), pbox);

		// Compute angle 2 (j-k-l)
		geom.angle2 =
			AngleGeometry::compute(positions, int3(indices.y, indices.z, indices.t), pbox);

		// Check for singularities
		geom.is_singular =
			(geom.angle1.angle < mars_real(1e-6) || geom.bond.distance < mars_real(1e-6) ||
			 geom.angle2.angle < mars_real(1e-6));

		// Optional: compute combined metric for coupled potentials
		// geom.combined_metric = f(geom.angle1.angle, geom.bond.distance, geom.angle2.angle);

		return geom;
	}

	/**
	 * @brief Compute geometry for AngleAngle product potential
	 * @param positions Particle positions
	 * @param indices 4 particle indices (i, j, k, l)
	 * @param pbox Periodic boundary conditions
	 * @return AngleAngle geometry
	 */
	DEVICE static ProductPotentialGeometry compute_angle_angle(DEVICE_PTR(Vector3) positions,
															   int4 indices, // i, j, k, l
															   const PeriodicBox* pbox) {
		ProductPotentialGeometry geom;

		// Compute angle 1 (i-j-k)
		geom.angle_a =
			AngleGeometry::compute(positions, int3(indices.x, indices.y, indices.z), pbox);

		// Compute angle 2 (j-k-l)
		geom.angle_b =
			AngleGeometry::compute(positions, int3(indices.y, indices.z, indices.t), pbox);

		// Check for singularities
		geom.is_singular =
			(geom.angle_a.angle < mars_real(1e-6) || geom.angle_b.angle < mars_real(1e-6));

		return geom;
	}

	/**
	 * @brief Compute geometry for BondBond product potential
	 * @param positions Particle positions
	 * @param indices 3 particle indices (i, j, k)
	 * @param pbox Periodic boundary conditions
	 * @return BondBond geometry
	 */
	DEVICE static ProductPotentialGeometry compute_bond_bond(DEVICE_PTR(Vector3) positions,
															 int3 indices, // i, j, k
															 const PeriodicBox* pbox) {
		ProductPotentialGeometry geom;

		// Compute bond 1 (i-j)
		geom.bond_a = BondGeometry::compute(positions, int2(indices.x, indices.y), pbox);

		// Compute bond 2 (j-k)
		geom.bond_b = BondGeometry::compute(positions, int2(indices.y, indices.z), pbox);

		// Check for singularities
		geom.is_singular =
			(geom.bond_a.distance < mars_real(1e-6) || geom.bond_b.distance < mars_real(1e-6));

		return geom;
	}
};

} // namespace MARS
#ifdef USE_SYCL
#include <sycl/sycl.hpp>
template<>
struct sycl::is_device_copyable<MARS::ProductPotentialGeometry> : std::true_type {};
template<>
struct sycl::is_device_copyable<MARS::AngleGeometry> : std::true_type {};
template<>
struct sycl::is_device_copyable<MARS::DihedralGeometry> : std::true_type {};
#endif
