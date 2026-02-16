#pragma once
#include "Interactions/Interactions.h"
#include "System/PeriodicBox.h"
#include "Types/Math.h"
#include "Types/Types.h"
#include "Types/Vector3.h"
namespace ARBD {

using BondGeometry = CalcDistance;
// CalcDistance is already defined in Interactions.h since it is also used in non-bonded
// interactions

struct AngleGeometry {
	Vector3 ab, bc, ac; // Vectors
	float angle;		// Computed angle
	float cos_angle;	// Cosine of angle
	float sin_angle;	// Sine of angle

	KERNEL_FUNC static AngleGeometry
	compute(const Vector3* positions, const int4& particle_indices, const PeriodicBox* pbox) {
		AngleGeometry geom;
		geom.ab = pbox->wrapDiff(positions[particle_indices.y] - positions[particle_indices.x]);
		geom.bc = pbox->wrapDiff(positions[particle_indices.z] - positions[particle_indices.y]);
		geom.ac = pbox->wrapDiff(positions[particle_indices.z] - positions[particle_indices.x]);

		// Compute angle using law of cosines
		float distab2 = geom.ab.length2();
		float distbc2 = geom.bc.length2();
		float distac2 = geom.ac.length2();

		geom.cos_angle =
			(distab2 + distbc2 - distac2) * 0.5f / (math::sqrt(distab2) * math::sqrt(distbc2));

		// Clamp cosine to valid range
		if (geom.cos_angle < -1.0f)
			geom.cos_angle = -1.0f;
		if (geom.cos_angle > 1.0f)
			geom.cos_angle = 1.0f;

		geom.angle = acos(geom.cos_angle);
		geom.sin_angle = math::sqrt(1.0f - geom.cos_angle * geom.cos_angle);

		return geom;
	}
};

struct DihedralGeometry {
	Vector3 ab, bc, cd;	  // Vectors
	float dihedral_angle; // Computed dihedral angle
	Vector3 f1, f2, f3;	  // force directions

	KERNEL_FUNC static DihedralGeometry
	compute(const Vector3* positions, const int4& particle_indices, const PeriodicBox* pbox) {
		DihedralGeometry geom;
		geom.ab = pbox->wrapDiff(positions[particle_indices.y] - positions[particle_indices.x]);
		geom.bc = pbox->wrapDiff(positions[particle_indices.z] - positions[particle_indices.y]);
		geom.cd = pbox->wrapDiff(positions[particle_indices.t] - positions[particle_indices.z]);

		Vector3 crossABC = geom.ab.cross(geom.bc);
		Vector3 crossBCD = geom.bc.cross(geom.cd);
		Vector3 crossX = geom.bc.cross(crossABC);

		float cos_phi = crossABC.dot(crossBCD) / (crossABC.length() * crossBCD.length());
		float sin_phi = crossX.dot(crossBCD) / (crossX.length() * crossBCD.length());

		geom.dihedral_angle = -atan2(sin_phi, cos_phi);

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
	float combined_metric;
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
	ProductPotentialGeometry(const float combined_metric, const bool is_singular)
		: combined_metric(combined_metric), is_singular(is_singular) {}

	/**
	 * @brief Compute geometry for BondAngle product potential
	 * @param positions Particle positions
	 * @param indices 4 particle indices (i, j, k, l)
	 * @param pbox Periodic boundary conditions
	 * @return BondAngle geometry
	 */
	KERNEL_FUNC static ProductPotentialGeometry compute_bond_angle(DEVICE_PTR(Vector3) positions,
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
			(geom.angle1.angle < 1e-6f || geom.bond.distance < 1e-6f || geom.angle2.angle < 1e-6f);

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
	KERNEL_FUNC static ProductPotentialGeometry compute_angle_angle(DEVICE_PTR(Vector3) positions,
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
		geom.is_singular = (geom.angle_a.angle < 1e-6f || geom.angle_b.angle < 1e-6f);

		return geom;
	}

	/**
	 * @brief Compute geometry for BondBond product potential
	 * @param positions Particle positions
	 * @param indices 3 particle indices (i, j, k)
	 * @param pbox Periodic boundary conditions
	 * @return BondBond geometry
	 */
	KERNEL_FUNC static ProductPotentialGeometry compute_bond_bond(DEVICE_PTR(Vector3) positions,
																  int3 indices, // i, j, k
																  const PeriodicBox* pbox) {
		ProductPotentialGeometry geom;

		// Compute bond 1 (i-j)
		geom.bond_a = BondGeometry::compute(positions, int2(indices.x, indices.y), pbox);

		// Compute bond 2 (j-k)
		geom.bond_b = BondGeometry::compute(positions, int2(indices.y, indices.z), pbox);

		// Check for singularities
		geom.is_singular = (geom.bond_a.distance < 1e-6f || geom.bond_b.distance < 1e-6f);

		return geom;
	}
};

} // namespace ARBD
#ifdef USE_SYCL
#include <sycl/sycl.hpp>
template<>
struct sycl::is_device_copyable<ARBD::ProductPotentialGeometry> : std::true_type {};
template<>
struct sycl::is_device_copyable<ARBD::AngleGeometry> : std::true_type {};
template<>
struct sycl::is_device_copyable<ARBD::DihedralGeometry> : std::true_type {};
#endif
