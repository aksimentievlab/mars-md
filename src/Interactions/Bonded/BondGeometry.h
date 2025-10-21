#pragma once
#include "System/PeriodicBox.h"
#include "Types/Math.h"
#include "Types/Types.h"
#include "Types/Vector3.h"

namespace ARBD {

struct BondGeometry {
	Vector3 r_ij;		 // Bond vector
	float distance;		 // Bond distance
	Vector3 unit_vector; // Normalized direction

	DEVICE static BondGeometry
	compute(const Vector3* positions, const int2& particle_indices, const PeriodicBox* pbox) {
		BondGeometry geom;
		geom.r_ij = pbox->wrapDiff(positions[particle_indices.y] - positions[particle_indices.x]);
		geom.distance = geom.r_ij.length();
		if (geom.distance > 1e-6f) {
			geom.unit_vector = geom.r_ij / geom.distance;
		} else {
			geom.unit_vector = Vector3(0.0f);
		}
		return geom;
	}
};

struct AngleGeometry {
	Vector3 ab, bc, ac; // Vectors
	float angle;		// Computed angle
	float cos_angle;	// Cosine of angle
	float sin_angle;	// Sine of angle

	DEVICE static AngleGeometry
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
	Vector3 ab, bc, cd;			// Vectors
	float dihedral_angle;		// Computed dihedral angle
	Vector3 crossABC, crossBCD; // Cross products

	DEVICE static DihedralGeometry
	compute(const Vector3* positions, const int4& particle_indices, const PeriodicBox* pbox) {
		DihedralGeometry geom;
		geom.ab = pbox->wrapDiff(positions[particle_indices.y] - positions[particle_indices.x]);
		geom.bc = pbox->wrapDiff(positions[particle_indices.z] - positions[particle_indices.y]);
		geom.cd = pbox->wrapDiff(positions[particle_indices.w] - positions[particle_indices.z]);

		geom.crossABC = geom.ab.cross(geom.bc);
		geom.crossBCD = geom.bc.cross(geom.cd);
		Vector3 crossX = geom.bc.cross(geom.crossABC);

		float cos_phi =
			geom.crossABC.dot(geom.crossBCD) / (geom.crossABC.length() * geom.crossBCD.length());
		float sin_phi = crossX.dot(geom.crossBCD) / (crossX.length() * geom.crossBCD.length());

		geom.dihedral_angle = -atan2(sin_phi, cos_phi);

		return geom;
	}
};
} // namespace ARBD
