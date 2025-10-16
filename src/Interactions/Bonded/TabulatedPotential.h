#pragma once
#include "Constants.h"
#include "Header.h"
#include "Objects/ForceEnergy.h"
#include "System/PeriodicBox.h"
#include "Types/BaseGrid.h"
#include "Types/Types.h"

namespace ARBD {
enum BondedPotentialType { UNSET, BOND, ANGLE, DIHEDRAL, VECANGLE };
auto BD_PI = constants::PI;

class TabulatedPotential {
  public:
	TabulatedPotential();
	TabulatedPotential(const TabulatedPotential& tab);
	TabulatedPotential(const float* dist, const float* pot, int n0);
	~TabulatedPotential();

	void truncate(float cutoff);
	bool truncate(float switchDist, float cutoff, float value);

	Vector3 computeForce(Vector3 r);

	int size() const {
		return n;
	}

	HOST DEVICE inline ForceEnergy compute(Vector3 r) {
		float d = r.length();
		float w = (d - r0) * drInv;
		int home = int(floorf(w));
		w = w - home;
		// if (home < 0) return ForceEnergy(v0[0], Vector3(0.0f));
		home = home < 0 ? 0 : home;
		if (home >= n)
			return ForceEnergy(Vector3(0.0f), v0[n - 1]);

		float u0 = v0[home];
		float du = home + 1 < n ? v0[home + 1] - u0 : 0;

		// Interpolate.
		float energy = du * w + u0;
		Vector3 force = (du * drInv / d) * r;
		return ForceEnergy(force, energy);
	}

	HOST DEVICE inline ForceEnergy compute(const Vector3 r, float d) const {
		d = sqrt(d);
		// float d = r.length();
		float w = (d - r0) * drInv;
		int home = int(floorf(w));
		w = w - home;
		// if (home < 0) return ForceEnergy(v0[0], Vector3(0.0f));
		home = home < 0 ? 0 : home;
		if (home >= n)
			return ForceEnergy(Vector3(0.0f), v0[n - 1]);

		float u0 = v0[home];
		float du = home + 1 < n ? v0[home + 1] - u0 : 0;

		// Interpolate.
		float energy = du * w + u0;
		Vector3 force = (du * drInv / d) * r;
		return ForceEnergy(force, energy);
	}
	HOST DEVICE inline Vector3 computef(const Vector3 r, float d) const {
		d = sqrt(d);
		// float d = r.length();
		// RBTODO: precompute so that initial blocks are zero; reduce computation here
		float w = (d - r0) * drInv;
		int home = int(floorf(w));
		w = w - home;
		// if (home < 0) return ForceEnergy(v0[0], Vector3(0.0f));
		home = home < 0 ? 0 : home;
		if (home >= n)
			return Vector3(0.0f);

		if (home + 1 < n)
			return ((v0[home + 1] - v0[home]) * drInv / d) * r;
		else
			return Vector3(0.0f);
	}

	// private:
  private:
	float* v0;
	int n;
	float drInv; // TODO replace with drInv
	float r0;
};

class BondedPotential {
  public:
	BondedPotential() {}
	BondedPotential(float* pot, float step_inv, unsigned int size, BondedPotentialType type)
		: pot(pot), step_inv(step_inv), size(size), type(type) {}

	float* pot;		   // actual potential values
	float step_inv;	   // angular increments of potential file
	unsigned int size; // number of data points in the file

	BondedPotentialType type;

	HOST DEVICE inline float compute_value(const Vector3* __restrict__ pos,
										   const BaseGrid<float>* __restrict__ sys,
										   const int* __restrict__ particles) const {
		float val;
		if (type == BOND)
			val = compute_bond(pos, sys, particles[0], particles[1]);
		else if (type == ANGLE)
			val = compute_angle(pos, sys, particles[0], particles[1], particles[2]);
		else if (type == DIHEDRAL)
			val =
				compute_dihedral(pos, sys, particles[0], particles[1], particles[2], particles[3]);
		else if (type == VECANGLE)
			val =
				compute_vecangle(pos, sys, particles[0], particles[1], particles[2], particles[3]);
		return val;
	}

	HOST DEVICE inline float2 compute_energy_and_deriv(float value) {
		float2 ret;
		if (type == DIHEDRAL) {
			ret = linearly_interpolate<true>(value, -BD_PI);
		} else {
			ret = linearly_interpolate<false>(value);
		}
	}

	HOST DEVICE inline float compute_bond(const Vector3* __restrict__ pos,
										  const PeriodicBox* __restrict__ pbox,
										  int i,
										  int j) const {
		return pbox->wrapDiff(pos[j] - pos[i]).length();
	}

	HOST DEVICE inline float compute_angle(const Vector3* __restrict__ pos,
										   const PeriodicBox* __restrict__ pbox,
										   int i,
										   int j,
										   int k) const {
		const Vector3 ab = pbox->wrapDiff(pos[j] - pos[i]);
		const Vector3 bc = pbox->wrapDiff(pos[k] - pos[j]);
		const Vector3 ac = pbox->wrapDiff(pos[k] - pos[i]);
		return compute_angle(ab.length2(), bc.length2(), ac.length2());
	}

	HOST DEVICE inline float compute_vecangle(const Vector3* __restrict__ pos,
											  const PeriodicBox* __restrict__ pbox,
											  int i,
											  int j,
											  int k,
											  int l) const {
		const Vector3 ab = pbox->wrapDiff(pos[j] - pos[i]);
		const Vector3 bc = pbox->wrapDiff(pos[l] - pos[k]);
		const Vector3 ac = bc + ab;
		return compute_angle(ab.length2(), bc.length2(), ac.length2());
	}

	HOST DEVICE inline float compute_angle(float distab2, float distbc2, float distac2) const {
		// Find the cosine of the angle we want - <ABC
		float cos = (distab2 + distbc2 - distac2);

		distab2 = 1.0f / sqrt(distab2); // TODO: test other functions
		distbc2 = 1.0f / sqrt(distbc2);
		cos *= 0.5f * distbc2 * distab2;

		// If the cosine is illegitimate, set it to 1 or -1 so that acos won't fail
		if (cos < -1.0f)
			cos = -1.0f;
		if (cos > 1.0f)
			cos = 1.0f;

		return acos(cos);
	}

	HOST DEVICE inline float compute_dihedral(const Vector3* __restrict__ pos,
											  const PeriodicBox* __restrict__ pbox,
											  int i,
											  int j,
											  int k,
											  int l) const {
		const Vector3 ab = -pbox->wrapDiff(pos[j] - pos[i]);
		const Vector3 bc = -pbox->wrapDiff(pos[k] - pos[j]);
		const Vector3 cd = -pbox->wrapDiff(pos[l] - pos[k]);

		const Vector3 crossABC = ab.cross(bc);
		const Vector3 crossBCD = bc.cross(cd);
		const Vector3 crossX = bc.cross(crossABC);

		const float cos_phi = crossABC.dot(crossBCD) / (crossABC.length() * crossBCD.length());
		const float sin_phi = crossX.dot(crossBCD) / (crossX.length() * crossBCD.length());

		return -atan2(sin_phi, cos_phi);
	}

	template<bool is_periodic>
	HOST DEVICE inline float2 linearly_interpolate(float x, float start = 0.0f) const {
		float w = (x - start) * step_inv;
		int home = int(floorf(w));
		w = w - home;
		// if (home < 0) return ForceEnergy(v0[0], Vector3(0.0f));
		if (home < 0) {
			if (is_periodic)
				home += size;
			else
				return float2(pot[0], 0.0f);
		} else if (home >= size) {
			if (is_periodic)
				home -= size;
			else
				return float2(pot[size - 1], 0.0f);
		}

		float u0 = pot[home];
		float du = home + 1 < size ? pot[home + 1] - u0 : is_periodic ? pot[0] - u0 : 0;

		return float2(du * w + u0, du * step_inv);
	}

	DEVICE inline void apply_force(const Vector3* __restrict__ pos,
								   const BaseGrid<float>* __restrict__ sys,
								   Vector3* __restrict__ forces,
								   int* particles,
								   float energy_deriv) const {
		if (type == BOND)
			apply_bond_force(pos, sys, forces, particles[0], particles[1], energy_deriv);
		else if (type == ANGLE)
			apply_angle_force(pos,
							  sys,
							  forces,
							  particles[0],
							  particles[1],
							  particles[2],
							  energy_deriv);
		else if (type == DIHEDRAL)
			apply_dihedral_force(pos,
								 sys,
								 forces,
								 particles[0],
								 particles[1],
								 particles[2],
								 particles[3],
								 energy_deriv);
		else if (type == VECANGLE)
			apply_vecangle_force(pos,
								 sys,
								 forces,
								 particles[0],
								 particles[1],
								 particles[2],
								 particles[3],
								 energy_deriv);
	}

	DEVICE inline void apply_bond_force(const Vector3* __restrict__ pos,
										const BaseGrid<float>* __restrict__ sys,
										Vector3* __restrict__ force,
										int i,
										int j,
										float energy_deriv) const {

		Vector3 f = sys->wrapDiff(pos[j] - pos[i]);
		f = f * energy_deriv / f.length();
		atomic_add(&force[i], f);
		atomic_add(&force[j], -f);
	}

	struct TwoVector3 {
		Vector3 v1;
		Vector3 v2;
	};

	DEVICE inline TwoVector3
	get_angle_force(const Vector3& ab, const Vector3& bc, float energy_deriv) const {
		// Find the distance between each pair of particles
		float distab = ab.length2();
		float distbc = bc.length2();
		const float distac2 = (ab + bc).length2();

		// Find the cosine of the angle we want - <ABC
		float cos = (distab + distbc - distac2);

		distab = 1.0f / sqrt(distab); // TODO: test other functions
		distbc = 1.0f / sqrt(distbc);
		cos *= 0.5f * distbc * distab;

		// If the cosine is illegitimate, set it to 1 or -1 so that acos won't fail
		if (cos < -1.0f)
			cos = -1.0f;
		if (cos > 1.0f)
			cos = 1.0f;

		float sin = sqrtf(1.0f - cos * cos);
		energy_deriv /= abs(sin) > 1e-3 ? sin : 1e-3; // avoid singularity
		if (abs(sin) < 1e-3) {
			printf("BAD ANGLE: sin, cos, energy_deriv, distab, distbc, distac2: (%f %f %f %f %f)\n",
				   sin,
				   cos,
				   energy_deriv,
				   distab,
				   distbc);
		}

		// Calculate the forces
		TwoVector3 force;
		force.v1 =
			(energy_deriv * distab) * (ab * (cos * distab) + bc * distbc); // force on 1st particle
		force.v2 = -(energy_deriv * distbc) *
				   (bc * (cos * distbc) + ab * distab); // force on last particle
		return force;
	}

	DEVICE inline void apply_angle_force(const Vector3* __restrict__ pos,
										 const BaseGrid<float>* __restrict__ sys,
										 Vector3* __restrict__ force,
										 int i,
										 int j,
										 int k,
										 float energy_deriv) const {

		const Vector3 ab = sys->wrapDiff(pos[j] - pos[i]);
		const Vector3 bc = sys->wrapDiff(pos[k] - pos[j]);
		// const Vector3 ac = sys->wrapDiff(pos[k] - pos[i]);

		TwoVector3 f = get_angle_force(ab, bc, energy_deriv);

		atomicAdd(&force[i], f.v1);
		atomicAdd(&force[j], -(f.v1 + f.v2));
		atomicAdd(&force[k], f.v2);
	}

	DEVICE inline void apply_dihedral_force(const Vector3* __restrict__ pos,
											const BaseGrid<float>* __restrict__ sys,
											Vector3* __restrict__ force,
											int i,
											int j,
											int k,
											int l,
											float energy_deriv) const {

		const Vector3 ab = -sys->wrapDiff(pos[j] - pos[i]);
		const Vector3 bc = -sys->wrapDiff(pos[k] - pos[j]);
		const Vector3 cd = -sys->wrapDiff(pos[l] - pos[k]);

		const Vector3 crossABC = ab.cross(bc);
		const Vector3 crossBCD = bc.cross(cd);
		const Vector3 crossX = bc.cross(crossABC);

		const float cos_phi = crossABC.dot(crossBCD) / (crossABC.length() * crossBCD.length());
		const float sin_phi = crossX.dot(crossBCD) / (crossX.length() * crossBCD.length());

		// return -atan2(sin_phi,cos_phi);
		Vector3 f1, f2, f3; // forces
		float distbc = bc.length();

		f1 = -distbc * crossABC.rLength2() * crossABC;
		f3 = -distbc * crossBCD.rLength2() * crossBCD;
		f2 = -(ab.dot(bc) * bc.rLength2()) * f1 - (bc.dot(cd) * bc.rLength2()) * f3;

		f1 *= energy_deriv;
		f2 *= energy_deriv;
		f3 *= energy_deriv;

		atomicAdd(&force[i], f1);
		atomicAdd(&force[j], f2 - f1);
		atomicAdd(&force[k], f3 - f2);
		atomicAdd(&force[l], -f3);
	}
	DEVICE inline void apply_vecangle_force(const Vector3* __restrict__ pos,
											const BaseGrid<float>* __restrict__ sys,
											Vector3* __restrict__ force,
											int i,
											int j,
											int k,
											int l,
											float energy_deriv) const {

		const Vector3 ab = -sys->wrapDiff(pos[j] - pos[i]);
		const Vector3 bc = -sys->wrapDiff(pos[k] - pos[j]);
		// const Vector3 ac = sys->wrapDiff(pos[k] - pos[i]);

		TwoVector3 f = get_angle_force(ab, bc, energy_deriv);

		atomicAdd(&force[i], f.v1);
		atomicAdd(&force[j], -f.v1);
		atomicAdd(&force[k], -f.v2);
		atomicAdd(&force[l], f.v2);
	}
};

} // namespace ARBD
