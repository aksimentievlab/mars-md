/*********************************************************************
 * @file  ComputeForce.h
 *
 * @brief Unified method for computing forces on particles and rigid bodies
 *********************************************************************/
#pragma once

#include "Interactions/Interaction.h"
#include "Objects/Particles/ParticlePatch.h"
#include "SimSystem.h"
#include "Types/IndexList.h"
#include "Types/Types.h"

namespace ARBD {

class ComputeForce {
	void updateNumber(int newNum);
	void makeTables(const BrownianParticleType* part);

	bool addTabulatedPotential(int type0, int type1);
	bool addBondPotential(int ind, Bond* bonds, BondAngle* bondAngles);
	bool addAnglePotential(int ind, Angle* angles, BondAngle* bondAngles);
	bool addDihedralPotential(int ind, Dihedral* dihedrals);

	IndexList decompDim() const;

	float decompCutoff();

	// Does nothing
	int* neighborhood(Vector3 r);

	float computeSoftcoreFull(bool get_energy);
	float computeElecFull(bool get_energy);

	float compute(bool get_energy);
	float computeFull(bool get_energy);

	// MLog: the commented function doesn't use bondList, uncomment for testing.
	/*float computeTabulated(Vector3* force, Vector3* pos, int* type,
					Bond* bonds, int2* bondMap, Exclude* excludes, int2*
	   excludeMap, Angle* angles, Dihedral* dihedrals, bool get_energy);*/
	float computeTabulated(bool get_energy);
	float computeTabulatedFull(bool get_energy);

	// MLog: new copy function to allocate memory required by ComputeForce class.
	void copyToCUDA(Vector3* forceInternal, Vector3* pos);
	void copyToCUDA(int simNum,
					int* type,
					Bond* bonds,
					int2* bondMap,
					Exclude* excludes,
					int2* excludeMap,
					Angle* angles,
					Dihedral* dihedrals,
					const Restraint* const restraints,
					const BondAngle* const bondAngles,
					const XpotMap simple_potential_map,
					const std::vector<SimplePotential> simple_potentials,
					const ProductPotentialConf* const product_potential_confs);
	void copyToCUDA(Vector3* forceInternal, Vector3* pos, Vector3* mom);
	void copyToCUDA(Vector3* forceInternal, Vector3* pos, Vector3* mom, float* random);

	// void createBondList(int3 *bondList);
	void copyBondedListsToGPU(int3* bondList,
							  int4* angleList,
							  int4* dihedralList,
							  int* dihedralPotList,
							  int4* bondAngleList,
							  int2* restraintList);

	std::vector<Vector3*> getForceInternal_d() {
		return forceInternal_d;
	}
	void setForceInternalOnDevice(Vector3* f);

	int* getType_d() {
		return type_d;
	}

	Bond* getBonds_d() {
		return bonds_d;
	}

	int2* getBondMap_d() {
		return bondMap_d;
	}

	Exclude* getExcludes_d() {
		return excludes_d;
	}

	int2* getExcludeMap_d() {
		return excludeMap_d;
	}

	Angle* getAngles_d() {
		return angles_d;
	}

	Dihedral* getDihedrals_d() {
		return dihedrals_d;
	}

	int3* getBondList_d() {
		return bondList_d;
	}

	float* getEnergy() {
		return energies_d;
	}

	void clear_force() {}
	void clear_energy() {}

	HOST DEVICE static EnergyForce coulombForce(Vector3 r, float alpha, float start, float len);

	HOST DEVICE static EnergyForce coulombForceFull(Vector3 r, float alpha);

	HOST DEVICE static EnergyForce softcoreForce(Vector3 r, float eps, float rad6);

  private:
	int numReplicas;
	int num;
	int numParts;
	int num_rb_attached_particles;
	int numBonds;
	int numExcludes;
	int numTabBondFiles;
	int numAngles;
	int numTabAngleFiles;
	int numDihedrals;
	int numTabDihedralFiles;

	int numGroupSites;
	int* comSiteParticles;
	int* comSiteParticles_d;

	float *tableEps, *tableRad6, *tableAlpha;
	TabulatedPotential** tablePot; // 100% on Host
	TabulatedPotential** tableBond;
	TabulatedAnglePotential** tableAngle;
	TabulatedDihedralPotential** tableDihedral;
	const BaseGrid* sys;
	float switchStart, switchLen, electricConst, cutoff2;
	CellDecomposition decomp;
	int numTablePots;
	float energy;

	// Device Variables
	std::vector<BaseGrid*> sys_d;
	CellDecomposition* decomp_d;
	float* energies_d;
	float *tableEps_d, *tableRad6_d, *tableAlpha_d;
	int gridSize;
	// TabulatedPotential **tablePot_d, **tablePot_addr;
	// We use this ugly approach because the array of tabulatePotentials may be
	// sparse... but it probably won't be large enough to cause problems if we
	// allocate more directly
	std::vector<TabulatedPotential**>
		tablePot_addr; // per-gpu vector of host-allocated device pointers
	std::vector<TabulatedPotential**>
		tablePot_d; // per-gpu vector of device-allocated device pointers

	TabulatedPotential **tableBond_d, **tableBond_addr;
	TabulatedAnglePotential **tableAngle_d, **tableAngle_addr;
	TabulatedDihedralPotential **tableDihedral_d, **tableDihedral_addr;

	// Pairlists
	float pairlistdist2;
	std::vector<int2*> pairLists_d;
	std::vector<cudaTextureObject_t> pairLists_tex;

	std::vector<int*> pairTabPotType_d;
	std::vector<cudaTextureObject_t> pairTabPotType_tex;

	int numPairs;
	std::vector<int*> numPairs_d;

	// Han-Yi Chou
	int* CellNeighborsList;
	// MLog: List of variables that need to be moved over to ComputeForce class.
	// Members of this list will be set to static to avoid large alterations in
	// working code, thereby allowing us to access these variables easily.
	cudaTextureObject_t neighbors_tex;

	// BrownianParticleType* part;
	// float electricConst;
	// int fullLongRange;
	std::vector<Vector3*> pos_d;
	std::vector<cudaTextureObject_t> pos_tex;
	Vector3* mom_d;
	float* ran_d;

	std::vector<Vector3*> forceInternal_d; // vector for multigpu
	int* type_d;

	Bond* bonds_d;
	int2* bondMap_d;

	Exclude* excludes_d;
	int2* excludeMap_d;

	Angle* angles_d;
	Dihedral* dihedrals_d;

	int numBondAngles;
	BondAngle* bondAngles_d;
	int4* bondAngleList_d;

	int numProductPotentials;
	float** simple_potential_pots_d;
	SimplePotential* simple_potentials_d;
	int* product_potential_particles_d;
	SimplePotential* product_potentials_d;
	uint2* product_potential_list_d;
	unsigned short* productCount_d;

	int3* bondList_d;
	int4* angleList_d;
	int4* dihedralList_d;
	int* dihedralPotList_d;
	int2* restraintList_d;

	int numRestraints;
	int* restraintIds_d;
	Vector3* restraintLocs_d;
	float* restraintSprings_d;
};
} // namespace ARBD
