#pragma once
#include "CellDecomposition.h"

class Configuration;

class NonbondedVerletCompute : public NonbondedCompute {
public:
    NonbondedVerletCompute(const Configuration &c, const int num_replicas);
    ~NonbondedVerletCompute();

    void decompose(const ComputeForce &compute);
    float computeTabulated(bool get_energy, const ComputeForce &compute);

private:
    CellDecomposition decomp;
    CellDecomposition* decomp_d;

    float cutoff2; // TODO: move to Base class?

    //Han-Yi Chou
    int *CellNeighborsList;

    // Pairlists
    int numPairs;
    std::vector<int*> numPairs_d;

    float pairlistdist2;
    std::vector<int2*> pairLists_d;
    std::vector<cudaTextureObject_t> pairLists_tex;

    std::vector<int*> pairTabPotType_d;
    std::vector<cudaTextureObject_t> pairTabPotType_tex;

    cudaTextureObject_t neighbors_tex;

};
