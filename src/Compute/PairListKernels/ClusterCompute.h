#pragma once
#include <thrust/device_vector.h>
#include "NonbondedCompute.h"

class BoundBox;
class BitMask;

class Configuration;
class NonbondedClusterCompute : public NonbondedCompute {
public:
    NonbondedClusterCompute(const Configuration &c, const int num_replicas, const float density_scale=1.2);
    ~NonbondedClusterCompute();
    const static int BLOCK_SIZE0 = 32;
    constexpr static int NUM_THREADS = 64;
    constexpr static int cluster_size = NUM_THREADS;
    // const int cluster_size;
    
private:
    __host__ __device__
    int excl_idx(int i, int j) const {
	assert(i < j);
	return (j*(j-1))/2 + i;
    }

    CellDecomposition  decomp;
    CellDecomposition* decomp_d;


    float cutoff2; // TODO: move to Base class?
    float pairlist2; // sq. pairlist distance (including cutoff)

    float pencil_width;

    float x_min, y_min, z_min, dz;
    float3 pbc;
    int x_num, y_num;
    int num_pencils;
    
    int* pencil_particle_num_d;
    int* pencil_num_clusters_d;
    float* pencil_z_coords_d;
    int* cluster_zorder_idx_d;
    thrust::device_vector<int> cluster_order;
    thrust::device_vector<int> particle_idx;

    Vector3* mapped_pos_d;	// Arrays to store reorder particle data
    int* mapped_type_d;
    cudaTextureObject_t* mapped_pos_tex;
    
    int max_clusters;
    int2* clusters_d;
    BoundBox* bb_d;
    int* cluster_num_d;		// number of clusters
    int cluster_num;
    
    int2* neighbor_list_full; // neighboring pairs of clusters
    int* neighbor_num_d;	// number of neighboring pairs of clusters
    int neighbor_num;

    BitMask* global_mask_d; // bitmask
    BitMask* mask_d; // Per cluster bitmask, concatenated
    bool mask_filled;
    
    void decompose(const ComputeForce &compute);
    float computeTabulated(bool get_energy, const ComputeForce &compute);

};

