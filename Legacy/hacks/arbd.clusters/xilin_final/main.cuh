#ifndef MAIN
#define MAIN

#include "test.cuh"


// entry of nonbonded interaction calculation program
int run(force_t *forces, int len, float3 bound, float cutoff2, char *pos_file);

// count and allocate memory for particles of each cluster grid. Prepare for build_clusters_kernel
template<int clusterSize>
__global__ void particle_histogram_kernel(int *grid_particle_num, Particle **grid_particle_array, Point *pos_arr_d, int len, float grid_width, float x_min, float y_min, int x_num, int y_num);

// cluster particles and build bounding boxes of clusters
template<int clusterSize>
__global__ void build_clusters_kernel(Cluster<clusterSize> *clusters_d, BoundBox *bb_d, int *cluster_num_d, int *grid_particle_num, Particle **grid_particle_array, Point *pos_arr_d, int len, float grid_width, float x_min, float y_min, int x_num, int y_num);

// construct an int2 array of neighbor cluster indices
template<int clusterSize>
__global__ void build_cluster_pair_list_kernel(BoundBox *bb_d, int cluster_num, int2 *neighbor_list, int *neighbor_num_d, float cutoff2, float3 bound);

// compute nonbonded interaction and verify neighbor cluster pair 
template<int clusterSize>
__global__
void calculate_force_clusters_kernel_trial(Cluster<clusterSize> *clusters, int2 *neighbor_list, int2 *neighbor_list_compact, int *neighbor_check, int pair_num, int *pair_num_compact, float cutoff2, float3 bound, force_t *forces);

// compute nonbonded interaction
template<int clusterSize>
__global__
void calculate_force_clusters_kernel(Cluster<clusterSize> *clusters, int2 *neighbor_list, int pair_num, float cutoff2, float3 bound, force_t *forces);


#endif
