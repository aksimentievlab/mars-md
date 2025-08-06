#ifndef SETUP
#define SETUP

#include <iostream>
#include <cmath>
#include <fstream>
#include <string.h>
#include <iomanip>
#include <vector>
#include <algorithm>  
#include <omp.h>

#include <thrust/sort.h>
#include <thrust/execution_policy.h>


// position file of particles
#define POSITION_FILE (char*)"BrownDyn.restart"
// number of particles
#define LEN 8000

#define BOX_X (float)43.083//86.167
#define BOX_Y (float)43.083//86.167
#define BOX_Z (float)43.083//86.167

// number of time steps
#define STEPS 100

// cluster size
#define CLUSTER_SIZE 32
// block size for calculate_force_clusters_kernel
#define BLOCK_SIZE CLUSTER_SIZE

// block size for 
#define BLOCK_SIZE0 1024
// block size for 
#define BLOCK_SIZE1 32
// block size for build_cluster_pair_list_kernel
#define BLOCK_SIZE2 32


// lookup for LJ potential
#define RM     (float)4
#define RM6    (float)4096
#define RM12   (float)16777216 


// precision of forces
#define FLOAT   1
#define DOUBLE  0

// cutoff distance
#define SIGMA   (float)RM/1.122
#define EPSILON  1
#define CUTOFF  (float)(2.5*SIGMA)
#define CUTOFF2 100

// use bounding box or bounding sphere for cluster?
#define BOUNDBOX 1
#define BOUNDSPHERE 0

// run CUDA
#define USE_GPU 0

// run openMP
#define USE_OMP 1
// number of threads on CPU for openMP
#define THREADS_OMP 12

// benchmark
#define TIMER_FLAG 1

// flags for debugging
#define PRINT_GRID_INFO_FLAG 0
#define PRINT_GRID_PARTICLES_NUM_FLAG 0
#define PRINT_ALL_CLUSTERS 0
#define PRINT_ALL_CLUSTERS_NEIGHBORS_FLAG 0
#define CHECK_CONVERT_TO_ARRAY_FLAG 0
#define PRINT_RESULTS_FLAG 0

#define COMPARE_RESULTS_FLAG 1


#endif