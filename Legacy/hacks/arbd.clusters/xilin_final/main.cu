#include "main.cuh"

int main(){
    force_t forces[LEN];
    memset(forces, 0.0f, sizeof(force_t)*LEN);
    run(forces, LEN, make_float3(BOX_X, BOX_Y, BOX_Z), CUTOFF2, POSITION_FILE);

    #if (COMPARE_RESULTS_FLAG == 1)
    COMPARE_RESULTS 
    #endif

    // for leak checking in cuda-memcheck
    cudaDeviceReset();
    return 0;
}

/* entry of nonbonded interaction calculation program
 * @params: forces: caller-allocated array of forces [output]
 *             len: number of particles
 *           bound: boundary of simulation box
 *          cutoff: square of cutoff distance
 *        pos_file: particle location file name(e.g. "BrownDyn.restart")
 */
int run(force_t *forces, int len, float3 bound, float cutoff2, char *pos_file){
    #if (TIMER_FLAG == 1)
    float time;
    cudaEvent_t start, stop;
    INFO(USE_GPU, BOUNDBOX, BOUNDSPHERE, FLOAT, DOUBLE)
    std::cout<<"Benchmark for cluster-cluster simulation:"<<std::endl;
    #endif

    /*************************************************************************/
    /* step 1: figure out grid and cluster information                       */
    /*************************************************************************/

    // read positions of particles from restart file, pass in position array directly in arbd
    Point *pos_arr = new Point[len];
    read_position(pos_file, pos_arr, len);

    #if (TIMER_FLAG == 1)
    TIMER_START("construct clusters")
    #endif

    // find simulation box size, pass in the size directly in arbd
    BoundBox range = simulation_range(pos_arr, len); // O(N)
    float x_min = range.x0;
    float y_min = range.y0;
    float z_min = range.z0;
    float x_range = range.x1- x_min;
    float y_range = range.y1 - y_min;
    float z_range = range.z1 - z_min;

    // set grid base width
    float volume = (x_range * y_range * z_range);
    float density = (float)len / volume;
    float grid_width = (float) pow((CLUSTER_SIZE / density),(float)1.0/3.0);

    // calculate the number of grids in x, y dimension
    int x_num = int(round(x_range/grid_width));
    int y_num = int(round(y_range/grid_width));


    #if (PRINT_GRID_INFO_FLAG == 1)
    PRINT_GRID_INFO
    #endif

    /*************************************************************************/
    /* step 2: construct clusters and bounding boxes                         */
    /*************************************************************************/
    
    Point *pos_arr_d; // position of particles

    Cluster<CLUSTER_SIZE> *clusters_d; // sufficiently large cluster array
    BoundBox *bb_d;                    // Boundbox array with same size as clusters_d 
    int *cluster_num_d;                // somewhere to store the actual number of clusters

    Particle **grid_particles_d;       // pointer to an array of to-be-allocated particles for each grid
    int *grid_particle_num;            // number of particles for each grid 

    /* step 2.1: count and allocate memory for particles of each cluster grid. Prepare for build_clusters_kernel */
    gpuErrchk(cudaMalloc((void**)&clusters_d, sizeof(Cluster<CLUSTER_SIZE>)*len/CLUSTER_SIZE*2));
    gpuErrchk(cudaMalloc((void**)&bb_d, sizeof(BoundBox)*len/CLUSTER_SIZE*2));
    gpuErrchk(cudaMalloc((void**)&pos_arr_d, sizeof(Point)*len));
    gpuErrchk(cudaMemcpy(pos_arr_d, pos_arr, sizeof(Point)*len, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMalloc((void**)&grid_particles_d, sizeof(Particle*)*x_num*y_num));
    gpuErrchk(cudaMalloc((void**)&grid_particle_num, sizeof(int)*x_num*y_num));
    gpuErrchk(cudaMemset(grid_particle_num, 0, sizeof(int)*x_num*y_num));
    gpuErrchk(cudaMalloc((void**)&cluster_num_d, sizeof(int)));
    gpuErrchk(cudaMemset(cluster_num_d, 0, sizeof(int)));

    cudaSafeCall((particle_histogram_kernel<CLUSTER_SIZE><<<dim3(1,1,1), dim3(BLOCK_SIZE0,1,1)>>>
    (grid_particle_num, grid_particles_d, pos_arr_d, len, grid_width, x_min, y_min, x_num, y_num)));

    /* step 2.2: cluster particles and build bounding boxes of clusters */
    cudaSafeCall((build_clusters_kernel<CLUSTER_SIZE><<<dim3(x_num*y_num,1,1), dim3(BLOCK_SIZE1,1,1)>>>
    (clusters_d, bb_d, cluster_num_d, grid_particle_num, grid_particles_d, pos_arr_d, len, grid_width, x_min, y_min, x_num, y_num)));

    int cluster_num;  // actual number of clusters
    
    gpuErrchk(cudaFree(pos_arr_d));
    gpuErrchk(cudaFree(grid_particles_d));
    gpuErrchk(cudaFree(grid_particle_num));
    gpuErrchk(cudaMemcpy(&cluster_num, cluster_num_d, sizeof(int), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaFree(cluster_num_d));

    #if (TIMER_FLAG == 1)
    TIMER_END
    #endif
    /*************************************************************************/
    /* step 3: construct neighbor cluster pair list for all clusters,        */
    /* pair list contains indices pair(int2) of neighbor clusters            */
    /*************************************************************************/

    #if (TIMER_FLAG == 1)
    TIMER_START("build neighbor cluster pair list on GPU")
    #endif

    int2 *neighbor_list_full;          // sufficiently large neighbr cluster pair array
    int2 *neighbor_list_d;             // shortened neighbor cluster pair array 
    int  *neighbor_num_d;              // number of neighbor cluster pair, including dummy pairs
    int  *pair_num_d;                  // actual number of neighbor cluster
    int  *neighbor_check_d;            // number of actual neighbor particles for each cluster pair
    int neighbor_num;                  
    int pair_num; 

    gpuErrchk(cudaMalloc((void**)&neighbor_list_full, sizeof(int2)*cluster_num*cluster_num>>1));
    gpuErrchk(cudaMalloc((void**)&neighbor_num_d, sizeof(int)));  
    gpuErrchk(cudaMemset(neighbor_num_d, 0, sizeof(int)));

    build_cluster_pair_list_kernel<CLUSTER_SIZE><<<dim3(cluster_num, 1, 1),dim3(BLOCK_SIZE2,1,1)>>>(bb_d, cluster_num, neighbor_list_full, neighbor_num_d, cutoff2, bound);

    gpuErrchk(cudaFree(bb_d));
    gpuErrchk(cudaMemcpy(&neighbor_num, neighbor_num_d, sizeof(int), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMalloc((void**)&neighbor_list_d, sizeof(int2)*neighbor_num));
    gpuErrchk(cudaMalloc((void**)&neighbor_check_d, sizeof(int)*neighbor_num));
    gpuErrchk(cudaMalloc((void**)&pair_num_d, sizeof(int)));
    gpuErrchk(cudaMemset(neighbor_check_d, 0, sizeof(int)*neighbor_num));
    gpuErrchk(cudaMemset(pair_num_d, 0, sizeof(int)));

    #if (TIMER_FLAG == 1)
    TIMER_END
    #endif

    /*************************************************************************/
    /* step 4: calculate force for every particle in every cluster           */
    /*************************************************************************/
    #if (TIMER_FLAG == 1)
    TIMER_START("allocate memory for forces on GPU")
    #endif 
    /* step 4.0 allocate memory on GPU */
    force_t *forces_d = NULL;
    gpuErrchk(cudaMalloc((void**)&forces_d, sizeof(force_t)*len));
    gpuErrchk(cudaMemset(forces_d, 0.0f, sizeof(force_t)*len));
    #if (TIMER_FLAG == 1)
    TIMER_END
    #endif

    /* step 4.1: calculate force for every particles in every cluster */
    dim3 block_dim(BLOCK_SIZE, 1, 1);
    dim3 grid_dim(std::min(neighbor_num, 65535), 1, 1);
    for(int i =0; i < STEPS; i++){
        // clear the forces for testing
        if(i) gpuErrchk(cudaMemset(forces_d, 0.0f, sizeof(force_t)*len));

        #if (TIMER_FLAG == 1)
        TIMER_START("compute non-bonded forces on GPU for 100 times")
        #endif

        /* step 4.1.0: In the first launch of kernel, construct a compacted cluster pair list(neighbor_list_d) besides computing the interaction.
         * Some cluster pairs in neighbor_list_full have no neighbor particle pair because distance between bounding boxes is the lower bound of distance between particles. 
         * Detect if the cluster pair has at least one neighbor particle pair and if so, push the cluster pair to neighbor_list_d
         */
        if(i ==0){
            cudaSafeCall((calculate_force_clusters_kernel_trial<CLUSTER_SIZE><<<grid_dim, block_dim>>>(clusters_d, neighbor_list_full, neighbor_list_d, neighbor_check_d, neighbor_num, pair_num_d, cutoff2, bound, forces_d)));
        }
        /* step 4.1.1: In the second launch of kernel, cudaMemcpy back to CPU the actual number of valid cluster pair(pair_num). Lauch the kernel with pair_num as the grid dimension
         * Free neighbor_list_full, neighbor_check_d, neighbor_num_d, pair_num_d
         */
        else if(i == 1){
            gpuErrchk(cudaMemcpy(&pair_num, pair_num_d, sizeof(int), cudaMemcpyDeviceToHost));
            gpuErrchk(cudaFree(neighbor_list_full));
            gpuErrchk(cudaFree(neighbor_check_d));
            gpuErrchk(cudaFree(neighbor_num_d));
            gpuErrchk(cudaFree(pair_num_d));
            grid_dim = dim3(std::min(pair_num, 65535), 1, 1);
            cudaSafeCall((calculate_force_clusters_kernel<CLUSTER_SIZE><<<grid_dim, block_dim>>>(clusters_d, neighbor_list_d, pair_num, cutoff2, bound, forces_d)));
        }
        /* step 4.1.2: In the second and future launches, call calculate_force_clusters_kernel with neighbor_list_d
         */
        else{
            cudaSafeCall((calculate_force_clusters_kernel<CLUSTER_SIZE><<<grid_dim, block_dim>>>(clusters_d, neighbor_list_d, pair_num, cutoff2, bound, forces_d)));
        }

        #if (TIMER_FLAG == 1)
        TIMER_END
        #endif

    }

    #if (TIMER_FLAG == 1)
    TIMER_START("transfer and deallocate memory on GPU and CPU")
    #endif

    /* step 4.2: transfer result and deallocate memory */
    gpuErrchk(cudaMemcpy(forces, forces_d, sizeof(force_t)*len, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaFree(forces_d));
    gpuErrchk(cudaFree(neighbor_list_d));
    gpuErrchk(cudaFree(clusters_d));
    
    #if (TIMER_FLAG == 1)
    TIMER_END
    #endif

    #if (PRINT_RESULTS_FLAG == 1)
    PRINT_RESULTS
    #endif

    return 0;
}


/* count and allocate memory for particles of each cluster grid. Prepare for build_clusters_kernel
 * template: clusterSize: number of particles of each cluster
 * @params:   grid_particle_num: caller-allocated int array(length of x_num*y_num) to store the number of particles for each cluster grid [output]
 *          grid_particle_array: caller-allocated Particle* array(length of x_num*y_num), each Particle* will be allocated a dynamic array of particles [output]
 *                    pos_arr_d: position of all particles
 *                          len: number of particles
 *                   grid_width: physical grid width
 *                        x_min: minimum x value of all particles
 *                        y_min: minimum y value of all particles
 *                        x_num: number of grids in x direction
 *                        y_num: number of grids in y direction
 *
 * NOTE: particle_histogram_kernel should be called before build_clusters_kernel
 */
template<int clusterSize>
__global__ void particle_histogram_kernel(int *grid_particle_num, Particle **grid_particle_array, Point *pos_arr_d, int len, float grid_width, float x_min, float y_min, int x_num, int y_num){
    int tid = threadIdx.x;

    // all threads histogram particles 
    for(int i = tid; i < len; i +=  BLOCK_SIZE0){
        // determine grid index
        int x_idx = (pos_arr_d[i].x - x_min) / grid_width;
        if(x_idx >= x_num) x_idx = x_num-1;
        int y_idx = (pos_arr_d[i].y - y_min) / grid_width;
        if(y_idx >= y_num) y_idx = y_num-1;
        atomicAdd(&grid_particle_num[x_idx*y_num + y_idx], 1);
    }
    
    // first x_num*y_num threads allocate memory for grid_particle_array[]
    __syncthreads();
    for(int grid_idx = tid; grid_idx < x_num*y_num; grid_idx += BLOCK_SIZE0){
        grid_particle_array[grid_idx] = (Particle*)malloc(sizeof(Particle) * grid_particle_num[grid_idx]);
    }
}


/* cluster particles and build bounding boxes of clusters
 * template: clusterSize: number of particles of each cluster
 * @params:    clusters_d: caller-allocated sufficiently large Cluster array to store constructed clusters [output]
 *                   bb_d: caller-allocated sufficiently large BoundBox array to store bounding boxes of clusters [output]
 *          cluster_num_d: a int pointer to somewhere to store the number of clusters constructed[output]
 *      grid_particle_num: number of particles for each cluster gird
 *    grid_particle_array: a pointer to an array of allocated space to temporarily store particles for sorting and clustering
 *              pos_arr_d: position of all particles
 *                    len: number of particles
 *             grid_width: physical grid width
 *                  x_min: minimum x value of all particles
 *                  y_min: minimum y value of all particles
 *                  x_num: number of grids in x direction
 *                  y_num: number of grids in y direction
 *
 * NOTE: 1. number of clusters is slightly larger than len/clusterSize, allocate 2 * (len/clusterSize) elements for clusters_d and bb_d for safety
 *       2. build_clusters_kernel should be called immediately after particle_histogram_kernel
 */
template<int clusterSize>
__global__ void build_clusters_kernel(Cluster<clusterSize> *clusters_d, BoundBox *bb_d, int *cluster_num_d, int *grid_particle_num, Particle **grid_particle_array, Point *pos_arr_d, int len, float grid_width, float x_min, float y_min, int x_num, int y_num){
    int grid_idx = blockIdx.x;
    int tid = threadIdx.x;
    int grid_num =  grid_particle_num[grid_idx];

    __shared__ int counter[1];
    if(tid == 0) counter[0] = 0;
    __syncthreads();

    // all thread in grid_idx-th thread box search for particles belonging to grid_idx-th grid
    for(int i = tid; i < len; i += blockDim.x){
        // determine grid index
        int x_idx = (pos_arr_d[i].x - x_min) / grid_width;
        if(x_idx >= x_num) x_idx = x_num-1;
        int y_idx = (pos_arr_d[i].y - y_min) / grid_width;
        if(y_idx >= y_num) y_idx = y_num-1;
        if(grid_idx == x_idx * y_num + y_idx){
            // critical section to push particle to grid_particle_array[grid_idx]
            int part_idx = atomicAdd(counter, 1);
            grid_particle_array[grid_idx][part_idx] = Particle(pos_arr_d[i], i);
        }
    }
    __syncthreads();

    // 0th thread sorts all particles in the grid
    if(tid == 0)
        thrust::sort(thrust::seq, grid_particle_array[grid_idx], grid_particle_array[grid_idx] + grid_num);
    __syncthreads();

    // first ceil(1.0*grid_num/clusterSize) threads construct clusters
    for(int start_idx = tid*clusterSize; start_idx < grid_num; start_idx += blockDim.x*clusterSize){
        int part_num = grid_num-start_idx < clusterSize ? grid_num-start_idx : clusterSize;
        int cluster_idx = atomicAdd(cluster_num_d, 1);
        clusters_d[cluster_idx] = Cluster<clusterSize>(&grid_particle_array[grid_idx][start_idx], part_num);
        bb_d[cluster_idx] =  BoundBox(clusters_d[cluster_idx].particles, part_num);
    }

    // 0th thread free memory
    __syncthreads();
    if(tid == 0)
        free(grid_particle_array[grid_idx]);
}




/* construct an int2 array of neighbor cluster indices
 * template: clusterSize: number of particles of each cluster
 * @params:        bb_d: an array of bounding boxes of all clusters
 *          cluster_num: number of clusters or length of bb_d
 *        neighbor_list: caller-allocated sufficiently large int2 array to store neighbor cluster indices [output]
 *       neighbor_num_d: a int pointer to somewhere to store the number of neighbor cluster pair [output]
 *              cutoff2: square of cutoff distance
 *                bound: boundary of simulation box
 *
 * NOTE: 1. the maximum length of neighbor_list can be 0.5*(cluster_num)^2, allocate that number for neighbor_list for safety
 *       2. neighbor_list does not allow repetition(e.g. (1,2) but no (2,1)), but allow self-pair(e.g. (2,2))
 */
template<int clusterSize>
__global__ void build_cluster_pair_list_kernel(BoundBox *bb_d, int cluster_num, int2 *neighbor_list, int *neighbor_num_d, float cutoff2, float3 bound){
    int cluster_idx = blockIdx.x;
    int tid = threadIdx.x;

    for(int i = cluster_idx + tid; i < cluster_num; i += BLOCK_SIZE2){
        if(i == cluster_idx || dist2_BoundBox(bb_d[cluster_idx], bb_d[i], bound) < cutoff2){
            // critical section to push indices to neighbor_list
            int index = atomicAdd(neighbor_num_d, 1);
            neighbor_list[index] = make_int2(cluster_idx, i);
        }
    }
}

/* compute nonbonded interaction and verify neighbor cluster pair 
 * template: clusterSize: number of particles of each cluster
 * @params:             clusters: an array of constructed clusters containing particle positions 
 *                 neighbor_list: neighbor cluster indices array containing dummy pairs
 *         neighbor_list_compact: shortened neighbor cluster indices array without dummy pairs [output]
 *                neighbor_check: an array of counters of number of particles in cluster i that are neighbor of any particles in cluster j for pair(i,j), same length as neighbor_list
 *              pair_num_compact: number of valid neighbor cluster pair [output]
 *                       cutoff2: square of cutoff distance
 *                         bound: boundary of simulation box
 *                        forces: caller-allocated array of forces [output]
 *
 * NOTE: 1. calculate_force_clusters_kernel_trial is called for the first/only time after cluster and cluster neighbor pair list reconstruction
 *       2. pair_num_compact is slightly smaller than length of neighbor_list, allocate number of all neighbor pairs for neighbor_list_compact for safety
 */
template<int clusterSize>
__global__
void calculate_force_clusters_kernel_trial(Cluster<clusterSize> *clusters, int2 *neighbor_list, int2 *neighbor_list_compact, int *neighbor_check, int pair_num, int *pair_num_compact, float cutoff2, float3 bound, force_t *forces){
    int tid = threadIdx.x;
    int pair_id = blockIdx.x;
    int stride = gridDim.x;

    // shared memory for particles in neighbor cluster
    __shared__ Particle neighbor_particles[BLOCK_SIZE];

    while(pair_id < pair_num){
        int2 pair = neighbor_list[pair_id];
        Particle target_particle = clusters[pair.x].particles[tid];
        int target_index = target_particle.index;
        force_t force_acc = 0;

        // only used for first launch, counter of the number of neighbor particles for this thread(particle)
        char check = 0;

        // load neighbor particles to shared memory
        __syncthreads();
        neighbor_particles[tid] = clusters[pair.y].particles[tid];
        __syncthreads(); 

        // cluster self-pair
        if(pair.x == pair.y){
            for(int i = 0; i < clusterSize; i++){
                float dist2 = target_particle.dist2(neighbor_particles[i], bound);
                if(dist2 == INVALID) break;
                // compare to cutoff and accumulate force
                if(dist2 < cutoff2){
                    force_t force = LJP(dist2);
                    force_acc += force;
                    check++;
                }
            }
        }
        // cluster neighbor pair
        else{
            for(int i = 0; i < clusterSize; i++){
                float dist2 = target_particle.dist2(neighbor_particles[i], bound);
                if(dist2 == INVALID) break;
                // compare to cutoff and accumulate force
                if(dist2 < cutoff2){
                    force_t force = LJP(dist2);
                    force_acc += force;
                    atomicAdd(&forces[neighbor_particles[i].index], force);
                    check++;
                }
            }
        }

        // accumulate force for tid-th particle
        if(force_acc  && target_index!= -1) atomicAdd(&forces[target_index],  force_acc);

        // only for first launch, accumulate the number of used particles
        if(check) atomicAdd(&neighbor_check[pair_id], 1);
        __syncthreads();

        // push neighbor cluster pair to neighbor_list_compact if there is at least one pair of neighbor particles
        if(tid == 0 && neighbor_check[pair_id] > 0){
            int pair_id_compact = atomicAdd(pair_num_compact, 1);
            neighbor_list_compact[pair_id_compact] = neighbor_list[pair_id];
        }

        pair_id += stride;
    }
}

/* compute nonbonded interaction and verify neighbor cluster pair 
 * template: clusterSize: number of particles of each cluster
 * @params:             clusters: an array of constructed clusters containing particle positions 
 *                 neighbor_list: neighbor cluster indices array containing dummy pairs
 *         neighbor_list_compact: shortened neighbor cluster indices array without dummy pairs [output]
 *                neighbor_check: an array of counters of number of particles in cluster i that are neighbor of any particles in cluster j for pair(i,j), same length as neighbor_list
 *              pair_num_compact: number of valid neighbor cluster pair [output]
 *                       cutoff2: square of cutoff distance
 *                         bound: boundary of simulation box
 *                        forces: caller-allocated array of forces [output]
 *
 * NOTE: calculate_force_clusters_kernel is called after calculate_force_clusters_kernel_trial is already called once
 */
template<int clusterSize>
__global__
void calculate_force_clusters_kernel(Cluster<clusterSize> *clusters, int2 *neighbor_list, int pair_num, float cutoff2, float3 bound, force_t *forces){
    int tid = threadIdx.x;
    int pair_id = blockIdx.x;
    int stride = gridDim.x;

    // shared memory for particles in neighbor cluster
    __shared__ Particle neighbor_particles[BLOCK_SIZE];

    while(pair_id < pair_num){
        int2 pair = neighbor_list[pair_id];
        Particle target_particle = clusters[pair.x].particles[tid];
        int target_index = target_particle.index;
        force_t force_acc = 0;

        // load neighbor particles to shared memory
        __syncthreads();
        neighbor_particles[tid] = clusters[pair.y].particles[tid];
        __syncthreads(); 

        // cluster self-pair
        if(pair.x == pair.y){
            for(int i = 0; i < clusterSize; i++){
                float dist2 = target_particle.dist2(neighbor_particles[i], bound);
                if(dist2 == INVALID) break;
                // compare to cutoff and accumulate force
                if(dist2 < cutoff2){
                    force_acc += LJP(dist2);
                }
            }
        }
        // cluster neighbor pair
        else{
            for(int i = 0; i < clusterSize; i++){
                float dist2 = target_particle.dist2(neighbor_particles[i], bound);
                if(dist2 == INVALID) break;
                // compare to cutoff and accumulate force
                if(dist2 < cutoff2){
                    force_t force = LJP(dist2);
                    force_acc += force;
                    atomicAdd(&forces[neighbor_particles[i].index], force);
                }
            }
        }

        // accumulate force for tid-th particle
        if(force_acc && target_index != -1) atomicAdd(&forces[target_index],  force_acc);

        pair_id += stride;
    }
}