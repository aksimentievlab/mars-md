#pragma once
//#include <limits>
#include <cmath>
#include <cassert>
#include "CudaUtil.cuh"
#include "TabulatedMethods.cuh"
#include "BitMask.h"


struct BoundBox{
public:
    // constructors
    __host__ __device__
    BoundBox(Vector3 *points, int num) {
	if(points == NULL || num < 1) return;
	float x_min = points[0].x; float x_max = x_min;
	float y_min = points[0].y; float y_max = y_min;
	float z_min = points[0].z; float z_max = z_min;
	// find bounding coordinates
	for(int i = 1; i < num; i++){
	    if(points[i].x < x_min) x_min = points[i].x;
	    if(points[i].x > x_max) x_max = points[i].x;
	    if(points[i].y < y_min) y_min = points[i].y;
	    if(points[i].y > y_max) y_max = points[i].y;
	    if(points[i].z < z_min) z_min = points[i].z;
	    if(points[i].z > z_max) z_max = points[i].z;
	}
	x0 = x_min; x1 = x_max; y0 = y_min; y1 = y_max; z0 = z_min; z1 = z_max;
    };
    __device__
    BoundBox(int *parts, int num, cudaTextureObject_t PosTex) {
	if(parts == NULL || num < 1 ) return;
	Vector3 pos = Vector3(tex1Dfetch<float4>(PosTex, parts[0]));
	float x_min = pos.x; float x_max = x_min;
	float y_min = pos.y; float y_max = y_min;
	float z_min = pos.z; float z_max = z_min;
	// find bounding coordinates
	for(int i = 1; i < num && parts[i] != -1; i++){
	    Vector3 pos = Vector3(tex1Dfetch<float4>(PosTex, parts[i]));
	    if(pos.x < x_min) x_min = pos.x;
	    if(pos.x > x_max) x_max = pos.x;
	    if(pos.y < y_min) y_min = pos.y;
	    if(pos.y > y_max) y_max = pos.y;
	    if(pos.z < z_min) z_min = pos.z;
	    if(pos.z > z_max) z_max = pos.z;
	}
	x0 = x_min; x1 = x_max; y0 = y_min; y1 = y_max; z0 = z_min; z1 = z_max;
    };

    __host__ __device__
    BoundBox() {
	x0 = y0 = z0 = INFINITY;
	x1 = y1 = z1 = -x0;
    };

    __host__ __device__ void add_point(const Vector3& pos) {
	if(pos.x < x0) x0 = pos.x;
	if(pos.x > x1) x1 = pos.x;
	if(pos.y < y0) y0 = pos.y;
	if(pos.y > y1) y1 = pos.y;
	if(pos.z < z0) z0 = pos.z;
	if(pos.z > z1) z1 = pos.z;
    }

    /* return square of distances between 2 bounding boxes
     * @params: other: a reference to another BoundBox
     *          bound: boundary of simulation box
     */
    __host__ __device__
    bool within(const float dist2, const BoundBox &b, const float3 &bound) const {
    // note: assume two bounding boxes do not intersect or one contains the other,
    // because bounding boxes are constructed on exclusive clusters
    // reference: https://gamedev.stackexchange.com/questions/154036/efficient-minimum-distance-between-two-axis-aligned-squares

// #ifdef DEBUG
	// if ((x1-x0 < 0.25*bound.x) || (y1-y0 < 0.25*bound.y) || (b.x1-b.x0 < 0.25*bound.x) || (b.y1-b.y0 < 0.25*bound.y)) {
	//     printf("x0,x1,y0,y1,boundx,boundy : %0.1f %0.1f %0.1f %0.1f %0.1f %0.1f\n",
	// 	   x0,x1,y0,y1,bound.x,bound.y);
	// }
	assert(x1-x0 < 0.25*bound.x);
	assert(y1-y0 < 0.25*bound.y);
	assert(b.x1-b.x0 < 0.25*bound.x);
	assert(b.y1-b.y0 < 0.25*bound.y);
// #endif

	// Assume BoundBox does not span more than 1/4 of system, except possibly along z
	int dx2 = (x1-b.x0)*(x1-b.x0);
	if ( dx2 > dist2 && dx2 < 0.25*bound.x*bound.x) return false;
	int dy2 = (y1-b.y0)*(y1-b.y0);
	if ( dy2 > dist2 && dy2 < 0.25*bound.y*bound.y) return false;

	return this->dist2(b, bound) < dist2;
    }
	

    /* return square of distances between 2 bounding boxes
     * @params: other: a reference to another BoundBox
     *          bound: boundary of simulation box
     */
    __host__ __device__
    float dist2(const BoundBox &other, const float3 &bound) const {
    // note: assume two bounding boxes do not intersect or one contains the other,
    // because bounding boxes are constructed on exclusive clusters
    // reference: https://gamedev.stackexchange.com/questions/154036/efficient-minimum-distance-between-two-axis-aligned-squares

	
    // fetch x,y,z coordinate range, e.g. b0_x0 is the smallest x for box 0
    float b0_x0 = x0, b0_x1 = x1, 
          b0_y0 = y0, b0_y1 = y1,
          b0_z0 = z0, b0_z1 = z1, 
          b1_x0 = other.x0, b1_x1 = other.x1, 
          b1_y0 = other.y0, b1_y1 = other.y1,
          b1_z0 = other.z0, b1_z1 = other.z1;

    // calculate 2*center of bounding box
    float center0_x = (b0_x0 + b0_x1);
    float center0_y = (b0_y0 + b0_y1);
    float center0_z = (b0_z0 + b0_z1);
    float center1_x = (b1_x0 + b1_x1);
    float center1_y = (b1_y0 + b1_y1);
    float center1_z = (b1_z0 + b1_z1);

    // TODO check that this catches all the corners! I think it probably doesn't
    // TODO optimize

    
    // calculate distance difference in x, y, z. If the range of coordinates overlaps in any direction, set distance to 0
    float delta;
    if(center1_x > center0_x){
        delta = (b1_x0 - b0_x1) > 0.0f ? (b1_x0 - b0_x1) : 0.0f;
        if(delta*2 > bound.x) delta = pbc_dist(b0_x0, b1_x1, bound.x);
    }
    else{
        delta = (b0_x0 - b1_x1) > 0.0f ? (b0_x0 - b1_x1) : 0.0f;
        if(delta*2 > bound.x) delta = pbc_dist(b0_x1, b1_x0, bound.x);
    }
    float dist2 = delta*delta;
    if(center1_y > center0_y){
        delta = (b1_y0 - b0_y1) > 0.0f ? (b1_y0 - b0_y1) : 0.0f;
        if(delta*2 > bound.y) delta = pbc_dist(b0_y0, b1_y1, bound.y);
    }
    else{
        delta = (b0_y0 - b1_y1) > 0.0f ? (b0_y0 - b1_y1) : 0.0f;
        if(delta*2 > bound.y) delta = pbc_dist(b0_y1, b1_y0, bound.y);
    }
    dist2 += delta*delta;    
    if(center1_z > center0_z){
        delta = (b1_z0 - b0_z1) > 0.0f ? (b1_z0 - b0_z1) : 0.0f;
        if(delta*2 > bound.z) delta = pbc_dist(b0_z0, b1_z1, bound.z);
    }
    else{
        delta = (b0_z0 - b1_z1) > 0.0f ? (b0_z0 - b1_z1) : 0.0f;
        if(delta*2 > bound.z) delta = pbc_dist(b0_z1, b1_z0, bound.z);
    }
    dist2 += delta*delta;    
    return dist2;
}

    __host__ __device__
    inline float pbc_dist(const float &a, const float &b, const float &bound) const{
	float delta = a>b ? a-b : b-a;
	if(delta*2 > bound) delta = bound - delta;
	return delta;
    }

public:
    // x0, x1, y0, y1, z0, z1 are the ranges of x, y, z coordinates
    // for example the bottom front left corner is (x0, y0, z0)
    float x0, x1, y0, y1, z0, z1;
};

__global__ void build_pencils_kernel(int *pencil_particle_num, cudaTextureObject_t PosTex, int num, float pencil_width, float x_min, float y_min, float z_min, float dz, int x_num, int y_num, float* pencil_z_coords) {
    int tid = threadIdx.x;
    
    // Count particles in each xy bin
    for(int i = tid+blockDim.x*blockIdx.x; i < num; i += blockDim.x*gridDim.x) {
	Vector3 pos = Vector3(tex1Dfetch<float4>(PosTex, i));
        // determine pencil index
	int x_idx = (pos.x - x_min) / pencil_width;
        if (x_idx >= x_num) x_idx = x_num-1;
	int y_idx = (pos.y - y_min) / pencil_width;
        if (y_idx >= y_num) y_idx = y_num-1;
	int pencil_idx = x_idx*y_num + y_idx;
        atomicAdd( &pencil_particle_num[pencil_idx], 1 );
	float z = (pos.z-z_min);
	z = z + dz*(pencil_idx - 0.5*x_num*y_num);
	pencil_z_coords[i] = z;
    }
}


/* cluster particles and build bounding boxes of clusters
 * template: cluster_size: number of particles of each cluster
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
 * NOTE: 1. number of clusters is slightly larger than len/cluster_size, allocate 2 * (len/cluster_size) elements for clusters_d and bb_d for safety
 *       2. build_clusters_kernel should be called immediately after particle_histogram_kernel
 */
__global__ void build_clusters_kernel(int cluster_size, int2* clusters_d, BoundBox* bb_d, int *cluster_num, int num_pencils, int *pencil_particle_num, int* pencil_offset, int* pencil_particle_idx, cudaTextureObject_t PosTex) {
    
    // One block per pencil
    const int pencil_idx = blockIdx.x;
    if (pencil_idx < num_pencils) {
	const int tid = threadIdx.x;
	const int particle_num =  pencil_particle_num[pencil_idx];
	const size_t pidx_start = pencil_offset[pencil_idx];

	// Thread adds cluster of particles
	for(int start_idx = tid*cluster_size; start_idx < particle_num; start_idx += cluster_size*blockDim.x) {
	    int part_num = particle_num-start_idx < cluster_size ? particle_num-start_idx : cluster_size; // particles in cluster
	    int cluster_idx = atomicAdd(cluster_num, 1);
	    clusters_d[cluster_idx] = make_int2(pidx_start+start_idx, part_num);
	    BoundBox& bb = bb_d[cluster_idx] = BoundBox();
	    for (int j = 0; j < part_num; ++j) {
		const int ai = pencil_particle_idx[ pidx_start+start_idx+j ];
		Vector3 pos = Vector3(tex1Dfetch<float4>(PosTex, ai));
		bb.add_point( pos );
	    }
	}
    }
}

__global__ void calc_cluster_zorder_idx(int2* clusters_d, BoundBox* bb_d, int *cluster_num, float cutoff, const BaseGrid* __restrict__ sys, int* zorder_idx ) {

    // determine bits for each axis
    int nx,ny,nz;
    float dx,dy,dz;
    Vector3 o;
    {
	Matrix3 B = sys->getBasis();
	dx = B.ex().x; dy = B.ey().y; dz = B.ez().z; // Only for rectilinear unit cells
	
	nx = min(static_cast<int>(floor(dx*0.1/cutoff)),1);
	ny = min(static_cast<int>(floor(dy*0.1/cutoff)),1);
	nz = min(static_cast<int>(floor(dz*0.1/cutoff)),1);

	
	dx = dx/nx; 		// distance between adjacent indices in z-order curve
	dy = dy/ny; 		// distance between adjacent indices in z-order curve
	dz = dz/nz; 		// distance between adjacent indices in z-order curve
	o = sys->getOrigin();
    }

    size_t cluster_idx = threadIdx.x + blockIdx.x*blockDim.x;
    while (cluster_idx < *cluster_num) {
	BoundBox& bb = bb_d[cluster_idx];
	int x = (bb.x0 - o.x) / dx;
	int y = (bb.y0 - o.y) / dy;
	int z = (bb.z0 - o.z) / dz;

	constexpr int max_n = 0x00000001 << 10;
	assert( x < max_n );
	assert( y < max_n );
	assert( z < max_n );

	// https://stackoverflow.com/questions/1024754/how-to-compute-a-3d-morton-number-interleave-the-bits-of-3-ints
	x = (x | (x << 16)) & 0x030000FF;
	x = (x | (x <<  8)) & 0x0300F00F;
	x = (x | (x <<  4)) & 0x030C30C3;
	x = (x | (x <<  2)) & 0x09249249;

	y = (y | (y << 16)) & 0x030000FF;
	y = (y | (y <<  8)) & 0x0300F00F;
	y = (y | (y <<  4)) & 0x030C30C3;
	y = (y | (y <<  2)) & 0x09249249;

	z = (z | (z << 16)) & 0x030000FF;
	z = (z | (z <<  8)) & 0x0300F00F;
	z = (z | (z <<  4)) & 0x030C30C3;
	z = (z | (z <<  2)) & 0x09249249;

	zorder_idx[cluster_idx] = x | (y << 1) | (z << 2);
	cluster_idx += *cluster_num;
    }
}

// __global__ void reorder_particles(int cluster_size, int2* clusters_d, BoundBox* bb_d, int *cluster_num, int num_pencils, int *pencil_particle_num, int* pencil_offset, int* pencil_particle_idx, cudaTextureObject_t PosTex) {


//     size_t cluster_list_idx = threadIdx.x + blockIdx.x*gridDim.x;
//     while (cluster_list_idx < cluster_num) {
// 	size_t cluster_idx = zorder_cluster_ids[cluster_list_idx];
	

    
// 	// One block per pencil
// 	const int pencil_idx = blockIdx.x;
// 	if (pencil_idx < num_pencils) {
// 	    const int tid = threadIdx.x;
// 	    const int particle_num =  pencil_particle_num[pencil_idx];
// 	    const size_t pidx_start = pencil_offset[pencil_idx];

// 	    // Thread adds cluster of particles to 
// 	    for(int start_idx = tid*cluster_size; start_idx < particle_num; start_idx += cluster_size*blockDim.x) {
// 		int part_num = particle_num-start_idx < cluster_size ? particle_num-start_idx : cluster_size; // particles in cluster
// 		int cluster_idx = atomicAdd(cluster_num, 1);
// 		clusters_d[cluster_idx] = make_int2(pidx_start+start_idx, part_num);
// 		BoundBox& bb = bb_d[cluster_idx] = BoundBox();
// 		for (int j = 0; j < part_num; ++j) {
// 		    const int ai = pencil_particle_idx[ pidx_start+start_idx+j ];
// 		    Vector3 pos = Vector3(tex1Dfetch<float4>(PosTex, ai));
// 		    bb.add_point( pos );
// 		}
// 	    }
// 	}
//     }
// }

// NOTE THIS IS NOT CURRENTLY USABLE DUE TO MISSING MAPPING; JUST FOR TESTING
__global__ void sort_particles_by_pencil_z(int particle_num, int* __restrict__ pencil_particle_idx,
					   cudaTextureObject_t PosTex, const int* __restrict__ type,
					   Vector3* new_pos, int* new_type) {

    size_t i = threadIdx.x + blockIdx.x*blockDim.x;
    while (i < particle_num) {
	int j = pencil_particle_idx[i];
	assert( j >= 0 );
	new_pos[i] = Vector3(tex1Dfetch<float4>(PosTex, j));
	new_type[i] = type[j];
	i += blockDim.x*gridDim.x;
    }
}


/* construct an int2 array of neighbor cluster indices
 * template: cluster_size: number of particles of each cluster
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
__global__ void build_cluster_pair_list_kernel_A(const int cluster_num, int2 *neighbor_list, int *neighbor_num_d, const BaseGrid* __restrict__ sys, BoundBox *bb_d, float pl_dist2, float3 bound) {
    int cluster_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    for (int i = cluster_idx+1 + tid; i < cluster_num; i += blockDim.x) {
        // if (dist2_BoundBox(bb_d[cluster_idx], bb_d[i], bound) < pl_dist2) {}
	// printf("build_cluster_pairs: clusters %d,%d distance: %f\n",
	//      cluster_idx,i, bb_d[cluster_idx].dist2(bb_d[i],bound));
	if ( bb_d[cluster_idx].within(pl_dist2, bb_d[i], bound) ) {
            // critical section to push indices to neighbor_list
	    assert(cluster_idx != i);
            int index = atomicAdd(neighbor_num_d, 1);
            neighbor_list[index] = make_int2(cluster_idx, i);
        }
    }
}
// __global__ void build_cluster_pair_list_kernel_B(const int cluster_num, int2 *neighbor_list, int *neighbor_num_d) {
//     int tid = threadIdx.x;

//     // Place all "self" interactions together
//     for (int i = tid+blockIdx.x*blockDim.x; i < cluster_num; i += blockDim.x*gridDim.x) {
// 	int index = atomicAdd(neighbor_num_d, 1);
// 	neighbor_list[index] = make_int2(i, i);
//     }
// }


__global__
void calculate_force_clusters_build_excl_kernel(
    Vector3* force, const BaseGrid* __restrict__ sys, const float cutoff2, const float pairlist_dist2,
    const int num_cluster_pairs, int2 *neighbor_list,
    const int num_clusters, int2* clusters,
    const BitMask* __restrict__ global_mask,
    BitMask* __restrict__ mask,
    const int numParts, const int* __restrict__ type,
    TabulatedPotential** __restrict__ tablePot,
    const cudaTextureObject_t PosTex)
// Vector3* __restrict__ pos)
{
    // , float cutoff2, force_t *forces){}
    // TODO exclusions!

    // TODO assert cluster_size <= block_size

    int tid = threadIdx.x;
    int pair_id = blockIdx.x;
    const int& stride = gridDim.x;

    constexpr int block_size = NonbondedClusterCompute::NUM_THREADS;
    constexpr int cluster_size = block_size;
    //const int block_size = 32;
    // shared memory for particles in neighbor cluster
    __shared__ int particle_j[block_size*(2+2*sizeof(Vector3)/sizeof(int))];
    // __shared__ int particle_j[6*block_size];
    int* type_j = &particle_j[block_size];
    Vector3* pos_j =  (Vector3*) &particle_j[2*block_size];
    Vector3* force_j =  (Vector3*) &particle_j[(2+sizeof(Vector3)/sizeof(int))*block_size];

    // __shared__ int particle_j[block_size];
    // __shared__ Vector3 pos_j[block_size];
    // __shared__ Vector3 force_j[block_size];
    // __shared__ int type_j[block_size];

    while (pair_id < num_cluster_pairs+num_clusters) {

	int cidx_i, cidx_j;
	if (pair_id < num_cluster_pairs) {
	    assert( neighbor_list[pair_id].x <= neighbor_list[pair_id].y ); // TODO remove this
	    cidx_i = neighbor_list[pair_id].x;
	    cidx_j = neighbor_list[pair_id].y;
	} else {
	    cidx_i = cidx_j = pair_id - num_cluster_pairs;
	}
	int2& cluster_i = clusters[cidx_i];
	int2& cluster_j = clusters[cidx_j];

	const int ai = (tid < cluster_i.y) ? cluster_i.x+tid : -1;

        Vector3 force_acc = Vector3(0.0f);

	// load neighbor particles to shared memory
        __syncthreads();
	particle_j[tid] = (tid < cluster_j.y) ? cluster_j.x+tid : -1;
	if (particle_j[tid] >= 0) {
	    pos_j[tid] = Vector3(tex1Dfetch<float4>(PosTex, particle_j[tid]));
	    // pos_j[tid] = pos[particle_j[tid]];
	    force_j[tid] = Vector3(0.0f);
	    type_j[tid] = type[particle_j[tid]];
	}
        __syncthreads();

	if (ai >= 0) {
	    const int& i = tid;
	    if (pair_id >= num_cluster_pairs) {
		// cluster self-pair
		Vector3& a = pos_j[i];
		for (int j = 0; j < cluster_j.y; ++j) {
		    // int& aj = particle_j[j];
		    // size_t ajt = aj;
		    // const size_t g_ex_idx = ajt*(ajt-1)/2 + ai;
		    const int aj = particle_j[j];
		    const size_t ajt = aj;
		    const size_t g_ex_idx = (ajt*(ajt-1))/2 + ai;
		    if (aj < 0 || aj <= ai || global_mask->get_mask(g_ex_idx)) {
		        mask->set_mask( ((size_t) pair_id)*cluster_size*cluster_size +
				      j*cluster_size + i, true);
			continue;
		    }

		    Vector3& b = pos_j[j];
		    Vector3 dr = sys->wrapDiff(b-a);
		    float d2 = dr.length2();

		    if (d2 > pairlist_dist2) {
		        mask->set_mask( ((size_t) pair_id)*cluster_size*cluster_size +
				      j*cluster_size + i, true);
			continue;
		    }

		    int ind = type[ai] + type_j[j] * numParts;
		    if (tablePot[ind] != NULL && d2 <= cutoff2) {
			Vector3 f = tablePot[ind]->computef(dr,d2);
			force_acc += f;
			atomicAdd( &(force_j[j]), -f );
		    }
		}
	    } else {
		// cluster neighbor pair
		Vector3 a(tex1Dfetch<float4>(PosTex, ai));
		// Vector3 a(pos[ai]);
		for (int j = 0; j < cluster_j.y; ++j) {
		    // int& aj = particle_j[j];
		    const int aj = particle_j[j];
		    const size_t ajt = static_cast<size_t>(aj);
		    const size_t g_ex_idx = ((ajt*(ajt-1)) >> 1) + ai;
		    // printf("excl %d.%d: %d %d %lld\n", tid, blockIdx.x, ai, aj, g_ex_idx);

		    if (aj < 0 || global_mask->get_mask(g_ex_idx)) {
			global_mask->get_mask(g_ex_idx);
		        mask->set_mask( ((size_t) pair_id)*cluster_size*cluster_size +
				      j*cluster_size + i, true);
			continue;
		    }
		    global_mask->get_mask(g_ex_idx);


		    Vector3& b = pos_j[j];
		    Vector3 dr = sys->wrapDiff(b-a);
		    float d2 = dr.length2();

		    if (d2 > pairlist_dist2) {
		        mask->set_mask( ((size_t) pair_id)*cluster_size*cluster_size +
				      j*cluster_size + i, true);
			continue;
		    }

		    // TODO: maybe cache pairTabPotType for clusters
		    int ind = type[ai] + type_j[j] * numParts;


		    if (tablePot[ind] != NULL && d2 <= cutoff2) {
			Vector3 f = tablePot[ind]->computef(dr,d2);
			force_acc += f;
			atomicAdd( &(force_j[j]), -f );
		    }
		}
	    }
	    // accumulate force for tid-th particle
	    // printf("force[ai(%d)] = %f %f %f\n", ai, force_acc.x, force_acc.y, force_acc.z);

	    // atomicAdd(&force[ai],  force_acc);
	}
	// accumulate force for j-th particle
	//
	for (int j = tid; j < cluster_j.y; ++j) {
	    const int& aj = particle_j[j];
	    if (aj >= 0) {
		// atomicAdd(&force[aj],  force_j[j]);
		// printf("force[aj(%d)] = %f %f %f\n", aj, force_j[j].x, force_j[j].y, force_j[j].z);
	    }
	}
        pair_id += stride;
    }
}


/* compute nonbonded interaction and verify neighbor cluster pair 
 * template: cluster_size: number of particles of each cluster
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
__global__
void calculate_force_clusters_kernel(Vector3* force, const BaseGrid* __restrict__ sys, const float cutoff2,
				     const int num_cluster_pairs, int2* __restrict__ neighbor_list,
				     const int num_clusters, int2* __restrict__ clusters,
				     const BitMask* __restrict__ mask,
				     const int numParts, const int* __restrict__ type,
				     TabulatedPotential* __restrict__ * __restrict__ tablePot,
				     const cudaTextureObject_t PosTex)
				     // Vector3* __restrict__ pos)
{
    // , float cutoff2, force_t *forces){}
    // TODO exclusions!

    // TODO assert cluster_size <= block_size
    
    int tid = threadIdx.x;
    int pair_id = blockIdx.x;
    const int& stride = gridDim.x;

    constexpr int block_size = NonbondedClusterCompute::NUM_THREADS;
    constexpr int cluster_size = block_size;
    //const int block_size = 32;
    // shared memory for particles in neighbor cluster
    __shared__ int particle_j[block_size*(2+2*sizeof(Vector3)/sizeof(int))];
    // __shared__ int particle_j[6*block_size];
    int* __restrict__ type_j = &particle_j[block_size];
    Vector3* __restrict__ pos_j =  (Vector3*) &particle_j[2*block_size];
    Vector3* __restrict__ force_j =  (Vector3*) &particle_j[(2+sizeof(Vector3)/sizeof(int))*block_size];

    // __shared__ int particle_j[block_size];
    // __shared__ Vector3 pos_j[block_size];
    // __shared__ Vector3 force_j[block_size];
    // __shared__ int type_j[block_size];
    
    // while (pair_id < num_cluster_pairs+num_clusters) {
    while (pair_id < num_cluster_pairs) {
	int cidx_i, cidx_j;
	if (pair_id < num_cluster_pairs) {
	    // assert( neighbor_list[pair_id].x <= neighbor_list[pair_id].y ); // TODO remove this
	    cidx_i = neighbor_list[pair_id].x;
	    cidx_j = neighbor_list[pair_id].y;
	} else {
	    cidx_i = cidx_j = pair_id - num_cluster_pairs;
	}
	int2& cluster_i = clusters[cidx_i];
	int2& cluster_j = clusters[cidx_j];
	// assert( block_size >= cluster_i.y );
	// assert( block_size >= cluster_j.y );
	const int ai = (tid < cluster_i.y) ? cluster_i.x+tid : -1;
	
	// int2 cluster_j = make_int2(0,0);
	// const int ai = 0;
        Vector3 force_acc = Vector3(0.0f);

	// load neighbor particles to shared memory (bottleneck)
        __syncthreads();
	const int& pj = particle_j[tid] = (tid < cluster_j.y) ? cluster_j.x+tid : -1;
	// const int& pj = particle_j[tid] = tid + pair_i;
	if (pj >= 0) {
	    pos_j[tid] = Vector3(tex1Dfetch<float4>(PosTex, pj));
	    // pos_j[tid] = pos[particle_j[tid]];
	    force_j[tid] = Vector3(0.0f);
	    type_j[tid] = type[pj] * numParts;
	}
        __syncthreads();


	if (ai >= 0) {
	    const int& i = tid;
	    
	    if (pair_id >= num_cluster_pairs) {
		// cluster self-pair
		Vector3& a = pos_j[i];
		const int type_a = type[ai];
		for (int j = 0; j < cluster_j.y; ++j) {
		    // const size_t ex_idx = ((size_t) pair_id)*cluster_size*cluster_size + j*cluster_size + i;
		    // if (mask->get_mask( ex_idx )) {
		    // 	continue;
		    // }

		    Vector3& b = pos_j[j];
		    Vector3 dr = sys->wrapDiff(b-a);
		    float d2 = dr.length2();

		    int ind = type_a + type_j[j];
		    if (tablePot[ind] != NULL && d2 <= cutoff2) {
			Vector3 f = tablePot[ind]->computef(dr,d2);
			force_acc += f;
			atomicAdd( &(force_j[j]), -f );
		    }
		}
	    } else {
		// cluster neighbor pair
		Vector3 a(tex1Dfetch<float4>(PosTex, ai));
		const int type_a = type[ai];
		for (int j = 0; j < cluster_j.y; ++j) {
		    const size_t ex_idx = ((size_t) pair_id)*cluster_size*cluster_size + j*cluster_size + i;
		    if (mask->get_mask( ex_idx )) {
			continue;
		    }

		    Vector3& b = pos_j[j];
		    Vector3 dr = sys->wrapDiff(b-a);
		    float d2 = dr.length2();

		    // TODO: maybe cache pairTabPotType for clusters
		    // int ind = type[ai] + type_j[j] * numParts;
		    int ind = type_a + type_j[j];

		    if (tablePot[ind] != NULL && d2 <= cutoff2) {
			Vector3 f = tablePot[ind]->computef(dr,d2);
			force_acc += f;
			atomicAdd( &(force_j[j]), -f );
		    }
		}
	    }
	    // accumulate force for tid-th particle
	    atomicAdd(&force[ai],  force_acc);
	}
	// accumulate force for j-th particle
	//
	// for (int j = tid; j < cluster_j.y; ++j)
	{
	    const int& j = tid;
	    const int& aj = particle_j[j];
	    if (aj >= 0) atomicAdd(&force[aj],  force_j[j]);
	}
        pair_id += stride;
    }
}
