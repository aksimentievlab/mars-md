#include "Configuration.h"
#include "ComputeForce.h"

#include <cub/cub.cuh>
#include <thrust/transform.h>
#include <thrust/sort.h>

#include "ClusterCompute.h"
#include "ClusterCompute.cuh"

#ifndef gpuErrchk
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
#endif

#define gpuKernelCheck() {kernelCheckCL( __FILE__, __LINE__); }
inline void kernelCheckCL(const char* file, int line)
{
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::fprintf(stderr,"Error: %s in %s %d\n", cudaGetErrorString(err),file, line);
        assert(1==2);
    }
    //gpuErrchk(cudaDeviceSynchronize());
}

// CellDecomposition::CellDecomposition() {};

NonbondedClusterCompute::NonbondedClusterCompute(const Configuration& c, const int num_replicas, const float density_scale) :
    cutoff2((c.switchLen + c.switchStart) * (c.switchLen + c.switchStart)),
    pairlist2((c.switchLen + c.switchStart + c.pairlistDistance) * (c.switchLen + c.switchStart + c.pairlistDistance))
{
    const int& num = c.num;
    const BaseGrid& sys = *c.sys;
    const Matrix3& b = sys.getBasis();
    assert(b.isDiagonal());
    float volume = b.ex().x*b.ey().y*b.ez().z;
    float particle_density = (c.num/volume)*density_scale;

    // Ideally each cluster will have equal sizes along x,y,z
    pencil_width = pow((cluster_size/particle_density),0.33333333f);
    pencil_width = min(0.5*sqrt(cutoff2), pencil_width);
    
    x_min = sys.origin.x;
    y_min = sys.origin.y;
    z_min = sys.origin.z;

    dz = sys.basis.ez().z;
    pbc = make_float3(sys.basis.ex().x, sys.basis.ey().y, dz);

    x_num = ceil(sys.basis.ex().x / pencil_width);
    y_num = ceil(sys.basis.ey().y / pencil_width);
    num_pencils = x_num*y_num;

    gpuErrchk(cudaMalloc(&pencil_particle_num_d, sizeof(int)*num_pencils));
    // gpuErrchk(cudaMalloc(&pencil_num_cluster_d, sizeof(int)*num_pencils));
    gpuErrchk(cudaMalloc(&pencil_z_coords_d, sizeof(float)*num));
    gpuErrchk(cudaMemset(pencil_particle_num_d, 0, sizeof(int)*num_pencils));
    gpuErrchk(cudaMemset(pencil_z_coords_d, 0.0f, sizeof(float)*num));

    { // for reordering particles
	gpuErrchk(cudaMalloc(&mapped_pos_d, sizeof(Vector3)*num));
	gpuErrchk(cudaMalloc(&mapped_type_d, sizeof(int)*num));
	mapped_pos_tex = nullptr;
    }
    
    particle_idx = thrust::device_vector<int>(num,0);

    max_clusters = num_pencils+num/cluster_size+1;
    gpuErrchk(cudaMalloc(&clusters_d, sizeof(int2)*max_clusters));
    gpuErrchk(cudaMalloc(&bb_d, sizeof(BoundBox)*max_clusters));

    gpuErrchk(cudaMalloc(&cluster_num_d, sizeof(int)));
    gpuErrchk(cudaMalloc(&neighbor_num_d, sizeof(int)));
    neighbor_list_full = NULL;

    mask_d = NULL;

    const size_t stn = num;
    BitMask bm = BitMask( (stn*(stn-1))/2 );
    for (int i = 0; i < c.numExcludes; ++i) {
	size_t ai = c.excludes[i].ind1;
	size_t aj = c.excludes[i].ind2;
	if (ai < aj) {
	    size_t ex_idx = (aj*(aj-1)) + ai;
	    bm.set_mask(ex_idx,1);
	}
    }
    global_mask_d = bm.copy_to_cuda();
    
}

NonbondedClusterCompute::~NonbondedClusterCompute() {
    gpuErrchk(cudaFree(pencil_particle_num_d));
    gpuErrchk(cudaFree(pencil_z_coords_d));

    // TODO free other stuff
    if (mapped_pos_tex != nullptr) {
	gpuErrchk(cudaDestroyTextureObject(*mapped_pos_tex));
	delete mapped_pos_tex;
	mapped_pos_tex = nullptr;
    }
    /*
    for (std::size_t i = 0; i < gpuman.gpus.size(); ++i) {
	    gpuErrchk(cudaFree(numPairs_d[i]));
	    gpuErrchk(cudaDestroyTextureObject(pairLists_tex[i]));
	    gpuErrchk(cudaFree(pairLists_d[i]));
	    gpuErrchk(cudaDestroyTextureObject(pairTabPotType_tex[i]));
	    gpuErrchk(cudaFree(pairTabPotType_d[i]));
    }
    gpuErrchk(cudaDestroyTextureObject(neighbors_tex));
    gpuErrchk(cudaFree( CellNeighborsList));
    */
}

void NonbondedClusterCompute::decompose(const ComputeForce &compute) {
    const int num = compute.num;
    const std::vector<BaseGrid*>& sys_d = compute.sys_d;
    const BaseGrid& sys = *compute.sys;
    // const int num_rb_attached_particles = compute.num_rb_attached_particles;
    // const int n_particle_types = compute.numParts;
    const int num_replicas = compute.numReplicas;
    // const int* type_d = compute.type_d;
    // const int numExcludes = compute.numExcludes;
    // const Exclude* excludes_d = compute.excludes_d;
    // const int2* excludeMap_d = compute.excludeMap_d;

    const std::vector<cudaTextureObject_t>& pos_tex = compute.pos_tex;
    assert( num_replicas == 1);
    assert( sys.basis.isDiagonal() );
    
    
    // Bin particles into pencils, create z array, and count number of particles per bin
    gpuErrchk(cudaMemset(pencil_particle_num_d, 0, sizeof(int)*num_pencils));
    build_pencils_kernel<<<dim3(num/BLOCK_SIZE0+1,1,1), dim3(BLOCK_SIZE0,1,1)>>>
 	(pencil_particle_num_d, pos_tex[0], num, pencil_width, x_min, y_min, z_min, dz, x_num, y_num, pencil_z_coords_d);
    gpuKernelCheck();
    gpuman.sync();

    // Transform pencil_particle_num into offsets
    // thrust::device_ptr<int> pnum(pencil_particle_num_d);
    thrust::device_ptr<int> pnum = thrust::device_pointer_cast(pencil_particle_num_d);
    thrust::device_vector<int> poffset(num_pencils,0); // TODO move allocation to constructor?
    thrust::exclusive_scan(pnum, pnum + num_pencils, poffset.begin(), 0);

    // Sort particles according to pencil_z
    // thrust::device_ptr<float> z_d(pencil_z_coords_d); // TODO move allocation to constructor?
    thrust::device_ptr<float> z_d = thrust::device_pointer_cast(pencil_z_coords_d);
    // thrust::device_vector<int> particle_idx(num,0); // TODO move allocation to constructor?
    gpuman.sync();
    thrust::sequence(particle_idx.begin(), particle_idx.end());
    gpuman.sync();

    /* The following code (thrust or cub) causes initcheck errors:
========= Uninitialized __global__ memory read of size 4
=========     at 0x00003640 in void cub::RadixSortScanBinsKernel<cub::DeviceRadixSortPolicy<float, int, int>::Policy700, int>(int*, int)
=========     by thread (324,0,0) in block (0,0,0)
=========     Address 0x7fd711245f10
=========     Device Frame:void cub::RadixSortScanBinsKernel<cub::DeviceRadixSortPolicy<float, int, int>::Policy700, int>(int*, int) (void cub::RadixSortScanBinsKernel<cub::DeviceRadixSortPolicy<float, int, int>::Policy700, int>(int*, int) : 0x3640)
=========     Saved host backtrace up to driver entry point
=========     Host Frame:/lib64/libcuda.so.1 (cuLaunchKernel + 0x2b8) [0x2235d8]
=========     Host Frame:/data/server1/cmaffeo2/cuda-11.0/lib64/libcudart.so.11.0 [0xf62b]
=========     Host Frame:/data/server1/cmaffeo2/cuda-11.0/lib64/libcudart.so.11.0 (cudaLaunchKernel + 0x1c1) [0x4f5b1]
=========     Host Frame:../../src/arbd_dbg [0x735c4]
=========     Host Frame:../../src/arbd_dbg [0x72948]
=========     Host Frame:../../src/arbd_dbg [0x7297f]
    */

    {
	thrust::sort_by_key( z_d, z_d + num, particle_idx.begin() );
    }

    /*
    {
	// Determine temporary device storage requirements
	void     *d_temp_storage = NULL;
	size_t   temp_storage_bytes = 0;

	float* d_keys_in = pencil_z_coords_d;
	float* d_keys_out;
	// int* d_values_in = particle_idx.data().get();
	int* d_values_in;
	gpuErrchk(cudaMalloc(&d_values_in, sizeof(int)*num));
	gpuErrchk(cudaMemset(d_values_in, 0, sizeof(int)*num));
	int* d_values_out;
	gpuErrchk(cudaMalloc(&d_keys_out, sizeof(float)*num));
	gpuErrchk(cudaMalloc(&d_values_out, sizeof(int)*num));
	gpuErrchk(cudaMemset(d_values_out, 0, sizeof(int)*num));
	gpuErrchk(cudaMemset(d_keys_out, 0, sizeof(float)*num));

	gpuman.sync();
	cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
					d_keys_in, d_keys_out, d_values_in, d_values_out, num);
	// Allocate temporary storage
	cudaMalloc(&d_temp_storage, temp_storage_bytes);

	cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
					d_keys_in, d_keys_out, d_values_in, d_values_out, num);
    }
    return;
    */
    
    // Group particles into clusters
    int* poffset_d = thrust::raw_pointer_cast( poffset.data() );
    int* particle_idx_d = thrust::raw_pointer_cast( particle_idx.data() );
    gpuErrchk(cudaMemset(cluster_num_d, 0, sizeof(int)));

    assert(NUM_THREADS == cluster_size);
    build_clusters_kernel<<<dim3(num/BLOCK_SIZE0+1,1,1), dim3(BLOCK_SIZE0,1,1)>>>
	(cluster_size, clusters_d, bb_d, cluster_num_d, num_pencils, pencil_particle_num_d, poffset_d, particle_idx_d, pos_tex[0]);
    gpuKernelCheck();
    gpuman.sync();

    gpuErrchk(cudaMemcpyAsync(&cluster_num, cluster_num_d, sizeof(int), cudaMemcpyDeviceToHost));
    gpuman.sync();
    // {
    // 	// Sort clusters according to z-order

    // 	thrust::device_ptr<float> z_d = thrust::device_pointer_cast(pencil_z_coords_d);
    // 	    cluster_idx = thrust::device_vector<int>(cluster_num,0);
    // 	    gpuman.sync();
    // 	    thrust::sequence(cluster_idx.begin(), cluster_idx.end());
    // 	    gpuman.sync();

    // 	    thrust::sort_by_key( z_d, z_d + num, particle_idx.begin() );
	
	
    // 	// Map particles into order of cluster appearance
    // }

    // Sort clusters according to z-order
    {
	gpuErrchk(cudaMalloc(&cluster_zorder_idx_d, sizeof(int)*cluster_num));

	// First compute z-order of cluster
	calc_cluster_zorder_idx<<<dim3(num/BLOCK_SIZE0+1,1,1), dim3(BLOCK_SIZE0,1,1)>>>
	    (clusters_d, bb_d, cluster_num_d, sqrt(cutoff2), sys_d[0], cluster_zorder_idx_d);
	gpuKernelCheck();

	gpuman.sync(); // need cluster_num
	thrust::device_ptr<int> zorder_idx_p = thrust::device_pointer_cast(cluster_zorder_idx_d);
	thrust::device_ptr<int2> clusters_p = thrust::device_pointer_cast(clusters_d);
	cluster_order = thrust::device_vector<int>(cluster_num,0);	
	gpuman.sync();
	thrust::sequence(cluster_order.begin(), cluster_order.end());
	gpuman.sync();
	thrust::sort_by_key( zorder_idx_p, zorder_idx_p + cluster_num, clusters_p );
	
	// // Map particles into order of cluster appearance
	// int* sorted_cluster_idx_d = thrust::raw_pointer_cast( cluster_idx.data() );
    }

    { // Reorder particles according to pencil z order
	const int* type_d = compute.type_d;
	printf("Launching sort kernel\n");	       	    
	sort_particles_by_pencil_z<<<dim3(num/BLOCK_SIZE0+1,1,1), dim3(BLOCK_SIZE0,1,1)>>>
	    (num, particle_idx_d, pos_tex[0], type_d, mapped_pos_d, mapped_type_d); // TODO: put in seperate stream
	gpuKernelCheck();

	if (mapped_pos_tex != nullptr) {
	    gpuErrchk(cudaDestroyTextureObject(*mapped_pos_tex));
	    // delete mapped_pos_tex;
	    assert( mapped_pos_tex == nullptr );
	    mapped_pos_tex = nullptr;
	}

	//Han-Yi bind to the texture
	cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeLinear;
	resDesc.res.linear.devPtr = mapped_pos_d;
	resDesc.res.linear.desc.f = cudaChannelFormatKindFloat;
	resDesc.res.linear.desc.x = 32; // bits per channel
	resDesc.res.linear.desc.y = 32; // bits per channel
	resDesc.res.linear.desc.z = 32; // bits per channel
	resDesc.res.linear.desc.w = 32; // bits per channel
	resDesc.res.linear.sizeInBytes = num*sizeof(float4);
	    
	cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.readMode = cudaReadModeElementType;
	    
	// create texture object: we only have to do this once!
	if (mapped_pos_tex == nullptr)  mapped_pos_tex = new cudaTextureObject_t;
	cudaCreateTextureObject(mapped_pos_tex, &resDesc, &texDesc, NULL);

	gpuman.sync(); // need cluster_num and mapped_pos stuff
	gpuKernelCheck();
    }    

    
    if (neighbor_list_full != NULL) gpuErrchk(cudaFree(neighbor_list_full));
    gpuErrchk(cudaMalloc((void**)&neighbor_list_full, sizeof(int2)*(cluster_num*cluster_num>>1) ));
    gpuErrchk(cudaMemset(neighbor_num_d, 0, sizeof(int)));
    
    build_cluster_pair_list_kernel_A<<<dim3(cluster_num-1,1,1), dim3(BLOCK_SIZE0,1,1)>>>
	(cluster_num, neighbor_list_full, neighbor_num_d, sys_d[0], bb_d, pairlist2, pbc);
    gpuKernelCheck();

    gpuErrchk(cudaMemcpy(&neighbor_num, neighbor_num_d, sizeof(int), cudaMemcpyDeviceToHost));
    printf("%d non-self pairs of %d clusters\n", neighbor_num, cluster_num);

    size_t free_byte, total_byte;
    gpuErrchk( cudaMemGetInfo( &free_byte, &total_byte ) );
    printf("Memory: %d/%d (%d%%)\n", static_cast<int>(total_byte-free_byte), static_cast<int>(total_byte),
	   static_cast<int>((total_byte-free_byte)*100.f/total_byte));
    
    if (mask_d != NULL) {
	BitMask::remove_from_cuda(mask_d);
    }
    BitMask bm = BitMask( ((size_t) neighbor_num+cluster_num+1)*cluster_size*cluster_size);
    mask_d = bm.copy_to_cuda();
    mask_filled = false;

    gpuErrchk( cudaMemGetInfo( &free_byte, &total_byte ) );
    printf("Memory: %d/%d (%d%%)\n", static_cast<int>(total_byte-free_byte), static_cast<int>(total_byte),
	   static_cast<int>((total_byte-free_byte)*100.f/total_byte));

}

float NonbondedClusterCompute::computeTabulated(bool get_energy, const ComputeForce &compute) {
    // TODO implement get_energy

    std::vector<BaseGrid*> sys_d = compute.sys_d;
    // const int num_rb_attached_particles = compute.num_rb_attached_particles;
    // const int numGroupSites = compute.numGroupSites;
    // const int num_replicas = compute.numReplicas;
    const int n_particle_types = compute.numParts;

    std::vector<Vector3*> forceInternal_d = compute.forceInternal_d;
    // float* energies_d = compute.energies_d;
    
    // const std::vector<Vector3*>& pos_d = compute.pos_d;
    const std::vector<cudaTextureObject_t>& pos_tex = compute.pos_tex;

    std::vector<TabulatedPotential**> tablePot_d = compute.tablePot_d;
    const int* type_d = compute.type_d;

    // /* step 4.1: calculate force for every particles in every cluster */
    dim3 block_dim(NUM_THREADS, 1, 1);
    dim3 grid_dim(std::min(neighbor_num, 65535), 1, 1);
    
    // step 4.1.0: In the first launch of kernel, construct a compacted cluster pair list(neighbor_list_d) besides computing the interaction.
    // TODO fix following documentation and/or determine if such filtering improves performance
    //  * Some cluster pairs in neighbor_list_full have no neighbor particle pair because distance between bounding boxes is the lower bound of distance between particles. 
    //  * Detect if the cluster pair has at least one neighbor particle pair and if so, push the cluster pair to neighbor_list_d
    //  * Free neighbor_list_full, neighbor_check_d, neighbor_num_d, pair_num_d

    int* particle_idx_d = thrust::raw_pointer_cast( particle_idx.data() );

    if (!mask_filled) {
	calculate_force_clusters_build_excl_kernel<<<grid_dim, block_dim>>>
	    (forceInternal_d[0], sys_d[0], cutoff2, pairlist2, neighbor_num, neighbor_list_full,
	     cluster_num, clusters_d, global_mask_d, mask_d, n_particle_types, mapped_type_d, tablePot_d[0], *mapped_pos_tex);
	mask_filled = true;
    } else {
	// step 4.1.1: In subsequent launches of kernel, cudaMemcpy back to CPU the actual number of valid cluster pair(pair_num). Lauch the kernel with pair_num as the grid dimensio
	calculate_force_clusters_kernel<<<grid_dim, block_dim>>>
	    (forceInternal_d[0], sys_d[0], cutoff2, neighbor_num, neighbor_list_full,
	     cluster_num, clusters_d, mask_d, n_particle_types, mapped_type_d, tablePot_d[0], *mapped_pos_tex);
    }
    
    return 0.0f;
}
