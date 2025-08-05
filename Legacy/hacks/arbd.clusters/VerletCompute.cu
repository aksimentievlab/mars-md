#include "Configuration.h"
#include "ComputeForce.h"
#include "VerletCompute.h"
#include "VerletCompute.cuh"

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

NonbondedVerletCompute::NonbondedVerletCompute(const Configuration& c, const int num_replicas) :
    cutoff2((c.switchLen + c.switchStart) * (c.switchLen + c.switchStart)),
    decomp(c.sys->getBox(), c.sys->getOrigin(), c.switchStart + c.switchLen + c.pairlistDistance, num_replicas)
{

    // Grow vectors for per-gpu device pointers
    for (int i = 0; i < gpuman.gpus.size(); ++i) {
	int s = gpuman.gpus.size();
	pairLists_d = std::vector<int2*>(s);
	pairLists_tex = std::vector<cudaTextureObject_t>(s);
	pairTabPotType_d = std::vector<int*>(s);
	pairTabPotType_tex = std::vector<cudaTextureObject_t>(s);
	numPairs_d = std::vector<int*>(s);
    }

    // Allocate device for pairlists
    // RBTODO: select maxpairs in better way; add assertion in kernel to avoid going past this
    const int maxPairs = 1<<25;
    for (std::size_t i = 0; i < gpuman.gpus.size(); ++i) {
	gpuman.use(i);
	gpuErrchk(cudaMalloc(&numPairs_d[i],       sizeof(int)));
	gpuErrchk(cudaMalloc(&pairLists_d[i],      sizeof(int2)*maxPairs));
	// gpuErrchk(cudaBindTexture(0, pairListsTex, pairLists_d[i], sizeof(int2)*maxPairs)); //Han-Yi
	gpuErrchk(cudaMalloc(&pairTabPotType_d[i], sizeof(int)*maxPairs));
    }

    // create texture object
    for (std::size_t i = 0; i < gpuman.gpus.size(); ++i) {
	gpuman.use(i);
	cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeLinear;
	resDesc.res.linear.devPtr = pairLists_d[i];
	resDesc.res.linear.desc.f = cudaChannelFormatKindSigned;
	resDesc.res.linear.desc.x = 32; // bits per channel
	resDesc.res.linear.desc.y = 32; // bits per channel
	resDesc.res.linear.sizeInBytes = maxPairs*sizeof(int2);

	cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.readMode = cudaReadModeElementType;

	// create texture object: we only have to do this once!
	pairLists_tex[i]=0;
	cudaCreateTextureObject(&pairLists_tex[i], &resDesc, &texDesc, NULL);
    }

    // create texture object
    for (std::size_t i = 0; i < gpuman.gpus.size(); ++i) {
	gpuman.use(i);
	cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeLinear;
	resDesc.res.linear.devPtr = pairTabPotType_d[i];
	resDesc.res.linear.desc.f = cudaChannelFormatKindSigned;
	resDesc.res.linear.desc.x = 32; // bits per channel
	resDesc.res.linear.sizeInBytes = maxPairs*sizeof(int);

	cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.readMode = cudaReadModeElementType;

	// create texture object: we only have to do this once!
	pairTabPotType_tex[i] = 0;
	cudaCreateTextureObject(&pairTabPotType_tex[i], &resDesc, &texDesc, NULL);

    }
    gpuman.use(0);

    //Han-Yi Chou
    int nCells = decomp.nCells.x * decomp.nCells.y * decomp.nCells.z;
    decomp_d = NULL;

    //int* nCells_dev;
    if (nCells < MAX_CELLS_FOR_CELLNEIGHBORLIST) {
	int3 *Cells_dev;
	size_t sz = 27*nCells*sizeof(int);
	gpuErrchk(cudaMalloc(&CellNeighborsList, sz));
	//gpuErrchk(cudaMalloc(&nCells_dev,sizeof(int)));
	gpuErrchk(cudaMalloc(&Cells_dev,sizeof(int3)));
	//gpuErrchk(cudaMemcpy(nCells_dev,&nCells,1,cudaMemcpyHostToDevice);
	gpuErrchk(cudaMemcpy(Cells_dev,&(decomp.nCells),sizeof(int3),cudaMemcpyHostToDevice));
	createNeighborsList<<<256,256>>>(Cells_dev,CellNeighborsList);
	gpuErrchk(cudaFree(Cells_dev));

	// create texture object
	{
	    cudaResourceDesc resDesc;
	    memset(&resDesc, 0, sizeof(resDesc));
	    resDesc.resType = cudaResourceTypeLinear;
	    resDesc.res.linear.devPtr = CellNeighborsList;
	    resDesc.res.linear.desc.f = cudaChannelFormatKindSigned;
	    resDesc.res.linear.desc.x = 32; // bits per channel
	    resDesc.res.linear.sizeInBytes = sz;

	    cudaTextureDesc texDesc;
	    memset(&texDesc, 0, sizeof(texDesc));
	    texDesc.readMode = cudaReadModeElementType;

	    // create texture object: we only have to do this once!
	    neighbors_tex=0;
	    cudaCreateTextureObject(&neighbors_tex, &resDesc, &texDesc, NULL);
	}
    }

    pairlistdist2 = (sqrt(cutoff2) + c.pairlistDistance);
    pairlistdist2 *= pairlistdist2;

}
NonbondedVerletCompute::~NonbondedVerletCompute() {
    for (std::size_t i = 0; i < gpuman.gpus.size(); ++i) {
	    gpuErrchk(cudaFree(numPairs_d[i]));
	    gpuErrchk(cudaDestroyTextureObject(pairLists_tex[i]));
	    gpuErrchk(cudaFree(pairLists_d[i]));
	    gpuErrchk(cudaDestroyTextureObject(pairTabPotType_tex[i]));
	    gpuErrchk(cudaFree(pairTabPotType_d[i]));
    }
    gpuErrchk(cudaDestroyTextureObject(neighbors_tex));
    gpuErrchk(cudaFree( CellNeighborsList));
}

void NonbondedVerletCompute::decompose(const ComputeForce &compute) {
    const size_t num = compute.num;
    const size_t num_rb_attached_particles = compute.num_rb_attached_particles;
    const std::vector<BaseGrid*>& sys_d = compute.sys_d;
    const size_t n_particle_types = compute.numParts;
    const size_t num_replicas = compute.numReplicas;
    const int* type_d = compute.type_d;
    const int numExcludes = compute.numExcludes;
    const Exclude* excludes_d = compute.excludes_d;
    const int2* excludeMap_d = compute.excludeMap_d;

    const std::vector<Vector3*>& pos_d = compute.pos_d;
    const std::vector<cudaTextureObject_t>& pos_tex = compute.pos_tex;

    // Reset the cell decomposition.
    if (decomp_d != NULL)
    {
	cudaFree(decomp_d);
	decomp_d = NULL;
    }
    decomp.decompose_d(pos_d[0], num + num_rb_attached_particles);
    decomp_d = decomp.copyToCUDA();

	// Update pairlists using cell decomposition (not sure this is really needed or good)
	//RBTODO updatePairlists<<< nBlocks, NUM_THREADS >>>(pos_d[0], num, numReplicas, sys_d[0], decomp_d);

	/* size_t free, total; */
	/* { */
	/* 	cuMemGetInfo(&free,&total); */
	/* 	printf("Free memory: %zu / %zu\n", free, total); */
	/* } */

	// initializePairlistArrays
	int nCells = decomp.nCells.x * decomp.nCells.y * decomp.nCells.z;

	/* cuMemGetInfo(&free,&total); */
	/* printf("Free memory: %zu / %zu\n", free, total); */

	int tmp = 0;
	gpuErrchk(cudaMemcpyAsync(numPairs_d[0], &tmp,	sizeof(int), cudaMemcpyHostToDevice));
	gpuErrchk(cudaDeviceSynchronize());

#ifdef DEBUGEXCLUSIONS
	initExSum();
	gpuErrchk(cudaDeviceSynchronize()); /* RBTODO: sync needed here? */
#endif

      #if __CUDA_ARCH__ >= 520
      createPairlists<64,64,8><<<dim3(128,128,num_replicas),dim3(64,1,1)>>>(compute.pos_d[0], num+num_rb_attached_particles, num_replicas, sys_d[0], decomp_d, nCells, numPairs_d[0],
                                                                             pairLists_d[0], n_particle_types, type_d, pairTabPotType_d[0], excludes_d,
									   excludeMap_d, numExcludes, pairlistdist2, pos_tex[0], neighbors_tex);
      #else //__CUDA_ARCH__ == 300
      createPairlists<64,64,8><<<dim3(256,256,num_replicas),dim3(64,1,1)>>>(pos_d[0], num+num_rb_attached_particles, num_replicas, sys_d[0], decomp_d, nCells, numPairs_d[0],
                                                                           pairLists_d[0], n_particle_types, type_d, pairTabPotType_d[0], excludes_d,
                                                                           excludeMap_d, numExcludes, pairlistdist2, pos_tex[0], neighbors_tex);
      #endif

      gpuKernelCheck();
      gpuErrchk(cudaDeviceSynchronize()); /* RBTODO: sync needed here? */

      #ifdef USE_NCCL
      if (gpuman.gpus.size() > 1) {
	  // Currently we don't use numPairs_d[i] for i > 0... might be able to reduce data transfer with some kind nccl scatter, and in that case we'd prefer to use all numPairs_d[i]
	  gpuErrchk(cudaMemcpy(&numPairs, numPairs_d[0], sizeof(int), cudaMemcpyDeviceToHost));
	  gpuman.nccl_broadcast(0, pairTabPotType_d, pairTabPotType_d, numPairs, -1);
	  gpuman.nccl_broadcast(0, pairLists_d, pairLists_d, numPairs, -1);
      }
      gpuman.sync();
      #endif

}
float NonbondedVerletCompute::computeTabulated(bool get_energy, const ComputeForce &compute) {
    const size_t num = compute.num;
    const size_t num_rb_attached_particles = compute.num_rb_attached_particles;
    const size_t numGroupSites = compute.numGroupSites;
    const std::vector<BaseGrid*>& sys_d = compute.sys_d;
    const size_t num_replicas = compute.numReplicas;

    const std::vector<Vector3*>& forceInternal_d = compute.forceInternal_d;
    float* energies_d = compute.energies_d;

    const std::vector<Vector3*>& pos_d = compute.pos_d;
    const std::vector<cudaTextureObject_t>& pos_tex = compute.pos_tex;

    const std::vector<TabulatedPotential**>& tablePot_d = compute.tablePot_d;


    const size_t gridSize = compute.gridSize;
    dim3 numBlocks(gridSize, 1, 1);
    dim3 numThreads(NUM_THREADS, 1, 1);

    // Call the kernel to calculate the forces
    // int nb = (decomp.nCells.x * decomp.nCells.y * decomp.nCells.z);
    // int nb = (1+(decomp.nCells.x * decomp.nCells.y * decomp.nCells.z)) * 75; /* RBTODO: number of pairLists */
    const int nb = 800;
    // printf("ComputeTabulated\n");


	// RBTODO: get_energy
	if (get_energy)
	//if (false)
	{
	    // TODO clear energies earlier and in a more sensible place
		//clearEnergies<<< nb, numThreads >>>(energies_d,num);
		//gpuErrchk(cudaDeviceSynchronize());
	        cudaMemset((void*)energies_d, 0, sizeof(float)*(num+num_rb_attached_particles+numGroupSites)*num_replicas);
		computeTabulatedEnergyKernel<<< nb, numThreads >>>(forceInternal_d[0], pos_d[0], sys_d[0],
						cutoff2, numPairs_d[0], pairLists_d[0], pairTabPotType_d[0], tablePot_d[0], energies_d);
	}

	else
	{
	    // Copy positions from device 0 to all others

                //gpuErrchk(cudaBindTexture(0,  PosTex, pos_d[0],sizeof(Vector3)*num*num_replicas));
		//computeTabulatedKernel<<< nb, numThreads >>>(forceInternal_d[0], pos_d[0], sys_d[0],

	    int ngpu = gpuman.gpus.size();
	    if (ngpu == 1) {
		int i = 0;
		computeTabulatedKernel<64><<< dim3(2048,1,1), dim3(64,1,1), 0, gpuman.gpus[i].get_next_stream() >>>
		    (forceInternal_d[i], sys_d[i], cutoff2, numPairs_d[i], pairLists_d[i], pairTabPotType_d[i], tablePot_d[i], pairLists_tex[i], pos_tex[i], pairTabPotType_tex[i]);

	    } else {
	    for (size_t i = 0; i < ngpu; ++i) {
		gpuman.use(i);
		int start =            floor( ((float) numPairs*i    )/ngpu );
		int end   = i < ngpu-1 ? floor( ((float) numPairs*(i+1))/ngpu ) : numPairs;

		if (i == ngpu-1) assert(end == numPairs);
		computeTabulatedKernel<64><<< dim3(2048,1,1), dim3(64,1,1), 0, gpuman.gpus[i].get_next_stream() >>>(forceInternal_d[i], sys_d[i],
														    cutoff2, pairLists_d[i], pairTabPotType_d[i], tablePot_d[i], pairLists_tex[i], pos_tex[i], pairTabPotType_tex[i], start, end-start);
                  gpuKernelCheck();
	    }
	    gpuman.use(0);
	    }
                //gpuErrchk(cudaUnbindTexture(PosTex));
	}
	/* printPairForceCounter<<<1,32>>>(); */

    return 0.0f;
}
