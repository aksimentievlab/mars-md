#include "test.cuh"

// entry of the test program
int test(){
   return 0;
}

// näive brute-force non-bonded force calculation on CPU for comparaison
void run_double_loop(force_t *forces, int len, float cutoff2, float3 bound){
    #if (TIMER_FLAG == 1)
    float time;
    cudaEvent_t start, stop;
    std::cout<<"______________________________________________________"<<std::endl;
    std::cout<<"Benchmark for particle-particle simulation:"<<std::endl;
    #endif
    Point *pos_arr = new Point[len];
    read_position(POSITION_FILE, pos_arr, len);
    memset(forces, 0.0f, sizeof(force_t )*len);
    #if (USE_GPU == 1)
        #if (TIMER_FLAG == 1)
        TIMER_START("allocate memory on GPU")
        #endif
        Point *pos_arr_d;
        force_t *forces_d;
        cudaMalloc((void**)&pos_arr_d, sizeof(Point)*len);
        cudaMalloc((void**)&forces_d, sizeof(force_t)*len);
        cudaMemset(forces_d, 0, sizeof(force_t)*len);
        cudaMemcpy(pos_arr_d, pos_arr, sizeof(Point)*len, cudaMemcpyHostToDevice);
        #if (TIMER_FLAG == 1)
        TIMER_END
        #endif
        #if (TIMER_FLAG == 1)
        TIMER_START("run brute-force double loop on GPU")
        #endif
        dim3 block_dim(1024,1,1);
        dim3 grid_dim(ceil((float)len/1024), 1, 1);
        run_double_loop_kernel<<<grid_dim, block_dim>>>(pos_arr_d, forces_d, len, cutoff2, bound);
        #if (TIMER_FLAG == 1)
        TIMER_END
        #endif
        #if (TIMER_FLAG == 1)
        TIMER_START("transfer and deallocate memory on GPU and CPU")
        #endif
        cudaMemcpy(forces,forces_d, sizeof(force_t)*len, cudaMemcpyDeviceToHost);
        cudaFree(pos_arr_d);
        cudaFree(forces_d);
        #if (TIMER_FLAG == 1)
        TIMER_END
        #endif
    #else
        #if (TIMER_FLAG == 1)
        TIMER_START("run brute-force double loop on CPU")
        #endif
        #if (USE_OMP == 1)
        #pragma omp parallel for num_threads(THREADS_OMP)
        #endif    
        for(int a = 0; a < len; a++){
            int neighbor_counter = 0;
            for(int b = 0; b < len; b++){
                if(a == b) continue;
                float dist2 = dist2_Point(pos_arr[a], pos_arr[b], bound);
                if(dist2 < cutoff2){
                    forces[a] += LJP(dist2);
                    neighbor_counter++;
                }
            }
        }
        delete []pos_arr;
        #if (TIMER_FLAG == 1)
        TIMER_END
        #endif
    #endif
}
// näive brute-force non-bonded force calculation on GPU for comparaison
__global__
void run_double_loop_kernel(Point *pos_arr, force_t *forces, int len, float cutoff2, float3 bound){
    int a = blockIdx.x * blockDim.x + threadIdx.x;
    if(a < len){
        for(int b = 0; b < len; b++){
            float dist2 = dist2_Point(pos_arr[a], pos_arr[b], bound);
            if(dist2 < cutoff2)
                forces[a] += LJP(dist2);
        }
    }
}



// test read_postion
void pos_read_test(Point *pos_arr, int len){
    for(int i =0; i < len; i++)
        std::cout<<pos_arr[i].x<<" "<<pos_arr[i].y<<" "<<pos_arr[i].z<<std::endl;
}

// test constructor of BoundBox
void construct_BoundBox_test(Point *pos_arr, int len){
    BoundBox b(pos_arr, len);
    // print_BoundBox(b);
}

// print BoundBox vertices
void print_BoundBox(const BoundBox &b){
   // print float
   std::cout.precision(8);
   std::cout<<std::setw(10)<<"BoundBox vertices:\n"
   <<"("<<b.x0<<","<<b.y0<<","<<b.z0<<"), "
   <<"("<<b.x0<<","<<b.y0<<","<<b.z1<<"), "
   <<"("<<b.x0<<","<<b.y1<<","<<b.z0<<"), "
   <<"("<<b.x0<<","<<b.y1<<","<<b.z1<<"), "
   <<"("<<b.x1<<","<<b.y0<<","<<b.z0<<"), "
   <<"("<<b.x1<<","<<b.y0<<","<<b.z1<<"), "
   <<"("<<b.x1<<","<<b.y1<<","<<b.z0<<"), "
   <<"("<<b.x1<<","<<b.y1<<","<<b.z1<<"), " << std::endl;
}

