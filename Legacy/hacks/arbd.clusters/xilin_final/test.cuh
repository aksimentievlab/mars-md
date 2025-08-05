#ifndef TEST
#define TEST

#include "utility.cuh"

// print the information of grid
// make sure x_range, y_range, z_range, volume, density, grid_width, 
// range, x_num, and y_num are declared and calculated
#define PRINT_GRID_INFO \
do{\
    std::cout.precision(6);\
    std::cout\
    <<"x_range: "<<x_range<<" from "<<range.x0<<" to "<<range.x1<<std::endl\
    <<"y_range: "<<y_range<<" from "<<range.y0<<" to "<<range.y1<<std::endl\
    <<"z_range: "<<z_range<<" from "<<range.z0<<" to "<<range.z1<<std::endl\
    <<"particles #: "<<(int)LEN<<" | "<<"volume:"<<volume<<" | "\
    <<"density: "<<density<<std::endl<<"grid width: "<<grid_width<<" | "\
    <<"grid # in x: "<<x_num<<" | " <<"grid # in y: "<<y_num<<std::endl;\
}while(0);


// print the number of particles per x,y grid of clusters
// make sure std::vector<Particle> particles_each_grid[x_num * y_num] is fully constructed beforehand
#define PRINT_GRID_PARTICLES_NUM \
do{\
    for(int i = 0; i < x_num; i++ )\
        for(int j = 0; j < y_num; j++)\
            std::cout<<"grid ("<<i<<", "<<j<<") has particles: "<<particles_each_grid[i * y_num + j].size()<<std::endl;\
}while(0);

// print all particles in (x, y) grid
// make sure std::vector<Particle> particles_each_grid[x_num * y_num] is fully constructed beforehand
#define PRINT_PARTICLES_IN_GRID(x, y)\
do{\
    std::cout<<"Grid("<<(int)x<<", "<<(int)y<<") has particles in range:"<<std::endl\
    <<"x: ["<<x_min + x * cluster_width<<", "<<x_min + (x+1) * cluster_width<<"] "\
    <<"y: ["<<y_min + y * cluster_width<<", "<<y_min + (y+1) * cluster_width<<"]"<<std::endl;\
    for(int i = 0; i<particles_each_grid[x * y_num + y].size(); i++)\
        particles_each_grid[x * y_num + y][i].print();\
}while(0);

// print all clusters， clusters are stored in a vector
// make sure std::vector<Particle> particles_each_grid[x_num * y_num] is fully constructed beforehand
#define PRINT_ALL_CLUSTERS_VECTOR \
do{\
    for(int cluster_idx = 0; cluster_idx < cluster_vector.size(); cluster_idx ++){\
        std::cout<<"No."<<cluster_idx<<" ";\
        cluster_vector[cluster_idx].print();\
    }\
}while(0);

// print results for all particles
// make sure forces are calculated beforehand
#define PRINT_RESULTS \
do{\
    for(int i = 0; i < LEN; i++){\
        std::cout.precision(8);\
        std::cout<<"Particle No."<<i<<": "<<forces[i]<<std::endl;\
    }\
}while(0);

// print to check how many particles in neighbor clusters are actually neighbor particles 
// make sure forces are calculated beforehand
#define PRINT_PARTICLES_HIT \
do{\
    for(int x = 0; x < x_num; x++)\
    for(int y = 0; y < y_num; y++)\
    for(int z = 0; z < grids[x * y_num + y].num; z++){\
        Cluster<CLUSTER_SIZE> *cluster = fetch_cluster(Point(x,y,z), grids, y_num);\
        int miss = cluster->miss, hit = cluster->hit;\
        std::cout<<"Cluster("<<x<<", "<<y<<", "<<z<<")'s neighbor clusters have "<<miss+hit<<" particles, "<<hit<<" are neighbors"<<std::endl;\
    }\
}while(0);

// compare results with brute-force double-loop method
// make sure forces are caluclated beforehand
#define COMPARE_RESULTS \
do{\
    force_t *forces1 = new force_t[LEN];\
    run_double_loop(forces1, LEN, CUTOFF2, make_float3(BOX_X, BOX_Y, BOX_Z));\
    int counter = 0;\
    int zero_counter = 0;\
    std::cout.precision(8);\
    for(int i = 0; i < LEN; i++){\
        if(forces[i] - forces1[i] > 1 || forces1[i] - forces[i] > 1 ){\
            std::cout<<"MISMATCH at particle "<<i<<": "<<forces[i]<<" vs "<<forces1[i]<<std::endl;\
            counter++;\
        }\
        if(forces[i]==0){zero_counter++;}\
    }\
    std::cout<<"______________________________________________________"<<std::endl;\
    if(counter){\
        std::cout<<"Comparison summary: total number of mismatches: "<<counter<<std::endl;\
        std::cout<<"Number of zeros: "<<zero_counter<<std::endl;\
    }\
    else{\
        std::cout<<"Comparison summary: no mismatch!"<<std::endl;\
    }\
    delete []forces1;\
}while(0);


// entry of the test program
int test();

// näive brute-force non-bonded force calculation on CPU for comparaison 
void run_double_loop(force_t *forces, int len, float cutoff2, float3 bound);

// näive brute-force non-bonded force calculation on GPU for comparaison 
__global__
void run_double_loop_kernel(Point *pos_arr, force_t *forces, int len, float cutoff2, float3 bound);

// test read_postion
void read_position_test(Point *pos_arr, int len);

// test constructor of BoundBox
void construct_BoundBox_test(Point *pos_arr, int len);

// print BoundBox vertices
void print_BoundBox(const BoundBox &b);



#endif