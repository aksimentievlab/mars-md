#include "utility.cuh"

/* calculate Lennard-Jones potential 
 * @params: r2: square distance between 2 particles
 */
__host__ __device__
force_t LJP(float r2){
    force_t r6 = r2 * r2 * r2;
    force_t r12 = r6 * r6;
    return EPSILON* (RM12/r12- 2*(RM6/r6));
}

/* read position from .restart file
 * @params: file: name of the .restart(plain-text) file
 *       pos_arr: caller allocated Point array to be filled with positions
 *           len: number of position(number of lines) to be read 
 *  return: number of positions read
 */
int read_position(const char *file, Point *pos_arr, int len){
    if(file == NULL | pos_arr == NULL) return -1;
    int idx = 0; // position index
    int n; // dummy
    float x, y, z;
    std::ifstream f(file);
    // read position line by line
    while(idx < len && f >>n >>x >>y >>z){
        pos_arr[idx] = Point(x, y, z);
        idx++;
    }
    return idx;
}

/* return simulation range as a BoundBox instance
 * @params: pos_arr: Point array of position of particles, caller should pass in all particles 
 *              len: length of pos_arr
 *  return: a BoundBox of all particles
 */
BoundBox simulation_range(Point *pos_arr, int len){
    return BoundBox(pos_arr, len);
}

