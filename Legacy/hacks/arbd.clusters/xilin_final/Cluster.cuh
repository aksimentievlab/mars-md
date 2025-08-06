#ifndef CLUSTER
#define CLUSTER

#include "BoundBox.cuh"

// cluster of nearby SIZE particles
template<int SIZE>
class Cluster{
    public:
        /* constructor
         * @params: part_idx: an int array of particle indices
         *               loc: a point array of particle locations
         *          part_num: number of particles
         */
        __host__ __device__
        Cluster(int *part_idx, Point *loc, int part_num){
            for(int i = 0; i < part_num; i++)
                particles[i] = Particle(loc[i], part_idx[i]);
            for(int i = part_num; i < SIZE; i++)
                particles[i] = Particle(Point(0.0f,0.0f,0.0f), -1);
        }
        /* constructor
         * @params:    parts: a particle array
         *          part_num: number of particles
         */
        __host__ __device__
        Cluster(Particle *parts, int part_num){
            for(int i = 0; i < part_num; i++)
                particles[i] = parts[i];
            for(int i = part_num; i < SIZE; i++)
                particles[i] = Particle(Point(0.0f,0.0f,0.0f), -1);
        }
        __host__ __device__
        // default constructor
        Cluster(){
        }

    public:
        // an array of particles belonging to the cluster
        Particle particles[SIZE];

};


#endif

