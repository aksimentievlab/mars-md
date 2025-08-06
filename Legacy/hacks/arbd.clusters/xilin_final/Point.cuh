#ifndef POINT
#define POINT

#ifndef NULL
#define NULL 0
#endif

#include "setup.h"

// illegal values for particle-particle distance
#define INVALID (float)65536  // distance of invalid particle i.e. index == -1
#define SAME    (float)65535  // distance of same particles

// 3D point class
struct Point{
    public:
        // constructors
        __host__ __device__
        Point(float i, float j, float k)
           :x(i), y(j), z(k){}  
        __host__ __device__
        Point(const Point &p)
           :x(p.x), y(p.y), z(p.z){} 
        __host__ __device__
        Point(){}
        // square of distance between 2 points
        __host__ __device__
        float dist2(const Point& p, const float3 &bound) const;

        // compare two particles by z coordinate, used for clustering
        __host__ __device__
        bool operator<(const Point &p) const;

        // check if two points are equal
        __host__ __device__
        bool operator==(const Point &p) const;

    public: 
        float x, y, z;

};

// a particle with index and location
struct Particle{
    public:
        // constructors
        __host__ __device__
        Particle(Point loc, int idx)
           :location(loc), index(idx){}

        __host__ __device__
        Particle(int x, int y, int z, int idx)
           :location(Point(x,y,z)), index(idx){}

        __host__ __device__
        Particle(){}

        // square of distance between 2 particles
        __host__ __device__
        float dist2(const Particle &p, const float3 &bound) const;

        // compare two particles by z coordinate, used for clustering
        __host__ __device__
        bool operator<(const Particle &p) const;

        // print particle info for debug
        void print();

    public:
        Point location;
        int index;

};

// return square of distances between 2 points
__host__ __device__
float dist2_Point(const Point &p0, const Point &p1, const float3 &bound);

// return square of distances between 2 particles
__host__ __device__
float dist2_Particle(const Particle &p0, const Particle &p1, const float3 &bound);
__host__ __device__
float dist2_Particle(const float4 &p0, const float4 &p1, const float3 &bound);

// return the midpoint of two points
__host__ __device__
Point midpoint(const Point &p0, const Point &p1); 

#endif