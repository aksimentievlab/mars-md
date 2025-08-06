#ifndef BB
#define BB

#include "Point.cuh"

// the bounding box of an array of points(particles)
struct BoundBox{
    public:
        // constructors
        __host__ __device__
        BoundBox(Point *points, int num);
        __host__ __device__
        BoundBox(Particle *parts, int num);
        __host__ __device__
        BoundBox(){};
        // return square of distances between 2 bounding boxes
        __host__ __device__
        float dist2(const BoundBox &other, const float3 &bound) const;

    public:
        // x0, x1, y0, y1, z0, z1 are the ranges of x, y, z coordinates
        // for example the bottom front left corner is (x0, y0, z0)
        float x0, x1, y0, y1, z0, z1;
};

// return square of distances between 2 bounding boxes
__host__ __device__
float dist2_BoundBox(const BoundBox &b0, const BoundBox &b1, const float3 &bound);

// the bounding sphere of an array of points
struct BoundSphere{
    public:
        // constructors
        __host__ __device__
        BoundSphere(Point *points, int num, const float3 &bound);
        __host__ __device__
        BoundSphere(Particle *parts, int num, const float3 &bound);
        __host__ __device__
        BoundSphere(const BoundSphere &ref);
        __host__ __device__
        BoundSphere(){};
        // return square of distances between 2 bounding spheres
        __host__ __device__
        float dist2(const BoundSphere &other, const float3 &bound) const;

    public:
        Point center;
        float radius;
};


// return square of distances between 2 bounding spheres
__host__ __device__
float dist2_BoundSphere(const BoundSphere &s0, const BoundSphere &s1, const float3 &bound);

#endif