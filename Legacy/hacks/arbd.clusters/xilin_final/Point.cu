#include "Point.cuh"



__host__ __device__
inline float pbc_dist(const float &a, const float &b, const float &bound){
    float delta = a>b ? a-b : b-a;
    if(delta*2 > bound) delta = bound - delta;
    return delta;
}

/* square of distance between 2 points
 * @params: p: another point
 */
__host__ __device__
inline float Point::dist2(const Point &p, const float3 &bound) const{
    float delta_x = pbc_dist(p.x, x, bound.x), delta_y = pbc_dist(p.y, y, bound.y), delta_z = pbc_dist(p.z, z, bound.z);
    return delta_x * delta_x + delta_y * delta_y + delta_z * delta_z;
}


// compare two particles by z coordinate, used for clustering
__host__ __device__
bool Point::operator<(const Point &p) const{
    return z < p.z;
}

// check if two points are equal
__host__ __device__
bool Point::operator==(const Point &p) const{
    return (x == p.x) && (y == p.y) && (z == p.z);
}

/* square of distance between 2 particle
 * @params: p: another particle
 */
__host__ __device__
float Particle::dist2(const Particle &p, const float3 &bound) const{
    if(index == -1 || p.index == -1) return INVALID;
    if(index == p.index) return SAME;
    return location.dist2(p.location, bound);
}

// compare two particles by z coordinate, used for clustering
__host__ __device__
bool Particle::operator<(const Particle &p) const{
    return location.z < p.location.z;
}

// print particle info for debug
void Particle::print(){
    std::cout<<"Particle "<<index<<": ("<<location.x<<", "<<location.y<<", "<<location.z<<")"<<std::endl;
}

// return square of distances between 2 points
__host__ __device__
float dist2_Point(const Point &p0, const Point &p1, const float3 &bound){
    return p0.dist2(p1, bound);
}

// return square of distances between 2 particles
__host__ __device__
float dist2_Particle(const Particle &p0, const Particle &p1, const float3 &bound){
    return p0.dist2(p1, bound);
}
__host__ __device__
float dist2_Particle(const float4 &p0, const float4 &p1, const float3 &bound){
    if((int)p0.w == -1 || (int)p1.w == -1) return INVALID;
    if((int)p0.w == (int)p1.w) return SAME;
    return dist2_Point(Point(p0.x, p0.y, p0.z), Point(p1.x, p1.y, p1.z), bound);
}
// return the midpoint of two points
__host__ __device__
Point midpoint(const Point &p0, const Point &p1){
    return Point( (p0.x + p1.x)/2.0, (p0.y + p1.y)/2.0, (p0.z + p1.z)/2.0 );
}



