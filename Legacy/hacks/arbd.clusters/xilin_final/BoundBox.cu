#include "BoundBox.cuh"
#include <iostream>


__host__ __device__
inline float pbc_dist(const float &a, const float &b, const float &bound){
    float delta = a>b ? a-b : b-a;
    if(delta*2 > bound) delta = bound - delta;
    return delta;
}


/* constructor
 * @params: points: an array of points to find their bounding box
 *             num: length of points
 */
__host__ __device__
BoundBox::BoundBox(Point *points, int num){
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
}

/* constructor
 * @params:  parts: an array of particles to find their bounding box
 *             num: length of part
 */
__host__ __device__
BoundBox::BoundBox(Particle *parts, int num){
    if(parts == NULL || num < 1 ) return;
    float x_min = parts[0].location.x; float x_max = x_min;
    float y_min = parts[0].location.y; float y_max = y_min;
    float z_min = parts[0].location.z; float z_max = z_min;
    // find bounding coordinates
    for(int i = 1; i < num && parts[i].index!=-1; i++){
        if(parts[i].location.x < x_min) x_min = parts[i].location.x;
        if(parts[i].location.x > x_max) x_max = parts[i].location.x;
        if(parts[i].location.y < y_min) y_min = parts[i].location.y;
        if(parts[i].location.y > y_max) y_max = parts[i].location.y;
        if(parts[i].location.z < z_min) z_min = parts[i].location.z;
        if(parts[i].location.z > z_max) z_max = parts[i].location.z;
    }
    x0 = x_min; x1 = x_max; y0 = y_min; y1 = y_max; z0 = z_min; z1 = z_max;
}


/* return square of distances between 2 bounding boxes
 * @params: other: a reference to another BoundBox
 *          bound: boundary of simulation box
 */
__host__ __device__
float BoundBox::dist2(const BoundBox &other, const float3 &bound) const{
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

    // calculate center of bounding box
    float center0_x = (b0_x0 + b0_x1)/2;
    float center0_y = (b0_y0 + b0_y1)/2;
    float center0_z = (b0_z0 + b0_z1)/2;
    float center1_x = (b1_x0 + b1_x1)/2;
    float center1_y = (b1_y0 + b1_y1)/2;
    float center1_z = (b1_z0 + b1_z1)/2;

    // calculate distance difference in x, y, z. If the range of coordinates overlaps in any direction, set distance to 0
    float delta_x, delta_y, delta_z;
    if(center1_x > center0_x){
        delta_x = (b1_x0 - b0_x1) > 0.0f ? (b1_x0 - b0_x1) : 0.0f;
        if(delta_x*2 > BOX_X) delta_x = pbc_dist(b0_x0, b1_x1, bound.x);
    }
    else{
        delta_x = (b0_x0 - b1_x1) > 0.0f ? (b0_x0 - b1_x1) : 0.0f;
        if(delta_x*2 > BOX_X) delta_x = pbc_dist(b0_x1, b1_x0, bound.x);
    }
    if(center1_y > center0_y){
        delta_y = (b1_y0 - b0_y1) > 0.0f ? (b1_y0 - b0_y1) : 0.0f;
        if(delta_y*2 > BOX_Y) delta_y = pbc_dist(b0_y0, b1_y1, bound.y);
    }
    else{
        delta_y = (b0_y0 - b1_y1) > 0.0f ? (b0_y0 - b1_y1) : 0.0f;
        if(delta_y*2 > BOX_Y) delta_y = pbc_dist(b0_y1, b1_y0, bound.y);
    }
    if(center1_z > center0_z){
        delta_z = (b1_z0 - b0_z1) > 0.0f ? (b1_z0 - b0_z1) : 0.0f;
        if(delta_z*2 > BOX_Z) delta_z = pbc_dist(b0_z0, b1_z1, bound.z);
    }
    else{
        delta_z = (b0_z0 - b1_z1) > 0.0f ? (b0_z0 - b1_z1) : 0.0f;
        if(delta_z*2 > BOX_Z) delta_z = pbc_dist(b0_z1, b1_z0, bound.z);
    }

    return delta_x * delta_x + delta_y * delta_y + delta_z * delta_z;
}

/* return square of distances between 2 bounding boxes
 * @params: b0: first bounding box
 *          b1: second bounding box
 *       bound: boundary of simulation box
 */
__host__ __device__
float dist2_BoundBox(const BoundBox &b0, const BoundBox &b1, const float3 &bound) {
    return b0.dist2(b1, bound);
}


/* constructor
 * @params: points: an array of points to find their bounding sphere
 *             num: number of points
 *           bound: boundary of simulation box
 */
__host__ __device__
BoundSphere::BoundSphere(Point *points, int num, const float3 &bound){
// Implementation of Ritter's algorithm. The time complexity is O(n)
// However the bounding spher is not alwasys optimal. It can be 5% to 20% larger than the smallest bounding sphere
    if(points == NULL || num < 1) return;
    if(num == 1){
        center = points[0]; radius = 0;
        return;
    }

    // step1: Pick a Point x from points[num], search a Point y in points[] to maximize dist2(x, y)
    Point x = points[0];
    Point y;
    float max_dist2 = 0;
    for(int i = 1; i < num; i++){
        float cur_dist2_xy = dist2_Point(x, points[i], bound);
        if(cur_dist2_xy > max_dist2){
            max_dist2 = cur_dist2_xy;
            y = points[i];
        }
    }
    // step2: Search a Point z in points[num], which maximizes dist2(y, z). Set up an initial sphere with diameter yz
    Point z = x;
    for(int i = 1; i < num; i++){
        Point temp = points[i];
        if(temp == y || temp == x) continue;
        float cur_dist2_yz = dist2_Point(y, temp, bound);
        if(cur_dist2_yz > max_dist2){
            max_dist2 = cur_dist2_yz;
            z = temp;
        }
    }

    Point c  = midpoint(y, z);  // initial center
    float r2 = max_dist2 / 4.0; // initial radius square
    
    // step3: iterate points[num] expand the sphere to include any point not included yet
    for(int i = 0; i < num; i++){
        Point p = points[i];
        // avoid updating point on perimeter
        if(p == y || p == z) continue;
        float d2 = dist2_Point(c, p, bound);
        if(d2 > r2){// expand the sphere
            float r = sqrt(r2); // expansive
            float d = sqrt(d2); // expansive
            float inc_dis = d - r;
            float d_x = p.x - c.x;
            float d_y = p.y - c.y;
            float d_z = p.z - c.z;
            // by properties of similar triangles
            float inc_x = 0.5*inc_dis * d_x / d;
            float inc_y = 0.5*inc_dis * d_y / d;
            float inc_z = 0.5*inc_dis * d_z / d;
            // center moves in the direction of cp with magnitude of half d-r
            c = Point(c.x+inc_x, c.y+inc_y, c.z+inc_z);
            r2 = dist2_Point(c, p, bound);
        }
    }
    
    center = c;
    radius = sqrt(r2);
}

/* constructor
 * @params:  parts: an array of particles to find their bounding sphere
 *             num: number of particles
 *           bound: boundary of simulation box
 */
__host__ __device__
BoundSphere::BoundSphere(Particle *parts, int num, const float3 &bound){
    Point *points = new Point[num];
    for(int i = 0; i < num; i++) points[i] = parts[i].location;
    BoundSphere s(points, num, bound);
    radius = s.radius;
    center = s.center;
    delete []points;
}

/* constructor
 * @params: ref: a reference to another BoundSphere
 */
__host__ __device__
BoundSphere::BoundSphere(const BoundSphere &ref){
    center = ref.center;
    radius = ref.radius;
}


/* return square of distances between 2 bounding spheres
 * @params: other: a reference to another BoundSphere
 *          bound: boundary of simulation box
 */
__host__ __device__
float BoundSphere::dist2(const BoundSphere &other, const float3 &bound) const{
    float center_to_center = (float)sqrt(dist2_Point(center, other.center, bound));
    float dist = center_to_center - radius - other.radius;
    if(dist < 0) dist = 0;
    return dist * dist;
}

/* return square of distances between 2 bounding spheres
 * @params: s0: first bounding sphere
 *          s1: second bounding sphere
 *       bound: boundary of simulation box
 */
__host__ __device__
float dist2_BoundSphere(const BoundSphere &s0, const BoundSphere &s1, const float3 &bound) {
    return s0.dist2(s1, bound);
}



