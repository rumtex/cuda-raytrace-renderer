#ifndef __GEOMETRY_H_
#define __GEOMETRY_H_

#define NO_INTERSECTION (INFINITY)

#define PI          3.14159265359
#define HALF_PI     1.570796326795
#define ONE_DEGREE  0.017453292519944 //(PI/180.)

typedef float fract_t;

#if defined(__cplusplus)

#define CUDA_OPTS

// #pragma once
// #include <cooperative_groups.h>
// namespace cg = cooperative_groups;

#ifdef __CUDACC__
#define CROSS_COMPILING_OPTS __host__ __device__
#else // __CUDACC__
#define CROSS_COMPILING_OPTS
#endif // __CUDACC__

extern "C" {
#else
#define CROSS_COMPILING_OPTS
#endif

// #include <math.h> // sqrt()

#include <geometry/vector.h>
#include <geometry/quaternion.h>
#include <geometry/ray.h>
#include <geometry/plane.h>
#include <geometry/sphere.h>

// /*
// ** y
// ** ^ ^z
// ** |/
// ** *---->x
// **     *----*
// **    /|   /|
// **   *----*-*
// **   |/   |/
// **   *----*
// */

// struct Tetragon //квадрат
// {
//     vec3 c1,c2,c3,c4;
// };

// struct Parallelogram
// {
//     // Parallelogram() = default;
//     // Parallelogram( vec3f v1, vec3f v2, vec3f anchor ):
//     // v1( v1 ), v2( v2 ), anchor( anchor )
//     // {
//     //     vec3f normal = normalize(cross( v1, v2 ));
//     //     float d = dot( normal, anchor );
//     //     this->v1 *= 1.0f / dot( v1, v1 );
//     //     this->v2 *= 1.0f / dot( v2, v2 );
//     //     plane = make_float4(normal, d);
//     // }
//     vec4  plane;
//     vec3  v1;
//     vec3  v2;
//     vec3  anchor;
// };

// __global__ void intersection_parallelogram(Parallelogram* pg, vec3& ray_orig, vec3& ray_dir, fract_t& ray_tmin, fract_t& ray_tmax, fract_t& t_result)
// {
//     vec3 n = *((vec3*)&pg->plane);
//     fract_t dt = sqsum3(ray_dir, n);
//     fract_t t = (pg->plane.a - sqsum3(n, ray_orig))/dt;
//     if(t > ray_tmin && t < ray_tmax)
//     {
//         vec3 p = sum_vec3_vec3(ray_orig, mul_vec3_scalar(ray_dir, t));
//         vec3 vi = diff_vec3_vec3(p, pg->anchor);
//         fract_t a1 = sqsum3(pg->v1, vi);
//         if(a1 >= 0 && a1 <= 1)
//         {
//             fract_t a2 = sqsum3(pg->v2, vi);
//             if(a2 >= 0 && a2 <= 1)
//             {
//                 t_result = t;
//             }
//         }
//     }
//     t_result = -1.;
// }


#if defined(__cplusplus)
} //extern "C"
#endif

#endif //__GEOMETRY_H_
