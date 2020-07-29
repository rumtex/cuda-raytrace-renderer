#include <geometry.h>

CROSS_COMPILING_OPTS plane_t make_plane(vec3 norm, fract_t d)
{
    plane_t result; result.norm = norm; result.d = d;
    return result;
}

CROSS_COMPILING_OPTS plane_t make_plane_at(vec3 norm, vec3 position)
{
    plane_t result; result.norm = norm; result.d = sqsum3(norm, position);
    return result;
}

CROSS_COMPILING_OPTS fract_t ray_intersect_plane_at(ray_t r, plane_t p)
{
    fract_t c = sqsum3(p.norm, r.direction);
    if (c == 0) return NO_INTERSECTION; // луч лежит в плоскости
    return (p.d - sqsum3(p.norm, r.position)) / c;
}