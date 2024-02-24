#include <geometry.h>

CROSS_COMPILING_OPTS sphere_t make_sphere(fract_t px, fract_t py, fract_t pz, fract_t r)
{
    sphere_t result; result.position.x = px; result.position.y = py; result.position.z = pz; result.radius = r;
    return result;
}

CROSS_COMPILING_OPTS vec3 sphere_normal(sphere_t sphere, vec3 vec)
{
    return div_vec3_scalar(diff_vec3_vec3(vec, sphere.position), sphere.radius);
}

CROSS_COMPILING_OPTS fract_t ray_intersect_sphere_at(ray_t r, sphere_t s)
{
    // расстояние начала луча до перпендикуляра к центру сферы (с) Ярошевич
    fract_t cdist = -sqsum3(diff_vec3_vec3(r.position, s.position), r.direction);
    vec3 q = sum_vec3_vec3(r.position, mul_vec3_scalar(r.direction, cdist));
    vec3 b = diff_vec3_vec3(q, s.position);
    fract_t bSq = sqsum3(b, b);
    fract_t rSq = s.radius*s.radius; // TODO можно дежать в производных сферы
    if (bSq > rSq) return NO_INTERSECTION;

    fract_t a = sqrt(rSq - bSq);

    if (cdist >= a)
        return cdist - a;

    if (cdist + a > 0)
        return cdist + a;

    return NO_INTERSECTION;
}

CROSS_COMPILING_OPTS vec3 spheric_params(vec3 position, fract_t r, fract_t vert_ang, fract_t horiz_ang)
{
    // (c) Wiki
    fract_t rhsin = r*sin(vert_ang);
    return make_vec3(
        position.x + rhsin*cos(horiz_ang),
        position.y + rhsin*sin(horiz_ang),
        position.z + r*cos(vert_ang)
    );
}
