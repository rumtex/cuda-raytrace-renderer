#include <geometry.h>

CROSS_COMPILING_OPTS ray_t make_ray(vec3 position, vec3 direction)
{
    ray_t result; result.position = position; result.direction = direction;
    return result;
}

CROSS_COMPILING_OPTS fract_t ray_intersect_ray_at(ray_t a, ray_t b)
{
    vec3 fr = diff_vec3_vec3(b.position, a.position);
    fract_t r = fr.x * a.direction.y * b.direction.z
              - fr.x * a.direction.z * b.direction.y
              - fr.y * a.direction.x * b.direction.z
              + fr.y * a.direction.z * b.direction.x
              + fr.z * a.direction.x * b.direction.y
              - fr.z * a.direction.y * b.direction.x;
    return r >= 0 && r < 0.1 ? 5. : 0.;
}
