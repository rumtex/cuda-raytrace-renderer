#include <geometry.h>

CROSS_COMPILING_OPTS ray_t make_ray(vec3 position, vec3 direction)
{
    ray_t result; result.position = position; result.direction = direction;
    return result;
}

    #define EPS 0.1
CROSS_COMPILING_OPTS fract_t ray_intersect_ray_at(ray_t a, ray_t b)
{
    // TODO таким способом ось Z не нарисовать :)
    plane_t derivative_plane = make_plane_at(make_vec3(b.direction.y, -b.direction.x, 0), b.position);
    fract_t plane_intersected_at = ray_intersect_plane_at(a, derivative_plane);
    vec3 plane_intersection_point = sum_vec3_vec3(a.position, mul_vec3_scalar(a.direction, plane_intersected_at));

    fract_t d = -sqsum3(diff_vec3_vec3(b.position, plane_intersection_point), b.direction);

    vec3 qPs = diff_vec3_vec3(plane_intersection_point, sum_vec3_vec3(b.position, mul_vec3_scalar(b.direction, d)));

    return sqrt(sqsum3(qPs, qPs)) < .2 ? plane_intersected_at : NO_INTERSECTION;

    // // (c) http://www.pm298.ru/prostr2.php Взаименое расположение двух прямых в пространстве (без расчета точки пересечения)
    // // условие параллельности
    // if (a.direction.x/b.direction.x == a.direction.y/b.direction.y
    //     && a.direction.x/b.direction.x == a.direction.z/b.direction.z) return NO_INTERSECTION;

    // vec3 fr = diff_vec3_vec3(b.position, a.position);
    // fract_t det = fr.x * a.direction.y * b.direction.z
    //           - fr.x * a.direction.z * b.direction.y
    //           - fr.y * a.direction.x * b.direction.z
    //           + fr.y * a.direction.z * b.direction.x
    //           + fr.z * a.direction.x * b.direction.y
    //           - fr.z * a.direction.y * b.direction.x;
    // return det >= 0. && det < 0.3 ? 100. : NO_INTERSECTION;
}
