
typedef struct {
    vec3 position;
    fract_t radius;
} sphere_t;

CROSS_COMPILING_OPTS sphere_t make_sphere(fract_t px, fract_t py, fract_t pz, fract_t r);

CROSS_COMPILING_OPTS vec3 sphere_normal(sphere_t, vec3);

/* ray intersect sphere */
CROSS_COMPILING_OPTS fract_t ray_intersect_sphere_at(ray_t, sphere_t);

// special
CROSS_COMPILING_OPTS vec3 spheric_params(vec3 position, fract_t r, fract_t vert_ang, fract_t horiz_ang);
