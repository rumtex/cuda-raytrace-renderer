
typedef struct {
    vec3    position;
    vec3    direction; // normalized
} ray_t;

CROSS_COMPILING_OPTS ray_t make_ray(vec3 position, vec3 direction);

CROSS_COMPILING_OPTS fract_t ray_intersect_ray_at(ray_t a, ray_t b);
