
typedef struct {
    vec3       norm;
    fract_t    d;
} plane_t;

CROSS_COMPILING_OPTS plane_t make_plane(vec3 norm, fract_t d);
CROSS_COMPILING_OPTS plane_t make_plane_at(vec3 norm, vec3 position);

CROSS_COMPILING_OPTS fract_t ray_intersect_plane_at(ray_t r, plane_t p);
