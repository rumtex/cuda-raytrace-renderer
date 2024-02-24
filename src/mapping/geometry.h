
#define GEOOBJ_GRID     1
#define GEOOBJ_SPHERE   2
#define GEOOBJ_RAY      3
#define GEOOBJ_PLANE    4

typedef struct
{
    int     gtype;
    vec4    color;
    union {
        plane_t   p;
        sphere_t  s;
        ray_t     r;
    };
} geometry_t;
