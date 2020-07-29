
typedef struct {
    fract_t x,y,z;
} vec3;

// пожалуйся, что это так:)
typedef struct {
    fract_t r,g,b,a;
} vec4;

// initialization
CROSS_COMPILING_OPTS vec3 make_vec3(fract_t x, fract_t y, fract_t z);
CROSS_COMPILING_OPTS vec4 make_vec4(fract_t r, fract_t g, fract_t b, fract_t a);

// multiplication
CROSS_COMPILING_OPTS vec3 mul_vec3_vec3(vec3, vec3);
CROSS_COMPILING_OPTS vec4 mul_vec4_vec4(vec4, vec4);

CROSS_COMPILING_OPTS vec3 mul_vec3_scalar(vec3, fract_t z);
CROSS_COMPILING_OPTS vec4 mul_vec4_scalar(vec4, fract_t b);

// dividing
CROSS_COMPILING_OPTS vec3 div_vec3_vec3(vec3, vec3);
CROSS_COMPILING_OPTS vec4 div_vec4_vec4(vec4, vec4);

CROSS_COMPILING_OPTS vec3 div_vec3_scalar(vec3, fract_t z);
CROSS_COMPILING_OPTS vec4 div_vec4_scalar(vec4, fract_t b);

// summary
CROSS_COMPILING_OPTS vec3 sum_vec3_vec3(vec3, vec3);
CROSS_COMPILING_OPTS vec4 sum_vec4_vec4(vec4, vec4);

CROSS_COMPILING_OPTS vec3 sum_vec3_scalar(vec3, fract_t z);
CROSS_COMPILING_OPTS vec4 sum_vec4_scalar(vec4, fract_t b);

// difference
CROSS_COMPILING_OPTS vec3 diff_vec3_vec3(vec3, vec3);
CROSS_COMPILING_OPTS vec4 diff_vec4_vec4(vec4, vec4);

// CROSS_COMPILING_OPTS vec3 diff_vec3_scalar(vec3, fract_t z);
// CROSS_COMPILING_OPTS vec4 diff_vec4_scalar(vec4, fract_t b);

CROSS_COMPILING_OPTS vec3 normalize_vec3(vec3);
CROSS_COMPILING_OPTS vec4 normalize_vec4(vec4);

CROSS_COMPILING_OPTS fract_t sqsum3(vec3, vec3);
CROSS_COMPILING_OPTS fract_t sqsum4(vec4, vec4);

CROSS_COMPILING_OPTS fract_t distance_vec3(vec3 a, vec3 b);

// 3D manipulations
CROSS_COMPILING_OPTS fract_t compute_angle_xy_rad(vec3 a, vec3 b);
CROSS_COMPILING_OPTS fract_t compute_vert_angle_rad(vec3 vec);

CROSS_COMPILING_OPTS vec3 rotate_x(vec3 vec, fract_t radian);
CROSS_COMPILING_OPTS vec3 rotate_y(vec3 vec, fract_t radian);
CROSS_COMPILING_OPTS vec3 rotate_z(vec3 vec, fract_t radian);
#define rotate_gorizontal rotate_z
CROSS_COMPILING_OPTS vec3 rotate_vertical(vec3 vec, fract_t radian);
