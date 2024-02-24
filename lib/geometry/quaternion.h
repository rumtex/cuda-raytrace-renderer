// интерпретация поворота (в т.ч. групп точек) вокруг заданной оси по формуле Ейлера
// vec4 q:
// r - radius
// g - i
// b - j
// a - k

typedef struct {
    vec4    q;
    vec4    qi; // inversed
} quaternion_t;

CROSS_COMPILING_OPTS quaternion_t make_quaternion(vec3 axis_vec, fract_t radian);
CROSS_COMPILING_OPTS vec4 make_im_quaternion(vec4 q);
CROSS_COMPILING_OPTS quaternion_t mul_quaternions(quaternion_t a, quaternion_t b);
CROSS_COMPILING_OPTS vec3 quaternion_rot(quaternion_t q, vec3 vec);
CROSS_COMPILING_OPTS vec3 quaternion_rot_mul(quaternion_t q, vec3 vec, fract_t mul);
