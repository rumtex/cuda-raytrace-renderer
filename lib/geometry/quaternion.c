#include <geometry.h>

// axis_vec must be normalized
CROSS_COMPILING_OPTS quaternion_t make_quaternion(vec3 axis_vec, fract_t radian)
{
    quaternion_t result;

    fract_t aSin = sin(radian * 0.5), aCos = cos(radian * 0.5);
    result.q = make_vec4(
        aCos,
        axis_vec.x * aSin,
        axis_vec.y * aSin,
        axis_vec.z * aSin
    );

    fract_t mag = sqsum4(result.q,result.q);
    result.q = make_vec4(
        result.q.r,
        result.q.g/mag,
        result.q.b/mag,
        result.q.a/mag
    );

    result.qi = make_im_quaternion(result.q);

    return result;
}

CROSS_COMPILING_OPTS vec4 make_im_quaternion(vec4 q)
{
    fract_t mag = sqsum4(q,q);
    vec4 result = make_vec4(
        q.r/mag,
        -q.g/mag,
        -q.b/mag,
        -q.a/mag
    );

    return result;
}

CROSS_COMPILING_OPTS quaternion_t mul_quaternions(quaternion_t a, quaternion_t b)
{
    quaternion_t result;

    // result.q.r = a.q.r*b.q.r - a.q.g*b.q.g - a.q.b*b.q.b - a.q.a*b.q.a;
    // result.q.g = a.q.r*b.q.g + a.q.g*b.q.r + a.q.b*b.q.a - a.q.a*b.q.b;
    // result.q.b = a.q.r*b.q.b - a.q.g*b.q.a + a.q.b*b.q.r + a.q.a*b.q.g;
    // result.q.a = a.q.r*b.q.a + a.q.g*b.q.b - a.q.b*b.q.g + a.q.a*b.q.r;


    result.q.r = -a.q.g * b.q.g - a.q.b * b.q.b - a.q.a * b.q.a;
    result.q.g = a.q.r * b.q.g + a.q.b * b.q.a - a.q.a * b.q.b;
    result.q.b = a.q.r * b.q.b + a.q.a * b.q.g - a.q.g * b.q.a;
    result.q.a = a.q.r * b.q.a + a.q.g * b.q.b - a.q.b * b.q.g;

    // result.q = mul_vec4_vec4(a.q,b.q);
    // result.qi = mul_vec4_vec4(a.qi,b.qi);
    result.qi = make_im_quaternion(result.q);

    return result;
}

CROSS_COMPILING_OPTS vec3 quaternion_rot(quaternion_t q, vec3 vec)
{
    fract_t r = vec.x*q.qi.g - vec.y*q.qi.b - vec.z*q.qi.a;
    fract_t g = vec.x*q.qi.r + vec.y*q.qi.a - vec.z*q.qi.b;
    fract_t b = vec.x*q.qi.a + vec.y*q.qi.r + vec.z*q.qi.g;
    fract_t a = vec.x*q.qi.b - vec.y*q.qi.g + vec.z*q.qi.r;

    vec3 result;
    result.x = q.q.r*g + q.q.g*r + q.q.b*a - q.q.a*b;
    result.y = q.q.r*b - q.q.g*a + q.q.b*r + q.q.a*g;
    result.z = q.q.r*a + q.q.g*b - q.q.b*g + q.q.a*r;

    return result;
}

// CROSS_COMPILING_OPTS vec3 quaternion_rot_mul(quaternion_t qm, vec3 vec, fract_t mul)
// {
//     // fract_t aSin = sin(radian * 0.5), aCos = cos(radian * 0.5);

//     vec4 qi = mul_vec4_scalar(qm.qi, mul);
//     vec4 q = mul_vec4_scalar(qm.q, mul);

//     fract_t r = vec.x*qi.g - vec.y*qi.b - vec.z*qi.a;
//     fract_t g = vec.x*qi.r + vec.y*qi.a - vec.z*qi.b;
//     fract_t b = vec.x*qi.a + vec.y*qi.r + vec.z*qi.g;
//     fract_t a = vec.x*qi.b - vec.y*qi.g + vec.z*qi.r;

//     vec3 result;
//     result.x = q.r*g + q.g*r + q.b*a - q.a*b;
//     result.y = q.r*b - q.g*a + q.b*r + q.a*g;
//     result.z = q.r*a + q.g*b - q.b*g + q.a*r;

//     return result;
// }
