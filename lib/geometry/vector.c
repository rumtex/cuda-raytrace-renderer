#include <geometry.h>

// initialization
CROSS_COMPILING_OPTS vec3 make_vec3(fract_t x, fract_t y, fract_t z)
{
    vec3 result; result.x = x; result.y = y; result.z = z;
    return result;
}
CROSS_COMPILING_OPTS vec4 make_vec4(fract_t r, fract_t g, fract_t b, fract_t a)
{
    vec4 result; result.r = r; result.g = g; result.b = b, result.a = a;
    return result;
}

// multiplication
CROSS_COMPILING_OPTS vec3 mul_vec3_vec3(vec3 a, vec3 b)
{
    vec3 result; result.x = a.x*b.x; result.y = a.y*b.y; result.z = a.z*b.z;
    return result;
}
CROSS_COMPILING_OPTS vec4 mul_vec4_vec4(vec4 a, vec4 b)
{
    vec4 result; result.r = a.r*b.r; result.g = a.g*b.g; result.b = a.b*b.b; result.a = a.a*b.a;
    return result;
}

CROSS_COMPILING_OPTS vec3 mul_vec3_scalar(vec3 a, fract_t b)
{
    vec3 result; result.x = a.x*b; result.y = a.y*b; result.z = a.z*b;
    return result;
}
CROSS_COMPILING_OPTS vec4 mul_vec4_scalar(vec4 a, fract_t b)
{
    vec4 result; result.r = a.r*b; result.g = a.g*b; result.b = a.b*b; result.a = a.a*b;
    return result;
}

// dividing
CROSS_COMPILING_OPTS vec3 div_vec3_vec3(vec3 a, vec3 b)
{
    vec3 result; result.x = a.x/b.x; result.y = a.y/b.y; result.z = a.z/b.z;
    return result;
}
CROSS_COMPILING_OPTS vec4 div_vec4_vec4(vec4 a, vec4 b)
{
    vec4 result; result.r = a.r/b.r; result.g = a.g/b.g; result.b = a.b/b.b; result.a = a.a/b.a;
    return result;
}

CROSS_COMPILING_OPTS vec3 div_vec3_scalar(vec3 a, fract_t b)
{
    vec3 result; result.x = a.x/b; result.y = a.y/b; result.z = a.z/b;
    return result;
}
CROSS_COMPILING_OPTS vec4 div_vec4_scalar(vec4 a, fract_t b)
{
    vec4 result; result.r = a.r/b; result.g = a.g/b; result.b = a.b/b; result.a = a.a/b;
    return result;
}

// summary
CROSS_COMPILING_OPTS vec3 sum_vec3_vec3(vec3 a, vec3 b)
{
    vec3 result; result.x = a.x+b.x; result.y = a.y+b.y; result.z = a.z+b.z;
    return result;
}
CROSS_COMPILING_OPTS vec4 sum_vec4_vec4(vec4 a, vec4 b)
{
    vec4 result; result.r = a.r+b.r; result.g = a.g+b.g; result.b = a.b+b.b; result.a = a.a+b.a;
    return result;
}

CROSS_COMPILING_OPTS vec3 sum_vec3_scalar(vec3 a, fract_t b)
{
    vec3 result; result.x = a.x+b; result.y = a.y+b; result.z = a.z+b;
    return result;
}
CROSS_COMPILING_OPTS vec4 sum_vec4_scalar(vec4 a, fract_t b)
{
    vec4 result; result.r = a.r+b; result.g = a.g+b; result.b = a.b+b; result.a = a.a+b;
    return result;
}

// difference
CROSS_COMPILING_OPTS vec3 diff_vec3_vec3(vec3 a, vec3 b)
{
    vec3 result; result.x = a.x-b.x; result.y = a.y-b.y; result.z = a.z-b.z;
    return result;
}
CROSS_COMPILING_OPTS vec4 diff_vec4_vec4(vec4 a, vec4 b)
{
    vec4 result; result.r = a.r-b.r; result.g = a.g-b.g; result.b = a.b-b.b; result.a = a.a-b.a;
    return result;
}

// vec3 diff_vec3_scalar(vec3 a, fract_t b)
// {
//     vec3 result; result.
//     return result;
// }
// vec4 diff_vec4_scalar(vec4 a, fract_t b)
// {
//     vec4 result; result.
//     return result;
// }

CROSS_COMPILING_OPTS vec3 normalize_vec3(vec3 vec)
{
    fract_t a = sqrt(vec.x*vec.x + vec.y*vec.y + vec.z*vec.z);
    return make_vec3(vec.x/a, vec.y/a, vec.z/a);
}
CROSS_COMPILING_OPTS vec4 normalize_vec4(vec4 vec)
{
    fract_t a = sqrt(vec.r*vec.r + vec.g*vec.g + vec.b*vec.b + vec.a*vec.a);
    return make_vec4(vec.r/a, vec.g/a, vec.b/a, vec.a/a);
}

CROSS_COMPILING_OPTS fract_t sqsum3(vec3 a, vec3 b)
{
    return (a.x*b.x + a.y*b.y + a.z*b.z);
}
CROSS_COMPILING_OPTS fract_t sqsum4(vec4 a, vec4 b)
{
    return (a.r*b.r + a.g*b.g + a.b*b.b + a.a*b.a);
}

CROSS_COMPILING_OPTS fract_t distance_vec3(vec3 a, vec3 b)
{
    vec3 diff = diff_vec3_vec3(a,b);
    return sqrt(sqsum3(diff, diff));
}

CROSS_COMPILING_OPTS fract_t compute_vert_angle_rad(vec3 vec)
{
    return acos((vec.z)\
    / sqrt(vec.x*vec.x + vec.y*vec.y + vec.z*vec.z));
}
// CROSS_COMPILING_OPTS fract_t compute_vert_angle_rad(vec3 a, vec3 b)
// {
//     fract_t y_axis_deg = compute_angle_xy_rad(make_vec3(a.x,0,0), a);
//     vec3 ta = rotate_z(a, -y_axis_deg);
//     y_axis_deg = compute_angle_xy_rad(make_vec3(b.x,0,0), b);
//     vec3 tb = rotate_z(b, -y_axis_deg);

//     return asin((ta.y*tb.y + ta.z*tb.z)\
//     / (sqrt(ta.y*ta.y + ta.z*ta.z) * sqrt(tb.y*tb.y + tb.z*tb.z)));
// }
CROSS_COMPILING_OPTS fract_t compute_angle_xy_rad(vec3 a, vec3 b)
{
    return asin((a.x*b.x + a.y*b.y)\
    / (sqrt(a.x*a.x + a.y*a.y) * sqrt(b.x*b.x + b.y*b.y)));
}

CROSS_COMPILING_OPTS vec3 rotate_x(vec3 vec, fract_t radian)
{
    vec3 result; result.x = vec.x;
    fract_t aSin = sin(radian), aCos = cos(radian);

    result.y = vec.y * aCos - vec.z * aSin;
    result.z = vec.y * aSin + vec.z * aCos;
    return result;
}
CROSS_COMPILING_OPTS vec3 rotate_y(vec3 vec, fract_t radian)
{
    vec3 result; result.y = vec.y;
    fract_t aSin = sin(radian), aCos = cos(radian);

    result.x = vec.x * aCos + vec.z * aSin;
    result.z = - vec.x * aSin + vec.z * aCos;
    return result;
}
CROSS_COMPILING_OPTS vec3 rotate_z(vec3 vec, fract_t radian)
{
    vec3 result; result.z = vec.z;
    fract_t aSin = sin(radian), aCos = cos(radian);

    result.x = vec.x * aCos - vec.y * aSin;
    result.y = vec.x * aSin + vec.y * aCos;
    return result;
}

CROSS_COMPILING_OPTS vec3 rotate_vertical(vec3 vec, fract_t radian)
{

//     // полу 2D спонтанное решение (70 fps)
//     fract_t y_axis_deg = compute_angle_xy_rad(make_vec3(vec.x,0,0), vec);

//     if (vec.y < 0 && vec.x > 0 || vec.y > 0 && vec.x < 0)
//     {
//         y_axis_deg = -y_axis_deg;
//     }

// // // не работает
// // // #ifdef __CUDACC__
// // //     cooperative_groups::sync(cooperative_groups::this_thread_block());
// // // #endif

//     vec3 result;
//     result = rotate_z(vec, y_axis_deg);
//     result = rotate_x(result, (vec.y < 0 || vec.x < 0) ? -radian : radian);
//     result = rotate_z(result, -y_axis_deg);


    // // линза родригеса с неминуемой погрешностью :) (100fps)
    // // make magnit vec3
    // vec3 magnit = make_vec3(0,0,.0000001);//make_vec3(vec.x, vec.y, -vec.z);

    // // pitch
    // fract_t aSin = sin(radian), aCos = cos(radian);
    // // fract_t aSin = sin(radian * 0.5), aCos = cos(radian * 0.5);
    // vec3 result = sum_vec3_vec3(
    //     sum_vec3_vec3(
    //       mul_vec3_scalar(vec, aCos),
    //       mul_vec3_scalar(mul_vec3_vec3(vec, magnit), aSin)
    //     ),
    //     mul_vec3_scalar(magnit, sqsum3(vec, magnit) * (1 - aCos))
    // );

    // return result;

    // // make perp vec3
    // vec3 perp = make_vec3(0,0,.0000001);//make_vec3(vec.x, vec.y, -vec.z);

    vec3 normalized_perp = normalize_vec3(make_vec3(vec.y,-vec.x,0));
    fract_t aSin = sin(radian * 0.5), aCos = cos(radian * 0.5);
    vec4 result, qi, q;
    q = make_vec4(aCos, normalized_perp.x*aSin, normalized_perp.y*aSin, normalized_perp.z*aSin);
    fract_t mag = sqsum4(q,q);
    q = make_vec4(q.r, q.g/mag, q.b/mag, q.a/mag);
    mag = sqsum4(q,q);
    qi = make_vec4(q.r/mag, -q.g/mag, -q.b/mag, -q.a/mag);

    // return make_vec3(qi.r, qi.g, qi.b);

    fract_t r = 0*qi.r - vec.x*qi.g - vec.y*qi.b - vec.z*qi.a;
    fract_t g = 0*qi.g + vec.x*qi.r + vec.y*qi.a - vec.z*qi.b;
    fract_t b = 0*qi.b - vec.x*qi.a + vec.y*qi.r + vec.z*qi.g;
    fract_t a = 0*qi.a + vec.x*qi.b - vec.y*qi.g + vec.z*qi.r;

    result.r = q.r*r - q.g*g - q.b*b - q.a*a;
    result.g = q.r*g + q.g*r + q.b*a - q.a*b;
    result.b = q.r*b - q.g*a + q.b*r + q.a*g;
    result.a = q.r*a + q.g*b - q.b*g + q.a*r;

    return make_vec3(result.g, result.b, result.a);
}

    // fract_t aSin = sin(radian), aCos = cos(radian);

    // result.x = vec.y * aCos * vec.z * aCos - vec.y * aCos * vec.z * aSin + vec.y * aSin;
    // result.y = vec.x * aSin * vec.y * aSin * vec.z * aCos + vec.x * aCos * vec.z * aSin - vec.x * aSin * vec.y * aSin * vec.z * aSin + vec.x * aCos * vec.z * aCos - vec.x * aSin * vec.y * aCos;
    // result.z = - vec.x * aCos * vec.y * aSin * vec.z * aCos + vec.x * aSin * vec.z * aSin + vec.x * aCos * vec.y * aSin * vec.z * aSin + vec.x * aSin * vec.z * aCos + vec.x * aCos * vec.y * aCos;
