#include <mapping/map.h>

#include <stdio.h>
camera_t make_camera(vec3 cpos, vec3 cdir, fract_t cdist, unsigned width, unsigned height)
{
    // direction must be normalized
    camera_t cam; cam.position = cpos; cam.direction = cdir; cam.projection_distance = cdist;

    // TODO надо пересчитывать когда меняется width
    cam._camera_rays_dispersion = ((CAMERA_WIDTH_ANGLE * 3.14 / 180) / width);

    cam._vert_ang = compute_vert_angle_rad(cam.direction);
    cam._horiz_ang = compute_angle_xy_rad(c.y_axis, cam.direction);
    compute_camera_derivative(&cam, width, height);

    return cam;
}

void compute_camera_derivative(camera_t* cam, unsigned width, unsigned height) {
    cam->_dir_perp_x = normalize_vec3(cam->_vert_ang >= 0 ? make_vec3(-cam->direction.y, cam->direction.x, 0) : make_vec3(cam->direction.y, -cam->direction.x, 0));

    cam->_dir_perp_y = spheric_params(
        c.zero,
        1,
        cam->_vert_ang + HALF_PI,
        cam->_horiz_ang
    );
    // rotate_vertical(cam->direction, -90. * ONE_DEGREE);//normalize_vec3(make_vec3(cam->direction.x, cam->direction.y, -cam->direction.z));
}

// под мышь
void setup_camera_ang(camera_t* cam, int dx, int dy, unsigned width, unsigned height)
{
    // кароче повороты сферические, а камера прямоугольная
    cam->_horiz_ang += dx * cam->_camera_rays_dispersion / cam->projection_distance;
    if (cam->_horiz_ang >= PI)
        cam->_horiz_ang = cam->_horiz_ang - 2*PI;
    if (cam->_horiz_ang <= -PI)
        cam->_horiz_ang = cam->_horiz_ang + 2*PI;
    cam->_vert_ang += dy * cam->_camera_rays_dispersion / cam->projection_distance;
    if (cam->_vert_ang >= PI)
        cam->_vert_ang = cam->_vert_ang - 2*PI;
    if (cam->_vert_ang <= -PI)
        cam->_vert_ang = cam->_vert_ang + 2*PI;
    cam->direction = spheric_params(
        c.zero,
        1,
        cam->_vert_ang,
        cam->_horiz_ang
    );
    compute_camera_derivative(cam, width, height);
}

// под WASD
void setup_camera_pos(camera_t* cam, int dx, int dy, unsigned width, unsigned height)
{
    vec3 delta = sum_vec3_vec3(mul_vec3_scalar(cam->_dir_perp_x, dx), mul_vec3_scalar(cam->direction, dy));

    cam->position = sum_vec3_vec3(cam->position, delta);

    // printf("new position: x %f y %f z %f\n", cam->position.x, cam->position.y, cam->position.z);

    compute_camera_derivative(cam, width, height);
}
