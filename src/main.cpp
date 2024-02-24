#include <UI/Window.h>
#include <utils/fps_counter.h>
#include <math.h>
int main()
{
    // ray_t a = make_ray(make_vec3(1,0,0), normalize_vec3(make_vec3(-1,1,0)));
    // printf("ray A: pos(%f;%f;%f) dir:(%f;%f;%f)\n", a.position.x, a.position.y, a.position.z, a.direction.x, a.direction.y, a.direction.z);
    // vec3 ad2 = sum_vec3_vec3(a.position, mul_vec3_scalar(a.direction, sqrt(2.)));
    // printf("after %f be posed(%f;%f;%f)\n", sqrt(2.), ad2.x, ad2.y, ad2.z);
    // exit(1);

    init_const();
    map_t* world = init_map_gpu();

    geometry_t white_sphere = {
        GEOOBJ_SPHERE,
        make_vec4(255,255,255,255),
        .s = make_sphere(300.,300.,50.,50.0)
    };
    geometry_t green_sphere = {
        GEOOBJ_SPHERE,
        make_vec4(0,255,0,255),
        .s = make_sphere(200.,200.,60.,20.0),
    };
    add_geometry_gpu(world, white_sphere);
    add_geometry_gpu(world, green_sphere);

    geometry_t plane = {
        GEOOBJ_GRID,
        make_vec4(0.,0.,255.,255.),
        .p = make_plane(normalize_vec3(make_vec3(0.,0.,1.)), 0.)
    };
    add_geometry_gpu(world, plane);

    geometry_t ray = {
        GEOOBJ_RAY,
        make_vec4(255.,0.,0.,255.),
        .r = make_ray(make_vec3(100,100,0), normalize_vec3(make_vec3(1.,1.,5./30)))
    };
    add_geometry_gpu(world, ray);

    // unsigned width = 1920, height = 1080;
    unsigned width = 1024, height = 768;
    window_t* main_window = new window_t(world, "Hey World", width, height);

    //loop
    while (main_window->running())
    {
        update_fps();
        main_window->loop();
    }

    delete main_window;

    clear_map_gpu(world);
    return 0;
}
