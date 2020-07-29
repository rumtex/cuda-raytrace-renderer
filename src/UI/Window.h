#include <UI/GL.h>
#include <UI/GL_etc.h>
#include <render/frame_buffer.h>
#include <mapping/map.h>

struct w_controls_t {
    double      cursor_x;
    double      cursor_y;
    bool        drag = false;

    bool        key_w = false;
    bool        key_a = false;
    bool        key_s = false;
    bool        key_d = false;
};

class window_t
{
private:
    unsigned                        w_id;
    int                             w_pos_x, w_pos_y, // backup only
                                    w_backup_width, w_backup_height,
                                    w_width, w_height,
                                    w_resolution_width, w_resolution_height;
    char*                           w_title;
    w_controls_t                    w_controls;

    GLFWwindow*                     gl_window = 0x0;
    GLPixelBufferObjectEnvironment  gl_pbo_env;

    frame_buffer_t                  w_output_buffer;
    camera_t                        w_cam;
    map_t*                          w_world;

public:

    window_t(map_t* world, char* title, unsigned resolution_width, unsigned resolution_height);
    ~window_t();
    void init_callbacks();

    void toggle_fullscreen();
    void update_resolution(int width, int height);
    void resize();
    void update_pos();

    void set_pressed(int key, int action, int mods);
    void set_camera_pos(int dx, int dy);
    void set_camera_ang(int dx, int dy);
    void update_controls();

    void update_frame();

    unsigned width() {return w_width;}
    unsigned height() {return w_height;}

    unsigned resolution_width() {return w_resolution_width;}
    unsigned resolution_height() {return w_resolution_height;}

    bool is_fullscreen();
    bool running() {return !glfwWindowShouldClose( gl_window );}

private:
};
