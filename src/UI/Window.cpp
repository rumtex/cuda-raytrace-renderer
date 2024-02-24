#include <stdio.h>

#include <UI/Window.h>
#include <UI/WindowCallbacks.h>

// #define SCALED_TO_RESOLUTION

// void initImGui( GLFWwindow* window )
// {
//     IMGUI_CHECKVERSION();
//     ImGui::CreateContext();
//     ImGuiIO& io = ImGui::GetIO();
//     (void)io;

//     ImGui_ImplGlfw_InitForOpenGL( window, false );
//     ImGui_ImplOpenGL3_Init();
//     ImGui::StyleColorsDark();
//     io.Fonts->AddFontDefault();

//     ImGui::GetStyle().WindowBorderSize = 0.0f;
// }

unsigned window_id_it = 0;
window_t::window_t(map_t* world, char* title, unsigned resolution_width = 0, unsigned resolution_height = 0) :
    w_id(window_id_it++),
    w_width(resolution_width),
    w_height(resolution_height),
    w_resolution_width(resolution_width),
    w_resolution_height(resolution_height),
    w_title(title),
    gl_window(initUI(title, w_width, w_height, false)),
    w_output_buffer(w_resolution_width, w_resolution_height),
    w_cam(make_camera(make_vec3(0,0,0), normalize_vec3(make_vec3(1,1,5./30.)), 1.7, w_resolution_width, w_resolution_height)),
    w_world(world)
    {
        glfwGetWindowPos(gl_window, &w_pos_x, &w_pos_y);
        w_output_buffer.setStream( 0u );
        init_callbacks();

        GLFWmonitor* monitor = glfwGetPrimaryMonitor();
        const GLFWvidmode * mode = glfwGetVideoMode(monitor);
        glfwSetWindowAspectRatio(gl_window,mode->width, mode->height); // 16:9
// glfwWindowHint(GLFW_TRANSPARENT_FRAMEBUFFER, GLFW_TRUE);
// glfwSetWindowAttrib(gl_window, GLFW_FLOATING, GLFW_TRUE);
// glfwSetWindowAttrib(gl_window, GLFW_FLOATING, GLFW_FALSE);
// glfwSetWindowAttrib(gl_window, GLFW_RESIZABLE, GLFW_FALSE);
// glfwSetWindowOpacity(gl_window, 0.5f);
        glfwWindowHint(GLFW_RED_BITS, mode->redBits);
        glfwWindowHint(GLFW_GREEN_BITS, mode->greenBits);
        glfwWindowHint(GLFW_BLUE_BITS, mode->blueBits);
        glfwWindowHint(GLFW_REFRESH_RATE, mode->refreshRate);
        glfwSetWindowSize(gl_window, mode->width/2, mode->height/2);
        // glViewport(0, 0, mode->width/2, mode->height/2);
        // update_resolution(mode->width/2, mode->height/2);
    }

window_t::~window_t()
{
    glfwDestroyWindow(gl_window);
    if (w_id == 0) glfwTerminate();
}

void window_t::init_callbacks()
{
    // glfwSetWindowFocusCallback( gl_window, windowFocusCallback );
    glfwSetWindowSizeCallback ( gl_window, windowSizeCallback  );
    // glfwSetWindowPosCallback  ( gl_window, windowPosCallback   );
    glfwSetMouseButtonCallback( gl_window, mouseButtonCallback );
    // glfwSetCursorPosCallback  ( gl_window, cursorPosCallback   );
    glfwSetKeyCallback        ( gl_window, keyCallback         );
    // glfwSetScrollCallback     ( gl_window, scrollCallback      );
    glfwSetWindowUserPointer  ( gl_window, &w_id               );
}

void window_t::toggle_fullscreen() {
    if (is_fullscreen())
    {
        // restore last window size and position
        glfwSetWindowMonitor(gl_window, NULL,  w_pos_x, w_pos_y, w_backup_width, w_backup_height, 0);
    }
    else
    {
        // backup window position and window size
        update_pos();
        glfwGetFramebufferSize(gl_window, &w_backup_width, &w_backup_height);

        GLFWmonitor* monitor = glfwGetPrimaryMonitor();

        // get resolution of monitor
        const GLFWvidmode * mode = glfwGetVideoMode(monitor);

        // switch to full screen
        glfwSetWindowMonitor(gl_window, monitor, 0, 0, mode->width, mode->height, 0);
    }
}

bool window_t::is_fullscreen()
{
    return glfwGetWindowMonitor( gl_window ) != NULL;
}

void window_t::update_resolution(int width, int height)
{
    w_resolution_width = width;
    w_resolution_height = height;

    compute_camera_derivative(&w_cam, w_resolution_width, w_resolution_height);
    w_output_buffer.resize(w_resolution_width, w_resolution_height);
}

void window_t::resize()
{
    glfwGetFramebufferSize(gl_window, &w_width, &w_height);
    glViewport( 0, 0, w_width, w_height);

#ifndef SCALED_TO_RESOLUTION
    // not update to autoscale
    update_resolution(w_width, w_height);
#endif
}

void window_t::update_pos()
{
    glfwGetWindowPos(gl_window, &w_pos_x, &w_pos_y);
}

#include "render/CUDA/cuda_kernel.cuh"
void window_t::loop() {
    glfwPollEvents();
    update_controls();

#ifndef CUDA_FRAME_FAST_SWITCH
    w_output_buffer.map();
#endif

#ifndef CUDA_FRAME_FAST_SWITCH
    w_output_buffer.unmap();
#endif

    cuda_make_frame(w_resolution_width, w_resolution_height, w_output_buffer.getDevicePointer(), w_world, w_cam);

    gl_pbo_env.render(
        w_resolution_width,
        w_resolution_height,
        w_width,
        w_height,
        w_output_buffer.getPBO()
    );

    CUDA_SYNC_CHECK();
    glfwSwapBuffers( gl_window );
}

void window_t::set_camera_pos(int dx, int dy) {
    setup_camera_pos(&w_cam, dx, dy, w_resolution_width, w_resolution_height);
}

void window_t::set_camera_ang(int dx, int dy) {
    setup_camera_ang(&w_cam, dx, -dy, w_resolution_width, w_resolution_height);
}

void window_t::set_pressed(int key, int action, int mods) {
    if (action == GLFW_REPEAT) return; // if (action == GLFW_PRESS || action == GLFW_RELEASE) {

    // if ( key == GLFW_MOUSE_BUTTON_LEFT )
    // {
    //     // w_controls.mouse_l = !w_controls.mouse_l;
    // }
    // else
    if (( key == GLFW_MOUSE_BUTTON_MIDDLE )
    || ( key == GLFW_MOUSE_BUTTON_LEFT && mods & GLFW_MOD_CONTROL )
    || ( key == GLFW_MOUSE_BUTTON_LEFT && w_controls.drag ) )
    {
        double xpos, ypos;
        glfwGetCursorPos( gl_window, &xpos, &ypos );
        w_controls.cursor_x = (int) xpos;
        w_controls.cursor_y = (int) ypos;

        w_controls.drag = !w_controls.drag;
    }
    else if ( key == GLFW_MOUSE_BUTTON_RIGHT )
    {
        // w_controls.mouse_r = !w_controls.mouse_r;
    }
    else if ( key == GLFW_KEY_W ) {
        w_controls.key_w = !w_controls.key_w;
    }
    else if ( key == GLFW_KEY_A ) {
        w_controls.key_a = !w_controls.key_a;
    }
    else if ( key == GLFW_KEY_S ) {
        w_controls.key_s = !w_controls.key_s;
    }
    else if ( key == GLFW_KEY_D ) {
        w_controls.key_d = !w_controls.key_d;
    }
    else if ( key == GLFW_KEY_ENTER && mods & GLFW_MOD_ALT && action == GLFW_PRESS ) {
        toggle_fullscreen();
    }
    else if ( key == GLFW_KEY_Q || key == GLFW_KEY_ESCAPE )
    {
        glfwSetWindowShouldClose( gl_window, true );
    }
}

void window_t::update_controls() {
    int dx = 0, dy = 0;
    if ( w_controls.key_w ) {
        dy++;
    }
    if ( w_controls.key_a ) {
        dx++;
    }
    if ( w_controls.key_s ) {
        dy--;
    }
    if ( w_controls.key_d ) {
        dx--;
    }
    if (dx || dy) set_camera_pos(dx, dy);

    if (w_controls.drag) {
        double xpos, ypos;
        glfwGetCursorPos( gl_window, &xpos, &ypos );

        dx = (int) xpos - w_controls.cursor_x;
        dy = (int) ypos - w_controls.cursor_y;

        if (dx || dy) set_camera_ang(dx, dy);
        w_controls.cursor_x = (int) xpos;
        w_controls.cursor_y = (int) ypos;
    }
}
