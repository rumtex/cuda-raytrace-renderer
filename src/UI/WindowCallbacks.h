
//------------------------------------------------------------------------------
//
// GLFW callbacks
//
//------------------------------------------------------------------------------

static void mouseButtonCallback( GLFWwindow* window, int button, int action, int mods )
{
    window_t* m_window = (window_t*) glfwGetWindowUserPointer( window );
    m_window->set_pressed(button, action, mods);
}

static void cursorPosCallback( GLFWwindow* window, double xpos, double ypos )
{
    // printf("cursorPosCallback %f %f\n", xpos, ypos);
    // Params* params = static_cast<Params*>( glfwGetWindowUserPointer( window ) );

    // if( mouse_button == GLFW_MOUSE_BUTTON_LEFT )
    // {
    //     trackball.setViewMode( sutil::Trackball::LookAtFixed );
    //     trackball.updateTracking( static_cast<int>( xpos ), static_cast<int>( ypos ), params->width, params->height );
    //     camera_changed = true;
    // }
    // else if( mouse_button == GLFW_MOUSE_BUTTON_RIGHT )
    // {
    //     trackball.setViewMode( sutil::Trackball::EyeFixed );
    //     trackball.updateTracking( static_cast<int>( xpos ), static_cast<int>( ypos ), params->width, params->height );
    //     camera_changed = true;
    // }
}

void keyCallback(GLFWwindow* window, int key, int /*scancode*/, int action, int mods)
{
    window_t* m_window = (window_t*) glfwGetWindowUserPointer( window );
    m_window->set_pressed(key, action, mods);
}

void windowSizeCallback( GLFWwindow* window, int res_x, int res_y )
{
    window_t* m_window = (window_t*) glfwGetWindowUserPointer( window );
    m_window->resize();
}

void windowPosCallback(GLFWwindow* window, int pos_x, int pos_y) {
    window_t* m_window = (window_t*) glfwGetWindowUserPointer( window );
    m_window->update_pos();
}

void windowFocusCallback(GLFWwindow* window, int focused) {
    printf("window focus %i\n", focused);
}

void scrollCallback( GLFWwindow* window, double xscroll, double yscroll )
{
    printf("scrollCallback\n");
}
