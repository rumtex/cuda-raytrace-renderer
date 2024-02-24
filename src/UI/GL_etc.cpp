//
// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

#include <UI/GL.h>
#include <UI/GL_etc.h>
#include <iostream>

static void errorCallback( int error, const char* description )
{
    std::cerr << "GLFW Error " << error << ": " << description << std::endl;
}

void initGL()
{
    glfwSwapInterval(0);
    printf("OpenGL version (%s): \n", glGetString(GL_VERSION));

    GL_CHECK( glClearColor( 0.212f, 0.271f, 0.31f, 1.0f ) );
    GL_CHECK( glClear( GL_COLOR_BUFFER_BIT ) );
}

GLFWwindow* initGLFW( const char* window_title, int width, int height, bool fullscreen)
{
    GLFWwindow* window = nullptr;
    glfwSetErrorCallback( errorCallback );

    if( !glfwInit() )
        throw "Failed to initialize GLFW";

    glfwWindowHint( GLFW_CONTEXT_VERSION_MAJOR, 3 );
    glfwWindowHint( GLFW_CONTEXT_VERSION_MINOR, 3 );
    glfwWindowHint( GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE );  // To make Apple happy -- should not be needed
    glfwWindowHint( GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE );

    GLFWmonitor* monitor = NULL;
    unsigned res_width, res_height;
    if (fullscreen) {
        // int count;
        // GLFWmonitor** monitors = glfwGetMonitors(&count);
        monitor = glfwGetPrimaryMonitor();

        const GLFWvidmode* mode = glfwGetVideoMode(monitor);
        res_width = mode->width;
        res_height = mode->height;


        printf("%u %u\n", res_width, res_height);
        glfwWindowHint(GLFW_RED_BITS, mode->redBits);
        glfwWindowHint(GLFW_GREEN_BITS, mode->greenBits);
        glfwWindowHint(GLFW_BLUE_BITS, mode->blueBits);
        glfwWindowHint(GLFW_REFRESH_RATE, mode->refreshRate);
    }

    window = glfwCreateWindow(
        fullscreen ? res_width : width,
        fullscreen ? res_height : height,
        window_title, monitor, NULL );

    if( !window )
        throw "Failed to create GLFW window";

    glfwMakeContextCurrent( window );
    glfwSwapInterval( 0 );  // No vsync

    return window;
}

// будет недостаточно
// volatile bool initiated = false;
//     if (!initiated)
//     initiated = true;
GLFWwindow* initUI(const char* window_title, int width, int height, bool fullscreen)
{
    initGL();
    GLFWwindow* window = initGLFW(window_title, width, height, fullscreen);
    // initImGui( window );
    return window;
}

GLuint createGLShader( const std::string& source, GLuint shader_type )
{
    GLuint shader = glCreateShader( shader_type );
    {
        const GLchar* source_data= reinterpret_cast<const GLchar*>( source.data() );
        glShaderSource( shader, 1, &source_data, nullptr );
        glCompileShader( shader );

        GLint is_compiled = 0;
        glGetShaderiv( shader, GL_COMPILE_STATUS, &is_compiled );
        if( is_compiled == GL_FALSE )
        {
            GLint max_length = 0;
            glGetShaderiv( shader, GL_INFO_LOG_LENGTH, &max_length );

            std::string info_log( max_length, '\0' );
            GLchar* info_log_data= reinterpret_cast<GLchar*>( &info_log[0]);
            glGetShaderInfoLog( shader, max_length, nullptr, info_log_data );

            glDeleteShader(shader);
            std::cerr << "Compilation of shader failed: " << info_log << std::endl;

            return 0;
        }
    }

    GL_CHECK_ERRORS();

    return shader;
}


GLuint createGLProgram(
        const std::string& vert_source,
        const std::string& frag_source
        )
{
    GLuint vert_shader = createGLShader( vert_source, GL_VERTEX_SHADER );
    if( vert_shader == 0 )
        return 0;

    GLuint frag_shader = createGLShader( frag_source, GL_FRAGMENT_SHADER );
    if( vert_shader == 0 )
    {
        glDeleteShader( vert_shader );
        return 0;
    }

    GLuint program = glCreateProgram();
    glAttachShader( program, vert_shader );
    glAttachShader( program, frag_shader );
    glLinkProgram( program );

    GLint is_linked = 0;
    glGetProgramiv( program, GL_LINK_STATUS, &is_linked );
    if (is_linked == GL_FALSE)
    {
        GLint max_length = 0;
        glGetProgramiv( program, GL_INFO_LOG_LENGTH, &max_length );

        std::string info_log( max_length, '\0' );
        GLchar* info_log_data= reinterpret_cast<GLchar*>( &info_log[0]);
        glGetProgramInfoLog( program, max_length, nullptr, info_log_data );
        std::cerr << "Linking of program failed: " << info_log << std::endl;

        glDeleteProgram( program );
        glDeleteShader( vert_shader );
        glDeleteShader( frag_shader );

        return 0;
    }

    glDetachShader( program, vert_shader );
    glDetachShader( program, frag_shader );

    GL_CHECK_ERRORS();

    return program;
}


GLint getGLUniformLocation( GLuint program, const std::string& name )
{
    GLint loc = glGetUniformLocation( program, name.c_str() );
    ASSERT_MSG( loc != -1, ("Failed to get uniform loc for '" + name + "'").c_str() );
    return loc;
}

//-----------------------------------------------------------------------------
//
// GLPixelBufferObjectEnvironment implementation
//
//-----------------------------------------------------------------------------

const std::string GLPixelBufferObjectEnvironment::s_vert_source = R"(
#version 330 core

layout(location = 0) in vec3 vertexPosition_modelspace;
out vec2 UV;

void main()
{
    gl_Position =  vec4(vertexPosition_modelspace,1);
    UV = (vec2( vertexPosition_modelspace.x, vertexPosition_modelspace.y )+vec2(1,1))/2.0;
}
)";

const std::string GLPixelBufferObjectEnvironment::s_frag_source = R"(
#version 330 core

in vec2 UV;
out vec3 color;

uniform sampler2D render_tex;
uniform bool correct_gamma;

void main()
{
    color = texture( render_tex, UV ).xyz;
}
)";



GLPixelBufferObjectEnvironment::GLPixelBufferObjectEnvironment()
{
    GLuint m_vertex_array;
    GL_CHECK( glGenVertexArrays(1, &m_vertex_array ) );
    GL_CHECK( glBindVertexArray( m_vertex_array ) );

    m_program = createGLProgram( s_vert_source, s_frag_source );
    m_render_tex_uniform_loc = getGLUniformLocation( m_program, "render_tex");

    GL_CHECK( glGenTextures( 1, &m_render_tex ) );
    GL_CHECK( glBindTexture( GL_TEXTURE_2D, m_render_tex ) );

    GL_CHECK( glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST ) );
    GL_CHECK( glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST ) );
    GL_CHECK( glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE ) );
    GL_CHECK( glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE ) );

    static const GLfloat g_quad_vertex_buffer_data[] = {
        -1.0f, -1.0f, 0.0f,
         1.0f, -1.0f, 0.0f,
        -1.0f,  1.0f, 0.0f,

        -1.0f,  1.0f, 0.0f,
         1.0f, -1.0f, 0.0f,
         1.0f,  1.0f, 0.0f,
    };

    GL_CHECK( glGenBuffers( 1, &m_quad_vertex_buffer ) );
    GL_CHECK( glBindBuffer( GL_ARRAY_BUFFER, m_quad_vertex_buffer ) );
    GL_CHECK( glBufferData( GL_ARRAY_BUFFER,
            sizeof( g_quad_vertex_buffer_data),
            g_quad_vertex_buffer_data,
            GL_STATIC_DRAW
            )
        );

    GL_CHECK_ERRORS();
}


void GLPixelBufferObjectEnvironment::render(
        const unsigned& screen_res_x,
        const unsigned& screen_res_y,
        const unsigned& framebuf_res_x,
        const unsigned& framebuf_res_y,
        const uint32_t pbo
        ) const
{
    GL_CHECK( glBindFramebuffer( GL_FRAMEBUFFER, 0 ) );
    GL_CHECK( glViewport( 0, 0, framebuf_res_x, framebuf_res_y ) );

    GL_CHECK( glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT ) );

    GL_CHECK( glUseProgram( m_program ) );

    // Bind our texture in Texture Unit 0
    GL_CHECK( glActiveTexture( GL_TEXTURE0 ) );
    GL_CHECK( glBindTexture( GL_TEXTURE_2D, m_render_tex ) );
    GL_CHECK( glBindBuffer( GL_PIXEL_UNPACK_BUFFER, pbo ) );

    GL_CHECK( glPixelStorei(GL_UNPACK_ALIGNMENT, 4) );
    glPixelStorei(GL_UNPACK_ALIGNMENT, 4);

    glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA8, screen_res_x, screen_res_y, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr );

    GL_CHECK( glBindBuffer( GL_PIXEL_UNPACK_BUFFER, 0 ) );
    GL_CHECK( glUniform1i( m_render_tex_uniform_loc , 0 ) );

    // 1st attribute buffer : vertices
    GL_CHECK( glEnableVertexAttribArray( 0 ) );
    GL_CHECK( glBindBuffer(GL_ARRAY_BUFFER, m_quad_vertex_buffer ) );
    GL_CHECK( glVertexAttribPointer(
            0,                  // attribute 0. No particular reason for 0, but must match the layout in the shader.
            3,                  // size
            GL_FLOAT,           // type
            GL_FALSE,           // normalized?
            0,                  // stride
            (void*)0            // array buffer offset
            )
        );

    // Draw the triangles !
    GL_CHECK( glDrawArrays(GL_TRIANGLES, 0, 6) ); // 2*3 indices starting at 0 -> 2 triangles

    GL_CHECK( glDisableVertexAttribArray(0) );

    GL_CHECK_ERRORS();
}
