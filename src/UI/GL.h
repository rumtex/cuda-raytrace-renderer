#ifndef __GRAPHIC_H_
#define __GRAPHIC_H_

#include <GLFW/glfw3.h>

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
    #include <GL/glew.h>
#endif

#if defined(__APPLE__) || defined(MACOSX)
    #include <OpenGL/gl.h>
    #pragma clang diagnostic ignored "-Wdeprecated-declarations"
#else
    #include <GL/gl.h>
    #ifdef __linux__
    #include <GL/glx.h>


    #ifdef HELPERGL_EXTERN_GL_FUNC_IMPLEMENTATION
    #define USE_GL_FUNC(name, proto) proto name = (proto) glXGetProcAddress ((const GLubyte *)#name)
    #else
    #define USE_GL_FUNC(name, proto) extern proto name
    #endif

    USE_GL_FUNC(glBindBuffer, PFNGLBINDBUFFERPROC);
    USE_GL_FUNC(glDeleteBuffers, PFNGLDELETEBUFFERSPROC);
    USE_GL_FUNC(glBufferData, PFNGLBUFFERDATAPROC);
    USE_GL_FUNC(glBufferSubData, PFNGLBUFFERSUBDATAPROC);
    USE_GL_FUNC(glGenBuffers, PFNGLGENBUFFERSPROC);

    USE_GL_FUNC(glGenVertexArrays, PFNGLGENVERTEXARRAYSPROC);
    USE_GL_FUNC(glBindVertexArray, PFNGLBINDVERTEXARRAYPROC);
    USE_GL_FUNC(glEnableVertexAttribArray, PFNGLENABLEVERTEXATTRIBARRAYPROC);
    USE_GL_FUNC(glVertexAttribPointer, PFNGLVERTEXATTRIBPOINTERPROC);
    USE_GL_FUNC(glDisableVertexAttribArray, PFNGLDISABLEVERTEXATTRIBARRAYPROC);

    USE_GL_FUNC(glCreateProgram, PFNGLCREATEPROGRAMPROC);
    USE_GL_FUNC(glBindProgramARB, PFNGLBINDPROGRAMARBPROC);
    USE_GL_FUNC(glGenProgramsARB, PFNGLGENPROGRAMSARBPROC);
    USE_GL_FUNC(glDeleteProgramsARB, PFNGLDELETEPROGRAMSARBPROC);
    USE_GL_FUNC(glDeleteProgram, PFNGLDELETEPROGRAMPROC);
    USE_GL_FUNC(glGetProgramInfoLog, PFNGLGETPROGRAMINFOLOGPROC);
    USE_GL_FUNC(glGetProgramiv, PFNGLGETPROGRAMIVPROC);
    USE_GL_FUNC(glProgramParameteriEXT, PFNGLPROGRAMPARAMETERIEXTPROC);
    USE_GL_FUNC(glProgramStringARB, PFNGLPROGRAMSTRINGARBPROC);
    USE_GL_FUNC(glUnmapBuffer, PFNGLUNMAPBUFFERPROC);
    USE_GL_FUNC(glMapBuffer, PFNGLMAPBUFFERPROC);
    USE_GL_FUNC(glGetBufferParameteriv, PFNGLGETBUFFERPARAMETERIVPROC);
    USE_GL_FUNC(glLinkProgram, PFNGLLINKPROGRAMPROC);
    USE_GL_FUNC(glUseProgram, PFNGLUSEPROGRAMPROC);
    USE_GL_FUNC(glAttachShader, PFNGLATTACHSHADERPROC);
    USE_GL_FUNC(glCreateShader, PFNGLCREATESHADERPROC);
    USE_GL_FUNC(glShaderSource, PFNGLSHADERSOURCEPROC);
    USE_GL_FUNC(glCompileShader, PFNGLCOMPILESHADERPROC);
    USE_GL_FUNC(glDeleteShader, PFNGLDELETESHADERPROC);
    USE_GL_FUNC(glDetachShader, PFNGLDETACHSHADERPROC);
    USE_GL_FUNC(glGetShaderInfoLog, PFNGLGETSHADERINFOLOGPROC);
    USE_GL_FUNC(glGetShaderiv, PFNGLGETSHADERIVPROC);
    USE_GL_FUNC(glUniform1i, PFNGLUNIFORM1IPROC);
    USE_GL_FUNC(glUniform1f, PFNGLUNIFORM1FPROC);
    USE_GL_FUNC(glUniform2f, PFNGLUNIFORM2FPROC);
    USE_GL_FUNC(glUniform3f, PFNGLUNIFORM3FPROC);
    USE_GL_FUNC(glUniform4f, PFNGLUNIFORM4FPROC);
    USE_GL_FUNC(glUniform1fv, PFNGLUNIFORM1FVPROC);
    USE_GL_FUNC(glUniform2fv, PFNGLUNIFORM2FVPROC);
    USE_GL_FUNC(glUniform3fv, PFNGLUNIFORM3FVPROC);
    USE_GL_FUNC(glUniform4fv, PFNGLUNIFORM4FVPROC);
    USE_GL_FUNC(glUniformMatrix4fv, PFNGLUNIFORMMATRIX4FVPROC);
    USE_GL_FUNC(glSecondaryColor3fv, PFNGLSECONDARYCOLOR3FVPROC);
    USE_GL_FUNC(glGetUniformLocation, PFNGLGETUNIFORMLOCATIONPROC);
    USE_GL_FUNC(glGenFramebuffers, PFNGLGENFRAMEBUFFERSEXTPROC);
    USE_GL_FUNC(glBindFramebuffer, PFNGLBINDFRAMEBUFFEREXTPROC);
    USE_GL_FUNC(glDeleteFramebuffers, PFNGLDELETEFRAMEBUFFERSEXTPROC);
    USE_GL_FUNC(glCheckFramebufferStatus, PFNGLCHECKFRAMEBUFFERSTATUSEXTPROC);
    USE_GL_FUNC(glGetFramebufferAttachmentParameteriv, PFNGLGETFRAMEBUFFERATTACHMENTPARAMETERIVEXTPROC);
    USE_GL_FUNC(glFramebufferTexture1D, PFNGLFRAMEBUFFERTEXTURE1DEXTPROC);
    USE_GL_FUNC(glFramebufferTexture2D, PFNGLFRAMEBUFFERTEXTURE2DEXTPROC);
    USE_GL_FUNC(glFramebufferTexture3D, PFNGLFRAMEBUFFERTEXTURE3DEXTPROC);
    USE_GL_FUNC(glGenerateMipmap, PFNGLGENERATEMIPMAPEXTPROC);
    USE_GL_FUNC(glGenRenderbuffers, PFNGLGENRENDERBUFFERSEXTPROC);
    USE_GL_FUNC(glDeleteRenderbuffers, PFNGLDELETERENDERBUFFERSEXTPROC);
    USE_GL_FUNC(glBindRenderbuffer, PFNGLBINDRENDERBUFFEREXTPROC);
    USE_GL_FUNC(glRenderbufferStorage, PFNGLRENDERBUFFERSTORAGEEXTPROC);
    USE_GL_FUNC(glFramebufferRenderbuffer, PFNGLFRAMEBUFFERRENDERBUFFEREXTPROC);
    USE_GL_FUNC(glClampColorARB, PFNGLCLAMPCOLORARBPROC);
    USE_GL_FUNC(glBindFragDataLocation, PFNGLBINDFRAGDATALOCATIONEXTPROC);

    #endif /* __linux__ */
#endif

#define DO_GL_CHECK
#ifdef DO_GL_CHECK

inline const char* getGLErrorString( GLenum error )
{
    switch( error )
    {
        case GL_NO_ERROR:            return "No error";
        case GL_INVALID_ENUM:        return "Invalid enum";
        case GL_INVALID_VALUE:       return "Invalid value";
        case GL_INVALID_OPERATION:   return "Invalid operation";
        //case GL_STACK_OVERFLOW:      return "Stack overflow";
        //case GL_STACK_UNDERFLOW:     return "Stack underflow";
        case GL_OUT_OF_MEMORY:       return "Out of memory";
        //case GL_TABLE_TOO_LARGE:     return "Table too large";
        default:                     return "Unknown GL error";
    }
}

#    define GL_CHECK( call )                                                   \
        do                                                                     \
        {                                                                      \
            call;                                                              \
            GLenum err = glGetError();                                         \
            if( err != GL_NO_ERROR )                                           \
            {                                                                  \
            printf("CUDA error on synchronize with error '%s' (%s:%i)\n",      \
               getGLErrorString( err ),                                        \
               __FILE__, __LINE__ );                                           \
                throw getGLErrorString( err );                                 \
            }                                                                  \
        }                                                                      \
        while (0)


#    define GL_CHECK_ERRORS( )                                                 \
        do                                                                     \
        {                                                                      \
            GLenum err = glGetError();                                         \
            if( err != GL_NO_ERROR )                                           \
            {                                                                  \
                printf("CUDA error on synchronize with error '%s' (%s:%i)\n",  \
                getGLErrorString( err ),                                       \
                __FILE__, __LINE__ );                                          \
                throw getGLErrorString( err );                                 \
            }                                                                  \
        }                                                                      \
        while (0)

#else
#    define GL_CHECK( call )   do { call; } while(0)
#    define GL_CHECK_ERRORS( ) do { ;     } while(0)
#endif

#define ASSERT_MSG( cond, msg )                                          \
    do                                                                         \
    {                                                                          \
        if( !(cond) )                                                          \
        {                                                                      \
            printf("CUDA error on synchronize with error '%s' (%s:%i)\n",      \
               msg,                                                            \
               __FILE__, __LINE__ );                                           \
            throw msg;                                                         \
        }                                                                      \
    } while( 0 )

#endif //__GRAPHIC_H_

