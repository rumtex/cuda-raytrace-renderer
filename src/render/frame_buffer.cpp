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

#include <render/frame_buffer.h>

frame_buffer_t::frame_buffer_t(unsigned width, unsigned height)
{
    // If using GL Interop, expect that the active device is also the display device.
    {
        int current_device, is_display_device;
        CUDA_CHECK( cudaGetDevice( &current_device ) );
        CUDA_CHECK( cudaDeviceGetAttribute( &is_display_device, cudaDevAttrKernelExecTimeout, current_device ) );
        if( !is_display_device )
        {
            throw   "GL interop is only available on display device, please use display device for optimal "
                    "performance.  Alternatively you can disable GL interop with --no-gl-interop and run with "
                    "degraded performance.";
        }
    }
    resize( width, height );
}

frame_buffer_t::~frame_buffer_t()
{
    try
    {
        {
            makeCurrent();
            unmap();
        }

        if( m_pbo != 0u )
        {
            GL_CHECK( glBindBuffer( GL_ARRAY_BUFFER, 0 ) );
            GL_CHECK( glDeleteBuffers( 1, &m_pbo ) );
        }
    }
    catch(const char* e)
    {
        printf("frame_buffer_t destructor caught exception: %s\n", e);
    }
}

void frame_buffer_t::resize( unsigned width, unsigned height )
{
    if( m_width == width && m_height == height )
        return;

    m_width  = width;
    m_height = height;

    {
        makeCurrent();
        if (m_pbo != 0x0) unmap();

        // GL buffer gets resized below
        GL_CHECK( glGenBuffers( 1, &m_pbo ) );
        GL_CHECK( glBindBuffer( GL_ARRAY_BUFFER, m_pbo ) );
        GL_CHECK( glBufferData( GL_ARRAY_BUFFER, sizeof(uchar4)*m_width*m_height, nullptr, GL_STREAM_DRAW ) );
        GL_CHECK( glBindBuffer( GL_ARRAY_BUFFER, 0u ) );

        CUDA_CHECK( cudaGraphicsGLRegisterBuffer(
                    &m_cuda_gfx_resource,
                    m_pbo,
                    cudaGraphicsMapFlagsWriteDiscard
                    ) );
        map();
    }
}

uchar4* frame_buffer_t::getDevicePointer() {
    return m_device_pixels;
}

uchar4* frame_buffer_t::map()
{
    makeCurrent();

    size_t buffer_size = 0u;
    CUDA_CHECK( cudaGraphicsMapResources( 1, &m_cuda_gfx_resource, m_stream ) );
    CUDA_CHECK( cudaGraphicsResourceGetMappedPointer(
        (void**)&m_device_pixels,
        &buffer_size,
        m_cuda_gfx_resource
    ));

    return m_device_pixels;
}


void frame_buffer_t::unmap()
{
    makeCurrent();
    CUDA_CHECK( cudaGraphicsUnmapResources( 1, &m_cuda_gfx_resource,  m_stream ) );
}


GLuint frame_buffer_t::getPBO()
{
    return m_pbo;
}
