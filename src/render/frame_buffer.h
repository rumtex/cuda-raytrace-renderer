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

#include <cuda_gl_interop.h>

#include <render/CUDA/cuda_kernel.h>

typedef cudaStream_t CUstream;

class frame_buffer_t
{
public:
    frame_buffer_t( unsigned width, unsigned height );
    ~frame_buffer_t();

    void setDevice( unsigned device_idx ) { m_device_idx = device_idx; }
    void setStream( CUstream stream    ) { m_stream     = stream;     }

    void resize( unsigned width, unsigned height );

    // Allocate or update device pointer as necessary for CUDA access
    uchar4* map();
    void unmap();

    unsigned&       width()  { return m_width;  }
    unsigned&       height() { return m_height; }

    // Get output buffer
    GLuint          getPBO();

    uchar4*         getDevicePointer();

private:
    void makeCurrent() { /*CUDA_CHECK( cudaSetDevice( m_device_idx ) );*/ }

    unsigned                    m_width             = 0u;
    unsigned                    m_height            = 0u;

    cudaGraphicsResource*       m_cuda_gfx_resource = nullptr;
    GLuint                      m_pbo               = 0u;
    uchar4*                     m_device_pixels     = nullptr;

    CUstream                    m_stream            = 0u;
    unsigned                    m_device_idx        = 0;
};
