#include <cuda_runtime.h>
#include <stdio.h>

#define CUDA_SYNC_CHECK()                                                      \
    {                                                                          \
        cudaDeviceSynchronize();                                               \
        cudaError_t error = cudaGetLastError();                                \
        if( error != cudaSuccess )                                             \
        {                                                                      \
            printf("CUDA error on synchronize with error '%s' (%s:%i)\n",      \
               cudaGetErrorString( error ),                                    \
               __FILE__, __LINE__ );                                           \
        }                                                                      \
    }

#define CUDA_CHECK( call )                                                     \
    {                                                                          \
        cudaError_t error = call;                                              \
        if( error != cudaSuccess )                                             \
        {                                                                      \
            printf("CUDA  call (%s) failed with error: '%s' (%s:%i)\n",        \
                #call,                                                         \
                cudaGetErrorString( error ),                                   \
                __FILE__, __LINE__ );                                          \
        }                                                                      \
    }
