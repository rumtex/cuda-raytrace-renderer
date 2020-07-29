#include <mapping/map.h>

const_t c;

void init_const() {
    // vectors
    c.zero      = make_vec3(0,0,0);
    c.x_axis    = make_vec3(1,0,0);
    c.y_axis    = make_vec3(0,1,0);
    c.z_axis    = make_vec3(0,0,1);

    // colors
    c.black = make_vec3(0, 0, 0);
    c.red   = make_vec3(255, 0, 0);
    c.green = make_vec3(0, 255, 0);
    c.blue  = make_vec3(0, 0, 255);
    c.white = make_vec3(255, 255, 255);
}

// когда сотня потоков тянется в одно место в памяти ГПУ - все оч плохо
// #include <render/CUDA/cuda_kernel.h>
// void load_const_to_gpu() {
//     CUDA_CHECK( cudaSetDevice(0) );
//     printf("%lu\n", sizeof(const_t));
//     CUDA_CHECK( cudaMalloc((void**)&c_gpu_ptr, sizeof(const_t)) );
//     CUDA_CHECK( cudaMemcpyAsync(c_gpu_ptr, &c, sizeof(const_t), cudaMemcpyHostToDevice, 0) );
// }
