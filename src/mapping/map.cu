// #include <mapping/map.h>

// #include <mapping/const.c>
// #include <mapping/camera.c>

#include <stdio.h>
#include <mapping/omg.c>
#include <mapping/map.c>

__global__ void init_map_gpu_k(map_t* map) {
    *map = init_map();
}

__global__ void clear_map_gpu_k(map_t* map) {
    clear_map(map);
}

__global__ void add_atom_gpu_k(map_t* map, atom_t a) { add_atom(map, a); }
__global__ void add_material_gpu_k(map_t* map, material_t m) { add_material(map, m); }
__global__ void add_object_gpu_k(map_t* map, object_t o) { add_object(map, o); }
__global__ void add_geometry_gpu_k(map_t* map, geometry_t g) { add_geometry(map, g); }

#include <render/CUDA/cuda_kernel.h>
map_t* init_map_gpu()
{
    map_t* map = NULL;
    CUDA_CHECK( cudaMalloc((void**)&map, sizeof(map_t)) );
    init_map_gpu_k<<<1, 1, 1, 0>>>(map);

    return map;
}

void clear_map_gpu(map_t* map)
{
    clear_map_gpu_k<<<1, 1, 1, 0>>>(map);
    // sync?
    CUDA_CHECK( cudaFree((void*)map) );
}

void add_atom_gpu(map_t* map, atom_t a) { add_atom_gpu_k<<<1, 1, 1, 0>>>(map, a); }
void add_material_gpu(map_t* map, material_t m) { add_material_gpu_k<<<1, 1, 1, 0>>>(map, m); }
void add_object_gpu(map_t* map, object_t o) { add_object_gpu_k<<<1, 1, 1, 0>>>(map, o); }
void add_geometry_gpu(map_t* map, geometry_t g) { add_geometry_gpu_k<<<1, 1, 1, 0>>>(map, g); }
