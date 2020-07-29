#ifndef __OBJECTS_omg_H_
#define __OBJECTS_omg_H_

#include <geometry.h>

#ifndef __CUDACC__
#include <stdlib.h>
// #include <render/CUDA/cuda_kernel.h>
#endif // __CUDACC__

#if defined(__cplusplus)
extern "C" {
#endif

#include <mapping/const.h>
#include <mapping/camera.h>

// organized memory group
#include <mapping/omg.h>

#include <mapping/object.h>
#include <mapping/geometry.h>

typedef struct {
    omg_t      atoms;
    omg_t      materials;
    omg_t      objects;

    char       has_geometry;
    omg_t      geometry;
} map_t;

CROSS_COMPILING_OPTS map_t init_map();
CROSS_COMPILING_OPTS void clear_map(map_t* map);

CROSS_COMPILING_OPTS void add_atom(map_t* map, atom_t a);
CROSS_COMPILING_OPTS void add_material(map_t* map, material_t m);
CROSS_COMPILING_OPTS void add_object(map_t* map, object_t o);
CROSS_COMPILING_OPTS void add_geometry(map_t* map, geometry_t g);

map_t* init_map_gpu();
void clear_map_gpu(map_t*);
void add_atom_gpu(map_t* map, atom_t a);
void add_material_gpu(map_t* map, material_t m);
void add_object_gpu(map_t* map, object_t o);
void add_geometry_gpu(map_t* map, geometry_t g);
// #ifndef __CUDACC__
// #endif

#if defined(__cplusplus)
}
#endif

#endif //__OBJECTS_omg_H_
