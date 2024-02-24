
// #pragma once

// #include <cooperative_groups.h>

// namespace cg = cooperative_groups;

#include "cuda_kernel.cuh"
#include <stdio.h>
#include <stdlib.h>

#include <geometry.h>

__device__ void conv255(vec4& col)
{
  col.r = (col.r > 255) ? 255 : (col.r < 0) ? 0 : col.r;
  col.g = (col.g > 255) ? 255 : (col.g < 0) ? 0 : col.g;
  col.b = (col.b > 255) ? 255 : (col.b < 0) ? 0 : col.b;
}

#define BIG_GRID 1E1
#define SMALL_GRID 1

// у кого как лучше считает
// #define BLOCK_SIZE 8
#define BLOCK_SIZE 16
__global__ void simple_vbo_kernel(unsigned width, unsigned height, uchar4* gpu_frame_ptr, map_t* map, camera_t cam)
{
    // printf("x: %i, y: %i blockIdx(x: %d, y: %d, z: %d)(%d;%d;%d)\tthreadIdx(x: %d, y: %d, z: %d)(%d;%d;%d)\n",
    //   x, y, blockIdx.x, blockIdx.y, blockIdx.z, gridDim.x, gridDim.y, gridDim.z, threadIdx.x, threadIdx.y, threadIdx.z, blockDim.x, blockDim.y, blockDim.z);

    // cooperative_groups::thread_block cta = cooperative_groups::this_thread_block();
    // cooperative_groups::sync(cta);

    // compute frame point coords
    int x = threadIdx.x * gridDim.x + blockIdx.x;
    int y = threadIdx.y * gridDim.y + blockIdx.y;

    if (x >= width || y >= height) return; // остаток с блока 16х16
    // __syncthreads();

    vec4 pix_col = make_vec4(0,0,0,0);

    int dx = (width / 2 - x);
    int dy = (height / 2 - y);

    ray_t cray = make_ray(
      cam.position, normalize_vec3(sum_vec3_vec3(
        sum_vec3_vec3(
          mul_vec3_scalar(cam._dir_perp_x, dx * cam._camera_rays_dispersion),
          mul_vec3_scalar(cam._dir_perp_y, dy * cam._camera_rays_dispersion)
        ),
        mul_vec3_scalar(cam.direction, cam.projection_distance)
      ))
    );

    vec3 point;
    fract_t dist, closest_dist = INFINITY;

    unsigned item_it = 0;
    block_t* block;

    geometry_t* closest_obj = NULL;

    if (map->has_geometry) {
      // render geometry objects
      block = map->geometry.first_block;
      geometry_t* obj = (geometry_t*) block->items;

      while (block->count != item_it++) {
        switch (obj->gtype)
        {
          case GEOOBJ_RAY:
            dist = ray_intersect_ray_at(cray, obj->r);

            if (dist >= cam.projection_distance && dist < closest_dist) {
              closest_dist = dist;
              closest_obj = obj;
            }

            break;

          case GEOOBJ_GRID:
             dist = ray_intersect_plane_at(cray, obj->p);

            if (dist >= cam.projection_distance && dist < closest_dist) {
              point = sum_vec3_vec3(cray.position, mul_vec3_scalar(cray.direction, dist));

              if (fabs(fmodf(point.x, 10)) >= SMALL_GRID && fabs(fmodf(point.y, 10)) >= SMALL_GRID) break;
              if (fabs(fmodf(point.x, 100)) >= BIG_GRID && fabs(fmodf(point.y, 100)) >= BIG_GRID) break;

              closest_dist = dist;
              closest_obj = obj;
            }

            break;

          case GEOOBJ_SPHERE:
            dist = ray_intersect_sphere_at(cray, obj->s);

            if (dist >= cam.projection_distance && dist < closest_dist) {
              closest_dist = dist;
              closest_obj = obj;
            }

            break;

          case GEOOBJ_PLANE:
             dist = ray_intersect_plane_at(cray, obj->p);

            if (dist >= cam.projection_distance && dist < closest_dist) {
              closest_dist = dist;
              closest_obj = obj;
            }

            break;

          default:
            printf("unknown geometry object type #%i\n", obj->gtype);
        }

        // __syncthreads();

        if (block->next_block && block->count == item_it) {
          block = (block_t*) block->next_block;
          item_it = 0;
          obj = (geometry_t*) block->items;
        } else {
          obj++;
        }

      }
    }

    if (closest_obj) pix_col = closest_obj->color;
    uchar4* pixel = gpu_frame_ptr + (y * width + x);

    // запись занимает 0.18 от времени отрисовки кадра с одним объектом
    pixel->x = (int) pix_col.r;
    pixel->y = (int) pix_col.g;
    pixel->z = (int) pix_col.b;
}

void cuda_make_frame(unsigned width, unsigned height, uchar4* gpu_frame_ptr, map_t* map, camera_t& cam)
{
    // cudaMemcpyAsync(fb->gpu_frame_ptr, fb->buf, fb->buf_size, cudaMemcpyHostToDevice, 0);
    simple_vbo_kernel<<<dim3(width/BLOCK_SIZE + 1, height/BLOCK_SIZE + 1), dim3(BLOCK_SIZE, BLOCK_SIZE), 1, 0>>>(width, height, gpu_frame_ptr, map, cam);
    // cudaMemcpyAsync(fb->gpu_frame_ptr, fb->buf, fb->buf_size, cudaMemcpyDeviceToHost, 0);
    cudaStreamSynchronize(0);
}
