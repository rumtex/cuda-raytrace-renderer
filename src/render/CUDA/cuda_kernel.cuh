#include <mapping/map.h>

extern "C" void cuda_make_frame(unsigned width, unsigned height, uchar4* gpu_frame_ptr, map_t* gpu_world_ptr, camera_t& cam);
