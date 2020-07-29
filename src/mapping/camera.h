
#define CAMERA_RAYS_DISPERSION 0.00009
#define CAMERA_WIDTH_ANGLE 69.

typedef struct {
    vec3            position;
    vec3            direction; // normalized
    fract_t         projection_distance;

    // производные (to not recompute each ray)
    fract_t         _vert_ang;
    fract_t         _horiz_ang;
    fract_t         _camera_rays_dispersion;
    // vec3            _zero_frame_coord;
    vec3            _dir_perp_x;
    vec3            _dir_perp_y;
} camera_t;

camera_t make_camera(vec3 cpos, vec3 cdir, fract_t cdist, unsigned width, unsigned height);

void compute_camera_derivative(camera_t* cam, unsigned width, unsigned height);
void setup_camera_ang(camera_t* cam, int dx, int dy, unsigned width, unsigned height);
void setup_camera_pos(camera_t* cam, int dx, int dy, unsigned width, unsigned height);
