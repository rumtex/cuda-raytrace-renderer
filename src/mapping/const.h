
typedef struct {
    vec3            zero;
    vec3            x_axis;
    vec3            y_axis;
    vec3            z_axis;

    vec3            black;
    vec3            red;
    vec3            green;
    vec3            blue;
    vec3            white;
} const_t;

extern const_t c;

void init_const();
