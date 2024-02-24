#include <stdio.h>
#include <chrono>

std::chrono::_V2::system_clock::time_point a_time = std::chrono::high_resolution_clock::now();
std::chrono::_V2::system_clock::time_point b_time;

unsigned ms, frame_it = 0;
constexpr unsigned frame_limit_to_calc = 100;
constexpr float ms_frames_to_div = (frame_limit_to_calc * 4 * 60. * 4000.);

void update_fps() {
    if (++frame_it == frame_limit_to_calc) {
        b_time = std::chrono::high_resolution_clock::now();
        ms = std::chrono::duration_cast<std::chrono::microseconds>(b_time - a_time).count();
        frame_it = 0;
        a_time = b_time;
        printf("FPS: %5.2f\n", (ms_frames_to_div/ms));
    }
}