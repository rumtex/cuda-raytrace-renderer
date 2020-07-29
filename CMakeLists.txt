cmake_minimum_required(VERSION 3.5)

option(DEBUG_LOG   "Debug logs"                ON)

SET(CMAKE_C_COMPILER /usr/bin/gcc-8)
SET(CMAKE_CXX_COMPILER /usr/bin/g++-8)
SET(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

# Set the project name
set(PROJECT cuda-raytracer)
project(${PROJECT} C CXX CUDA)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Wall")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -ccbin g++")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

###
# sources
###
file(GLOB_RECURSE SRC_CUDA src/*.cu)
file(GLOB_RECURSE SRC_CUDA_LIB lib/*.cu)
file(GLOB_RECURSE SRC_CXX src/*.cpp)
file(GLOB_RECURSE SRC_C src/*.c)
file(GLOB_RECURSE SRC_C_LIB lib/*.c)

# binary
add_executable(${PROJECT}
    ${SRC_CUDA}
    ${SRC_CUDA_LIB}
    # ${SRC_C}
    src/mapping/const.c
    src/mapping/camera.c
    # ${SRC_C_LIB} #CUDA капризничает
    ${SRC_CXX}
)

# deps
find_package(OpenGL REQUIRED)
# find_package(GLFW3 REQUIRED)

target_link_libraries(${PROJECT}
    pthread
    stdc++
    m
    cudart
    ${OPENGL_gl_LIBRARY}
    glfw
)

# -I/directory/path/
target_include_directories(${PROJECT}
    PRIVATE
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/lib
        /usr/local/cuda/targets/x86_64-linux/include
)

# We need to explicitly state that we need all CUDA files to be built with -dc as the member functions could be called by
# other libraries and executables
set_target_properties(${PROJECT}
    PROPERTIES CUDA_SEPARABLE_COMPILATION ON
)

if (DEBUG_LOG)
    ADD_DEFINITIONS(-DDEBUG)
endif()
