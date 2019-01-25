cmake_minimum_required(VERSION 3.5)
add_definitions(-std=c++11)

project(stereo)

find_package(OpenCV 2.4.13 REQUIRED)
find_package(CUDA REQUIRED)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    cpu_inc
    gpu_inc
)

cuda_add_executable(
    sgm
    main.cpp
    cpu_src/BM.cpp
    cpu_src/cost.cpp
    cpu_src/SGM.cpp
    cpu_src/Solver.cpp
    cpu_src/utils.cpp
    gpu_src/aggregation.cu
    gpu_src/cost.cu
    gpu_src/post_filter.cu
    gpu_src/SGM.cu
)

target_link_libraries(
    sgm
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
)
