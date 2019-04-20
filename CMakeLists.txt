cmake_minimum_required(VERSION 3.5)
add_definitions(-std=c++11)

project(stereo_matching)

find_package(OpenCV 2.4.13 REQUIRED)
find_package(CUDA REQUIRED)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  image_transport
  roscpp
  std_msgs
  tf
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES stereo_matching
#  CATKIN_DEPENDS roscpp std_msgs
#  DEPENDS system_lib
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
  cpu_inc
  gpu_inc
)

cuda_add_executable(
    sgm
    # demo1.cpp
    demo2.cpp
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
    ${catkin_LIBRARIES}
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
)

