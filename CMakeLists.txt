cmake_minimum_required(VERSION 3.5)
add_definitions(-std=c++11)

project(stereo_matching)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS_RELEASE "-std=c++11 -O3 -Wall -pthread -fopenmp")

find_package(OpenCV 2.4.13 REQUIRED)
find_package(CUDA REQUIRED)

FIND_PACKAGE( OpenMP REQUIRED)
if(OPENMP_FOUND)
message("OPENMP FOUND")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

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
    node.cpp
    cpu_src/BM.cpp
    cpu_src/cost.cpp
    cpu_src/SGM.cpp
    cpu_src/Solver.cpp
    cpu_src/utils.cpp
    cpu_src/roshelper.cpp
    gpu_src/aggregation.cu
    gpu_src/cost.cu
    gpu_src/post_filter.cu
    gpu_src/SGM.cu

    sky_detector/imageSkyDetector.cpp
)

target_link_libraries(
    sgm
    ${catkin_LIBRARIES}
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
)

