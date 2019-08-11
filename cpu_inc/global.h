#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <stdint.h>
#include <omp.h>
#include <unistd.h>
#include <tr1/memory>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <ros/ros.h>
#include <sensor_msgs/PointCloud.h>
// #include <sensor_msgs/image_encodings.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>

#include <cuda_runtime.h>
#define checkCudaErrors( a ) do { \
    if (cudaSuccess != (a)) { \
    fprintf(stderr, "Cuda runtime error in line %d of file %s \
    : %s \n", __LINE__, __FILE__, cudaGetErrorString(cudaGetLastError()) ); \
    } \
    } while(0);

using namespace cv;

const int SCALE = 1;

// kitti config
const int IMG_W = 1240 / SCALE;
const int IMG_H = 360 / SCALE;
const int MAX_DISP = 128;

//// middlebury config
//const int IMG_W = 450 / SCALE;
//const int IMG_H = 375 / SCALE;
//const int MAX_DISP = 64;

//// my personal stereo config
//const int IMG_W = 1280 / SCALE;
//const int IMG_H = 720 / SCALE;
//const int MAX_DISP = 64;

const int INVALID_DISP = MAX_DISP + 1;

const bool USE_GPU = 1;
