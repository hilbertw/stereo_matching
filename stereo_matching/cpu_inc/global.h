#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <stdint.h>
#include <omp.h>
#include <windows.h>

#include <core/core.hpp>
#include <highgui/highgui.hpp>
#include <imgproc/imgproc.hpp>
#include <calib3d/calib3d.hpp>

#include <cuda_runtime.h>
#define checkCudaErrors( a ) do { \
    if (cudaSuccess != (a)) { \
    fprintf(stderr, "Cuda runtime error in line %d of file %s \
    : %s \n", __LINE__, __FILE__, cudaGetErrorString(cudaGetLastError()) ); \
    } \
    } while(0);

using namespace cv;

const int SCALE = 2;
//const int IMG_W = 1240 / SCALE;
//const int IMG_H = 360 / SCALE;
//const int MAX_DISP = 128;

const int IMG_W = 1280 / SCALE;
const int IMG_H = 720 / SCALE;
const int MAX_DISP = 128;

//// middlebury config
//const int IMG_W = 450 / SCALE;
//const int IMG_H = 375 / SCALE;
//const int MAX_DISP = 64;

const int INVALID_DISP = MAX_DISP + 1;

const bool USE_GPU = 1;
