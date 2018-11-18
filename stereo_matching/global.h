#include <iostream>
#include <fstream>
#include <sstream>
#include <stdint.h>
#include <omp.h>

#include <core/core.hpp>
#include <highgui/highgui.hpp>
#include <imgproc\imgproc.hpp>
#include <calib3d\calib3d.hpp>

#include <cuda_runtime.h>

//#define checkCudaErrors( a ) do { \
//    if (cudaSuccess != (a)) { \
//    fprintf(stderr, "Cuda runtime error in line %d of file %s \
//    : %s \n", __LINE__, __FILE__, cudaGetErrorString(cudaGetLastError()) ); \
//	std::cin.get(); \
//    exit(EXIT_FAILURE); \
//    } \
//    } while(0);

#define checkCudaErrors( a ) do { \
    if (cudaSuccess != (a)) { \
    fprintf(stderr, "Cuda runtime error in line %d of file %s \
    : %s \n", __LINE__, __FILE__, cudaGetErrorString(cudaGetLastError()) ); \
    } \
    } while(0);

using namespace cv;