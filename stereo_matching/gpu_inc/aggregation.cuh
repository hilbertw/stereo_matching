#include "../global.h"


__global__ void cu_dp_L1(float *d_cost, float *d_L1, float *d_min_L1, int img_w, int img_h, int max_disp, int P1, int P2);

__global__ void cu_dp_L2(float *d_cost, float *d_L1, float *d_min_L1, int img_w, int img_h, int max_disp, int P1, int P2);

__global__ void cu_dp_L3(float *d_cost, float *d_L1, float *d_min_L1, int img_w, int img_h, int max_disp, int P1, int P2);

__global__ void cu_dp_L4(float *d_cost, float *d_L1, float *d_min_L1, int img_w, int img_h, int max_disp, int P1, int P2);

__global__ void aggregation(float *d_cost, float *d_L1, int img_w, int img_h, int max_disp);

__global__ void wta(float *d_cost, uchar *disparity, int img_w, int img_h, int max_disp, float ratio, uchar invalid);