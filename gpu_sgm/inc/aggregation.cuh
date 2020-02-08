#include "../../inc/global.h"
#include "cuda_inc.cuh"


__global__ void cu_dp_L1(float *d_cost, float *d_dp, float *d_dp_min, int img_w, int img_h, int max_disp, int P1, int P2);

__global__ void cu_dp_L2(float *d_cost, float *d_dp, float *d_dp_min, int img_w, int img_h, int max_disp, int P1, int P2);

__global__ void cu_dp_L3(float *d_cost, float *d_dp, float *d_dp_min, int img_w, int img_h, int max_disp, int P1, int P2);

__global__ void cu_dp_L4(float *d_cost, float *d_dp, float *d_dp_min, int img_w, int img_h, int max_disp, int P1, int P2);

__global__ void cu_dp_L5(float *d_cost, short *d_dp, float *d_dp_min, int idx, int img_w, int img_h, int max_disp, int P1, int P2);

__global__ void cu_dp_L6(float *d_cost, short *d_dp, float *d_dp_min, int idx, int img_w, int img_h, int max_disp, int P1, int P2);

__global__ void cu_dp_L7(float *d_cost, short *d_dp, float *d_dp_min, int idx, int img_w, int img_h, int max_disp, int P1, int P2);

__global__ void cu_dp_L8(float *d_cost, short *d_dp, float *d_dp_min, int idx, int img_w, int img_h, int max_disp, int P1, int P2);

__global__ void aggregation(float *d_cost_sum, float *d_L1, float *d_L2, float *d_L3, float *d_L4,
                            short *d_L5, short *d_L6, short *d_L7, short *d_L8,
                            int img_w, int img_h, int max_disp);

__global__ void wta(float *d_cost_sum, uchar *disparity, int img_w, int img_h, int max_disp, float ratio, int invalid);

__global__ void cu_dp_L5_truncated(float *d_cost, short *d_dp, float *d_dp_min, int img_w, int img_h, int max_disp, int P1, int P2);

__global__ void cu_dp_L6_truncated(float *d_cost, short *d_dp, float *d_dp_min, int img_w, int img_h, int max_disp, int P1, int P2);

__global__ void cu_dp_L7_truncated(float *d_cost, short *d_dp, float *d_dp_min, int img_w, int img_h, int max_disp, int P1, int P2);

__global__ void cu_dp_L8_truncated(float *d_cost, short *d_dp, float *d_dp_min, int img_w, int img_h, int max_disp, int P1, int P2);
