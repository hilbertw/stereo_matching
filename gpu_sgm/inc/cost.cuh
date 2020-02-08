#include "../../inc/global.h"
#include "cuda_inc.cuh"


__global__ void cu_build_cost_table(uchar *d_img_l, uchar *d_img_r,
                                    uint64_t *d_cost_table_l,
                                    uint64_t *d_cost_table_r,
                                    int img_w, int img_h,
                                    int win_w, int win_h);

__global__ void cu_build_dsi_from_table(uint64_t *d_cost_table_l,
                                        uint64_t *d_cost_table_r,
                                        float *d_cost,
                                        int img_w, int img_h, int scale, int max_disp);

__device__ int cu_hamming_cost(uint64_t ct_l, uint64_t ct_r);

__global__ void cu_cost_horizontal_filter(float *d_cost, int img_w, int img_h, int max_disp, int win_size);

__global__ void cu_cost_vertical_filter(float *d_cost, int img_w, int img_h, int max_disp, int win_size);

__global__ void cu_cost_horizontal_filter_new(float *d_cost, float *d_cost_tmp, int img_w, int img_h, int max_disp, int win_size);

__global__ void cu_cost_vertical_filter_new(float *d_cost, float *d_cost_tmp, int img_w, int img_h, int max_disp, int win_size);

__global__ void cu_cost_filter(float *d_cost, float *d_cost1, float *d_cost2, int img_w, int img_h, int max_disp);
