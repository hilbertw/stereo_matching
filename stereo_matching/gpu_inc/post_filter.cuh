#include "../cpu_inc/global.h"


__global__ void cu_subpixel(float *d_cost, uchar *d_disp, float *d_filtered_disp, int img_w, int img_h, int max_disp, int invalid);

__global__ void cu_mean_filter(float *d_filtered_disp, int img_w, int img_h, int max_disp, int win_w, int win_h);

__global__ void cu_speckle_filter_init(int *label, int *area, int img_w, int img_h);

__global__ void cu_speckle_filter_union_find(float *d_filtered_disp, int *label, int *area, int img_w, int img_h, int max_dis);

__global__ void cu_speckle_filter_sum_up(int *label, int *area, int img_w, int img_h);

__global__ void cu_speckle_filter_end(float *d_filtered_disp, int *label, int *area, int img_w, int img_h, int value, int max_size);
