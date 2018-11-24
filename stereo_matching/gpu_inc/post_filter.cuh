#include "../global.h"


__global__ void cu_subpixel(float *d_cost, uchar *d_disp, float *d_filtered_disp, int img_w, int img_h, int max_disp, int invalid);

__global__ void cu_mean_filter(float *d_filtered_disp, int img_w, int img_h, int max_disp, int win_w, int win_h);

__global__ void cu_speckle_filter(float *d_filtered_disp);