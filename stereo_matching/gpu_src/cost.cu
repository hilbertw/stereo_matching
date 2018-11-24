#include "../gpu_inc/cost.cuh"


__global__ void cu_Build_cost_table(uchar *d_ll, uchar *d_rr,
																   uint64_t *d_cost_table_l, 
	                                                               uint64_t *d_cost_table_r,
	                                                               int img_w, int img_h,
																   int win_w, int win_h)
{
	int index = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
	if (index > img_w * img_h - 1)  return;
	int col = index % img_w;
	int row = index / img_w;

	uint64_t value_l = 0, value_r = 0;
	uchar ctr_pixel_l = d_ll[index];
	uchar ctr_pixel_r = d_rr[index];

	for (int i = -win_h / 2; i <= win_h / 2; i++)
	{
		int y = MAX(row + i, 0);  // check border
		y = MIN(y, img_h - 1);
		for (int j = -win_w / 2; j <= win_w / 2; j++)
		{
			if (i == 0 && j == 0)
				continue;
			int x = MAX(col + j, 0);
			x = MIN(x, img_w - 1);
			int index_ = y * img_w + x;
			value_l = (value_l | (d_ll[index_] > ctr_pixel_l)) << 1;
			value_r = (value_r | (d_rr[index_] > ctr_pixel_r)) << 1;
		}
	}
	d_cost_table_l[index] = value_l;
	d_cost_table_r[index] = value_r;
}


__global__ void cu_Build_dsi_from_table(uint64_t *d_cost_table_l,
																		   uint64_t *d_cost_table_r,
																		   float *d_cost,
																		   int img_w, int img_h, int max_disp)
{
	int index = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
	if (index > img_w * img_h - 1)  return;
	int col = index % img_w;
	int row = index / img_w;

	for (int i = 0; i < max_disp; i++)
	{
		int dst_index = row * img_w * max_disp + col * max_disp + i;
		uint64_t ct_l = d_cost_table_l[row*img_w + col];
		uint64_t ct_r = d_cost_table_r[row*img_w + MAX(col - i, 0)];
		d_cost[dst_index] = cu_hamming_cost(ct_l, ct_r);
	}
}


__device__ int cu_hamming_cost(uint64_t ct_l, uint64_t ct_r)
{
	uint64_t not_the_same = ct_l ^ ct_r;
	// find the number of '1', log(N)
	int cnt = 0;
	while (not_the_same)
	{
		//std::cout << not_the_same << std::endl;
		cnt += (not_the_same & 1);
		not_the_same >>= 1;
	}
	return cnt;
}


__global__ void cu_cost_horizontal_filter(float *d_cost, int img_w, int img_h, int max_disp, int win_size)
{
	int index = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
	if (index > img_h * max_disp - 1)  return;
	int row = index % img_h;
	int disp = index / img_h;

	float sum = 0;
	int dst_index = row * img_w * max_disp + disp;
	// initialize
	for (int k = 0; k < win_size; k++)
	{
		sum += d_cost[dst_index];
		dst_index += max_disp;
	}
	// box filter
	for (int j = win_size / 2; j < img_w - win_size / 2; j++)
	{
		d_cost[row * img_w * max_disp + j * max_disp + disp] = sum / win_size;
		if (j < img_w - win_size / 2 - 1)
		{
			sum += d_cost[dst_index];
			sum -= d_cost[dst_index - win_size * max_disp];
			dst_index += max_disp;
		}
	}
}


__global__ void cu_cost_vertical_filter(float *d_cost, int img_w, int img_h, int max_disp, int win_size)
{
	int index = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
	if (index > img_w * max_disp - 1)  return;
	int col = index % img_w;
	int disp = index / img_w;

	float sum = 0;
	int dst_index = col * max_disp + disp;
	// initialize
	for (int k = 0; k < win_size; k++)
	{
		sum += d_cost[dst_index];
		dst_index += img_w * max_disp;
	}
	// box filter
	for (int i = win_size / 2; i < img_h - win_size / 2; i++)
	{
		d_cost[i * img_w * max_disp + col * max_disp + disp] = sum / win_size;
		if (i < img_h - win_size / 2 - 1)
		{
			sum += d_cost[dst_index];
			sum -= d_cost[dst_index - win_size * img_w * max_disp];
			dst_index += img_w * max_disp;
		}
	}
}


__global__ void cu_cost_horizontal_filter_new(float *d_cost, float *d_cost_tmp, int img_w, int img_h, int max_disp, int win_size)
{
	int index = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
	if (index > img_h * max_disp - 1)  return;
	int row = index % img_h;
	int disp = index / img_h;

	float sum = 0;
	int dst_index = row * img_w * max_disp + disp;
	// initialize
	for (int k = 0; k < win_size; k++)
	{
		sum += d_cost[dst_index];
		dst_index += max_disp;
	}
	// box filter
	for (int j = win_size / 2; j < img_w - win_size / 2; j++)
	{
		d_cost_tmp[row * img_w * max_disp + j * max_disp + disp] = sum / win_size;
		if (j < img_w - win_size / 2 - 1)
		{
			sum += d_cost[dst_index];
			sum -= d_cost[dst_index - win_size * max_disp];
			dst_index += max_disp;
		}
	}
}


__global__ void cu_cost_vertical_filter_new(float *d_cost, float *d_cost_tmp, int img_w, int img_h, int max_disp, int win_size)
{
	int index = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
	if (index > img_w * max_disp - 1)  return;
	int col = index % img_w;
	int disp = index / img_w;

	float sum = 0;
	int dst_index = col * max_disp + disp;
	// initialize
	for (int k = 0; k < win_size; k++)
	{
		sum += d_cost[dst_index];
		dst_index += img_w * max_disp;
	}
	// box filter
	for (int i = win_size / 2; i < img_h - win_size / 2; i++)
	{
		d_cost_tmp[i * img_w * max_disp + col * max_disp + disp] = sum / win_size;
		if (i < img_h - win_size / 2 - 1)
		{
			sum += d_cost[dst_index];
			sum -= d_cost[dst_index - win_size * img_w * max_disp];
			dst_index += img_w * max_disp;
		}
	}
}


__global__ void cu_cost_filter(float *d_cost, float *d_cost1, float *d_cost2, int img_w, int img_h, int max_disp)
{
	int index = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
	if (index > img_w * max_disp - 1)  return;
	int col = index % img_w;
	int disp = index / img_w;

	for (int i = 0; i < img_h; i++)
	{
		int dst_index = i * img_w * max_disp + col * max_disp + disp;
		d_cost[dst_index] = d_cost1[dst_index] + d_cost2[dst_index];
	}
}
