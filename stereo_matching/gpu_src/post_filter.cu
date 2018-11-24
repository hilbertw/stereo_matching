#include "../gpu_inc/post_filter.cuh"


__global__ void cu_subpixel(float *d_cost, uchar *d_disp, float *d_filtered_disp, int img_w, int img_h, int max_disp, int invalid)
{
	int index = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
	if (index > img_w * img_h - 1)  return;
	int col = index % img_w;
	int row = index / img_w;

	int d = d_disp[index];
	if (d > max_disp - 1)
	{
		d_filtered_disp[index] = invalid;
	}
	else if (!d || d == max_disp - 1)
	{
		d_filtered_disp[index] = d;
	}
	else
	{
		int idx = row * img_w * max_disp + col * max_disp + d;
		float cost_d = d_cost[idx];
		float cost_d_sub = d_cost[idx - 1];
		float cost_d_plus = d_cost[idx + 1];
		d_filtered_disp[index] = d + (cost_d_sub - cost_d_plus) / (2 * (cost_d_sub + cost_d_plus - 2 * cost_d));
		if (d_filtered_disp[index] > max_disp - 1)
		{
			d_filtered_disp[index] = max_disp - 1;
		}
	}
}

__global__ void cu_mean_filter(float *d_filtered_disp, int img_w, int img_h, int max_disp, int win_w, int win_h)
{
	int index = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
	if (index > img_w * img_h - 1)  return;
	int col = index % img_w;
	int row = index / img_w;

	if (row < win_h / 2 || row > img_h - win_h / 2 - 1 || col < win_w / 2 || col > img_w - win_w / 2)  return;
	if (d_filtered_disp[index] <= max_disp - 1)  return;

	float sum = 0;
	int valid_cnt = 0;
	for (int m = row - win_h / 2; m <= row + win_h / 2; m++)
	{
		for (int n = col - win_w / 2; n <= col + win_w / 2; n++)
		{
			int idx = m * img_w + n;
			if (d_filtered_disp[idx] <= max_disp - 1)
			{
				valid_cnt++;
				sum += d_filtered_disp[idx];
			}
		}
	}
	if (valid_cnt > win_w * win_h / 2)
	{
		d_filtered_disp[index] = sum / valid_cnt;
	}
}

__global__ void cu_speckle_filter(float *d_filtered_disp)
{}