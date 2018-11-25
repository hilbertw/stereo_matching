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


__global__ void cu_speckle_filter_init(int *label, int *area, int img_w, int img_h)
{
	int index = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
	if (index > img_w * img_h - 1)  return;

	label[index] = index;
	area[index] = 0;
}


__device__ int Find(int i, int *d_label)
{
	while (i != d_label[i])
	{
		i = d_label[i];
	}
	return i;
}


__device__ void Union(int i, int j, int *d_label)  // join i to j
{
	int label_a = Find(i, d_label);
	int label_b = Find(j, d_label);
	if (label_a != label_b)
	{
		//atomicExch(&d_label[label_a], label_b);
		atomicMin(&d_label[label_a], label_b);
	}
}


__global__ void cu_speckle_filter_union_find(float *d_filtered_disp, int *label, int *area, int img_w, int img_h, int max_dis)
{
	int index = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
	if (index > img_w * img_h - 1)  return;

	int row = index / img_w;
	int col = index % img_w;
	if (row > 0 && fabs(d_filtered_disp[index] - d_filtered_disp[index - img_w]) < max_dis)
	{
		Union(index - img_w, index, label);
	}
	if (row < img_h - 1 && fabs(d_filtered_disp[index] - d_filtered_disp[index + img_w]) < max_dis)
	{
		Union(index + img_w, index, label);
	}
	if (col > 0 && fabs(d_filtered_disp[index] - d_filtered_disp[index - 1]) < max_dis)
	{
		Union(index - 1, index, label);
	}
	if (col < img_w - 1 && fabs(d_filtered_disp[index] - d_filtered_disp[index + 1]) < max_dis)
	{
		Union(index + 1, index, label);
	}
}


__global__ void cu_speckle_filter_sum_up(int *label, int *area, int img_w, int img_h)
{
	int index = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
	if (index > img_w * img_h - 1)  return;

	label[index] = Find(index, label);
	atomicAdd(&area[label[index]], 1);
}


__global__ void cu_speckle_filter_end(float *d_filtered_disp, int *label, int *area, int img_w, int img_h, int value, int max_size)
{
	int index = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
	if (index > img_w * img_h - 1)  return;

	if (area[label[index]] <= max_size)
	{
		d_filtered_disp[index] = value;
	}
}