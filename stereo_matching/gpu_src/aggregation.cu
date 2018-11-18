#include "../gpu_inc/aggregation.cuh"


__device__ static float atomicMin(float* address, float val)
{
	int* address_as_i = (int*)address;
	int old = *address_as_i, assumed;
	do {
		assumed = old;
		old = ::atomicCAS(address_as_i, assumed,
			__float_as_int(::fminf(val, __int_as_float(assumed))));
	} while (assumed != old);
	return __int_as_float(old);
}


__global__ void cu_dp_L1(float *d_cost, float *d_dp, float *d_dp_min, int img_w, int img_h, int max_disp, int P1, int P2)
{
	int row = blockIdx.x;
	int disp = threadIdx.x;
	if (row > img_h - 1)  return;

	for (int j = 0; j < img_w; j++)
	{
		d_dp_min[row * img_w + j] = FLT_MAX;
		__syncthreads();

		int index = row * img_w * max_disp + j * max_disp + disp;
		if (j == 0)		//init
		{
			d_dp[index] = d_cost[index];
		}
		else
		{
			int index_L1_prev = row * img_w * max_disp + (j - 1) * max_disp;
			uchar d_sub_1 = MAX(disp - 1, 0);
			uchar d_plus_1 = MIN(disp + 1, max_disp - 1);
			d_dp[index] = MIN(d_dp[index_L1_prev + disp], d_dp[index_L1_prev + d_sub_1] + P1);
			d_dp[index] = MIN(d_dp[index], d_dp[index_L1_prev + d_plus_1] + P1);
			d_dp[index] = MIN(d_dp[index], d_dp_min[row * img_w + j - 1] + P2);
			d_dp[index] += (d_cost[index] - d_dp_min[row * img_w + j - 1]);
		}
		atomicMin(&d_dp_min[row * img_w + j], d_dp[index]);
	}
}


__global__ void cu_dp_L2(float *d_cost, float *d_dp, float *d_dp_min, int img_w, int img_h, int max_disp, int P1, int P2)
{
	int row = blockIdx.x;
	int disp = threadIdx.x;
	if (row > img_h - 1)  return;

	for (int j = img_w - 1; j >= 0; j--)
	{
		d_dp_min[row * img_w + j] = FLT_MAX;
		__syncthreads();

		int index = row * img_w * max_disp + j * max_disp + disp;
		if (j == img_w - 1)		//init
		{
			d_dp[index] = d_cost[index];
		}
		else
		{
			int index_L2_prev = row * img_w * max_disp + (j + 1) * max_disp;
			uchar d_sub_1 = MAX(disp - 1, 0);
			uchar d_plus_1 = MIN(disp + 1, max_disp - 1);
			d_dp[index] = MIN(d_dp[index_L2_prev + disp], d_dp[index_L2_prev + d_sub_1] + P1);
			d_dp[index] = MIN(d_dp[index], d_dp[index_L2_prev + d_plus_1] + P1);
			d_dp[index] = MIN(d_dp[index], d_dp_min[row * img_w + j + 1] + P2);
			d_dp[index] += (d_cost[index] - d_dp_min[row * img_w + j + 1]);
		}
		atomicMin(&d_dp_min[row * img_w + j], d_dp[index]);
	}
}


__global__ void cu_dp_L3(float *d_cost, float *d_dp, float *d_dp_min, int img_w, int img_h, int max_disp, int P1, int P2)
{
	int col = blockIdx.x;
	int disp = threadIdx.x;
	if (col > img_w - 1)  return;

	for (int i = 0; i < img_h; i++)
	{
		d_dp_min[i * img_w + col] = FLT_MAX;
		__syncthreads();

		int index = i * img_w * max_disp + col * max_disp + disp;
		if (i == 0)		//init
		{
			d_dp[index] = d_cost[index];
		}
		else
		{
			int index_L3_prev = (i - 1) * img_w * max_disp + col * max_disp;
			uchar d_sub_1 = MAX(disp - 1, 0);
			uchar d_plus_1 = MIN(disp + 1, max_disp - 1);
			d_dp[index] = MIN(d_dp[index_L3_prev + disp], d_dp[index_L3_prev + d_sub_1] + P1);
			d_dp[index] = MIN(d_dp[index], d_dp[index_L3_prev + d_plus_1] + P1);
			d_dp[index] = MIN(d_dp[index], d_dp_min[(i - 1) * img_w + col] + P2);
			d_dp[index] += (d_cost[index] - d_dp_min[(i - 1) * img_w + col]);
		}
		atomicMin(&d_dp_min[i * img_w + col], d_dp[index]);
	}
}


__global__ void cu_dp_L4(float *d_cost, float *d_dp, float *d_dp_min, int img_w, int img_h, int max_disp, int P1, int P2)
{
	int col = blockIdx.x;
	int disp = threadIdx.x;
	if (col > img_w - 1)  return;

	for (int i = img_h - 1; i >= 0; i--)
	{
		d_dp_min[i * img_w + col] = FLT_MAX;
		__syncthreads();

		int index = i * img_w * max_disp + col * max_disp + disp;
		if (i == img_h - 1)		//init
		{
			d_dp[index] = d_cost[index];
		}
		else
		{
			int index_L4_prev = (i + 1) * img_w * max_disp + col * max_disp;
			uchar d_sub_1 = MAX(disp - 1, 0);
			uchar d_plus_1 = MIN(disp + 1, max_disp - 1);
			d_dp[index] = MIN(d_dp[index_L4_prev + disp], d_dp[index_L4_prev + d_sub_1] + P1);
			d_dp[index] = MIN(d_dp[index], d_dp[index_L4_prev + d_plus_1] + P1);
			d_dp[index] = MIN(d_dp[index], d_dp_min[(i + 1) * img_w + col] + P2);
			d_dp[index] += (d_cost[index] - d_dp_min[(i + 1) * img_w + col]);
		}
		atomicMin(&d_dp_min[i * img_w + col], d_dp[index]);
	}
}


__global__ void cu_dp_L5(float *d_cost, short *d_dp, float *d_dp_min, int idx, int img_w, int img_h, int max_disp, int P1, int P2)
{
	int col = blockIdx.x;
	int disp = threadIdx.x;
	int row = idx;

	d_dp_min[row * img_w + col] = FLT_MAX;
	__syncthreads();

	int index = row * img_w * max_disp + col * max_disp + disp;
	if (row == 0 || col == 0)		//init
	{
		d_dp[index] = d_cost[index];
	}
	else
	{
		int index_L5_prev = (row - 1) * img_w * max_disp + (col - 1) * max_disp;
		uchar d_sub_1 = MAX(disp - 1, 0);
		uchar d_plus_1 = MIN(disp + 1, max_disp - 1);
		d_dp[index] = MIN(d_dp[index_L5_prev + disp], d_dp[index_L5_prev + d_sub_1] + P1);
		d_dp[index] = MIN(d_dp[index], d_dp[index_L5_prev + d_plus_1] + P1);
		d_dp[index] = MIN(d_dp[index], d_dp_min[(row - 1) * img_w + col - 1] + P2);
		d_dp[index] += (d_cost[index] - d_dp_min[(row - 1) * img_w + col - 1]);
	}
	atomicMin(&d_dp_min[row * img_w + col], d_dp[index]);
}


__global__ void cu_dp_L6(float *d_cost, short *d_dp, float *d_dp_min, int idx, int img_w, int img_h, int max_disp, int P1, int P2)
{
	int col = blockIdx.x;
	int disp = threadIdx.x;
	int row = idx;

	d_dp_min[row * img_w + col] = FLT_MAX;
	__syncthreads();

	int index = row * img_w * max_disp + col * max_disp + disp;
	if (row == 0 || col == img_w - 1)		//init
	{
		d_dp[index] = d_cost[index];
	}
	else
	{
		int index_L6_prev = (row - 1) * img_w * max_disp + (col + 1) * max_disp;
		uchar d_sub_1 = MAX(disp - 1, 0);
		uchar d_plus_1 = MIN(disp + 1, max_disp - 1);
		d_dp[index] = MIN(d_dp[index_L6_prev + disp], d_dp[index_L6_prev + d_sub_1] + P1);
		d_dp[index] = MIN(d_dp[index], d_dp[index_L6_prev + d_plus_1] + P1);
		d_dp[index] = MIN(d_dp[index], d_dp_min[(row - 1) * img_w + col + 1] + P2);
		d_dp[index] += (d_cost[index] - d_dp_min[(row - 1) * img_w + col + 1]);
	}
	atomicMin(&d_dp_min[row * img_w + col], d_dp[index]);
}


__global__ void cu_dp_L7(float *d_cost, short *d_dp, float *d_dp_min, int idx, int img_w, int img_h, int max_disp, int P1, int P2)
{
	int col = blockIdx.x;
	int disp = threadIdx.x;
	int row = idx;

	d_dp_min[row * img_w + col] = FLT_MAX;
	__syncthreads();

	int index = row * img_w * max_disp + col * max_disp + disp;
	if (row == img_h - 1 || col == 0)		//init
	{
		d_dp[index] = d_cost[index];
	}
	else
	{
		int index_L7_prev = (row + 1) * img_w * max_disp + (col - 1) * max_disp;
		uchar d_sub_1 = MAX(disp - 1, 0);
		uchar d_plus_1 = MIN(disp + 1, max_disp - 1);
		d_dp[index] = MIN(d_dp[index_L7_prev + disp], d_dp[index_L7_prev + d_sub_1] + P1);
		d_dp[index] = MIN(d_dp[index], d_dp[index_L7_prev + d_plus_1] + P1);
		d_dp[index] = MIN(d_dp[index], d_dp_min[(row + 1) * img_w + col - 1] + P2);
		d_dp[index] += (d_cost[index] - d_dp_min[(row + 1) * img_w + col - 1]);
	}
	atomicMin(&d_dp_min[row * img_w + col], d_dp[index]);
}


__global__ void cu_dp_L8(float *d_cost, short *d_dp, float *d_dp_min, int idx, int img_w, int img_h, int max_disp, int P1, int P2)
{
	int col = blockIdx.x;
	int disp = threadIdx.x;
	int row = idx;

	d_dp_min[row * img_w + col] = FLT_MAX;
	__syncthreads();

	int index = row * img_w * max_disp + col * max_disp + disp;
	if (row == img_h - 1 || col == img_w - 1)		//init
	{
		d_dp[index] = d_cost[index];
	}
	else
	{
		int index_L8_prev = (row + 1) * img_w * max_disp + (col + 1) * max_disp;
		uchar d_sub_1 = MAX(disp - 1, 0);
		uchar d_plus_1 = MIN(disp + 1, max_disp - 1);
		d_dp[index] = MIN(d_dp[index_L8_prev + disp], d_dp[index_L8_prev + d_sub_1] + P1);
		d_dp[index] = MIN(d_dp[index], d_dp[index_L8_prev + d_plus_1] + P1);
		d_dp[index] = MIN(d_dp[index], d_dp_min[(row + 1) * img_w + col + 1] + P2);
		d_dp[index] += (d_cost[index] - d_dp_min[(row + 1) * img_w + col + 1]);
	}
	atomicMin(&d_dp_min[row * img_w + col], d_dp[index]);
}


__global__ void aggregation(float *d_cost_sum, float *d_L1, float *d_L2, float *d_L3, float *d_L4,
												    short *d_L5, short *d_L6, short *d_L7, short *d_L8,
	                                                int img_w, int img_h, int max_disp)
{
	int index = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
	if (index > img_w * img_h *  max_disp - 1)  return;

	d_cost_sum[index] = d_L1[index] + d_L2[index] + d_L3[index] + d_L4[index]
										  + d_L5[index] + d_L6[index] + d_L7[index] + d_L8[index];
}


__global__ void wta(float *d_cost_sum, uchar *disparity, int img_w, int img_h, int max_disp, float ratio, uchar invalid)
{
	int index = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
	if (index > img_w * img_h - 1)  return;
	int col = index % img_w;
	int row = index / img_w;

	float min_cost = FLT_MAX;
	uchar min_d = invalid;
	for (int d = 0; d < max_disp; d++)
	{
		int idx = row * img_w * max_disp + col * max_disp + d;
		if (d_cost_sum[idx] < min_cost)
		{
			min_cost = d_cost_sum[idx];
			min_d = d;
		}
	}
	// unique check
	float sec_min_cost = FLT_MAX;
	uchar sec_min_d = invalid;
	for (int d = 0; d < max_disp; d++)
	{
		int idx = row * img_w * max_disp + col * max_disp + d;
		if (d_cost_sum[idx] < sec_min_cost && d_cost_sum[idx] != min_cost)
		{
			sec_min_cost = d_cost_sum[idx];
			sec_min_d = d;
		}
	}
	if (min_cost / sec_min_cost > ratio && abs(min_d - sec_min_d) > 1)
	{
		disparity[index] = invalid;
	}
	else
	{
		disparity[index] = min_d;
	}
}