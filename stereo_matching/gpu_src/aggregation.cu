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


__global__ void cu_dp_L1(float *d_cost, float *d_L1, float *d_min_L1, int img_w, int img_h, int max_disp, int P1, int P2)
{
	int row = blockIdx.x;
	int disp = threadIdx.x;
	if (row > img_h - 1)  return;

	for (int j = 0; j < img_w; j++)
	{
		d_min_L1[row * img_w + j] = FLT_MAX;
		__syncthreads();

		int index = row * img_w * max_disp + j * max_disp + disp;
		if (j == 0)		//init
		{
			d_L1[index] = d_cost[index];
		}
		else
		{
			int index_L1_prev = row * img_w * max_disp + (j - 1) * max_disp;
			uchar d_sub_1 = MAX(disp - 1, 0);
			uchar d_plus_1 = MIN(disp + 1, max_disp - 1);
			d_L1[index] = MIN(d_L1[index_L1_prev + disp], d_L1[index_L1_prev + d_sub_1] + P1);
			d_L1[index] = MIN(d_L1[index], d_L1[index_L1_prev + d_plus_1] + P1);
			d_L1[index] = MIN(d_L1[index], d_min_L1[row * img_w + j - 1] + P2);
			d_L1[index] += (d_cost[index] - d_min_L1[row * img_w + j - 1]);
		}
		atomicMin(&d_min_L1[row * img_w + j], d_L1[index]);
	}
}

__global__ void cu_dp_L2(float *d_cost, float *d_L1, float *d_min_L1, int img_w, int img_h, int max_disp, int P1, int P2)
{}


__global__ void cu_dp_L3(float *d_cost, float *d_L1, float *d_min_L1, int img_w, int img_h, int max_disp, int P1, int P2)
{}


__global__ void cu_dp_L4(float *d_cost, float *d_L1, float *d_min_L1, int img_w, int img_h, int max_disp, int P1, int P2)
{}

__global__ void aggregation(float *d_cost, float *d_L1, int img_w, int img_h, int max_disp)
{
	int index = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
	if (index > img_w * img_h *  max_disp - 1)  return;

	d_cost[index] = d_L1[index];
}


__global__ void wta(float *d_cost, uchar *disparity, int img_w, int img_h, int max_disp, float ratio, uchar invalid)
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
		if (d_cost[idx] < min_cost)
		{
			min_cost = d_cost[idx];
			min_d = d;
		}
	}
	// unique check
	float sec_min_cost = FLT_MAX;
	uchar sec_min_d = invalid;
	for (int d = 0; d < max_disp; d++)
	{
		int idx = row * img_w * max_disp + col * max_disp + d;
		if (d_cost[idx] < sec_min_cost && d_cost[idx] != min_cost)
		{
			sec_min_cost = d_cost[idx];
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