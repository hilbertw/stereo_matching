#include "../gpu_inc/SGM.cuh"


GPU_SGM::GPU_SGM()
{
	cudaSetDevice(0);
	checkCudaErrors(cudaStreamCreate(&stream1));
	checkCudaErrors(cudaStreamCreate(&stream2));
	checkCudaErrors(cudaStreamCreate(&stream3));
	checkCudaErrors(cudaStreamCreate(&stream4));
	checkCudaErrors(cudaStreamCreate(&stream5));
	checkCudaErrors(cudaStreamCreate(&stream6));
	checkCudaErrors(cudaStreamCreate(&stream7));
	checkCudaErrors(cudaStreamCreate(&stream8));

	checkCudaErrors(cudaMalloc((void**)&d_ll, IMG_H* IMG_W * sizeof(uchar)));
	checkCudaErrors(cudaMalloc((void**)&d_rr, IMG_H * IMG_W * sizeof(uchar)));
	checkCudaErrors(cudaMalloc((void**)&d_disp, IMG_H * IMG_W * sizeof(uchar)));
	checkCudaErrors(cudaMalloc((void**)&d_cost_table_l, IMG_H * IMG_W * sizeof(uint64_t)));
	checkCudaErrors(cudaMalloc((void**)&d_cost_table_r, IMG_H * IMG_W * sizeof(uint64_t)));
	checkCudaErrors(cudaMalloc((void**)&d_cost, IMG_H * IMG_W * CU_MAX_DISP * sizeof(float)));

	checkCudaErrors(cudaMalloc((void**)&d_L1, IMG_H * IMG_W * CU_MAX_DISP * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&d_L2, IMG_H * IMG_W * CU_MAX_DISP * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&d_min_L1, IMG_H * IMG_W * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&d_min_L2, IMG_H * IMG_W * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&d_L3, IMG_H * IMG_W * CU_MAX_DISP * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&d_L4, IMG_H * IMG_W * CU_MAX_DISP * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&d_min_L3, IMG_H * IMG_W * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&d_min_L4, IMG_H * IMG_W * sizeof(float)));

	checkCudaErrors(cudaMalloc((void**)&d_L5, IMG_H * IMG_W * CU_MAX_DISP * sizeof(short)));
	checkCudaErrors(cudaMalloc((void**)&d_L6, IMG_H * IMG_W * CU_MAX_DISP * sizeof(short)));
	checkCudaErrors(cudaMalloc((void**)&d_min_L5, IMG_H * IMG_W * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&d_min_L6, IMG_H * IMG_W * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&d_L7, IMG_H * IMG_W * CU_MAX_DISP * sizeof(short)));
	checkCudaErrors(cudaMalloc((void**)&d_L8, IMG_H * IMG_W * CU_MAX_DISP * sizeof(short)));
	checkCudaErrors(cudaMalloc((void**)&d_min_L7, IMG_H * IMG_W * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&d_min_L8, IMG_H * IMG_W * sizeof(float)));

	P1 = 10;
	P2 = 100;
}



GPU_SGM::~GPU_SGM()
{
	checkCudaErrors(cudaFree(d_ll));
	checkCudaErrors(cudaFree(d_rr));
	checkCudaErrors(cudaFree(d_disp));
	checkCudaErrors(cudaFree(d_cost_table_l));
	checkCudaErrors(cudaFree(d_cost_table_r));
	checkCudaErrors(cudaFree(d_cost));

	checkCudaErrors(cudaFree(d_L1));
	checkCudaErrors(cudaFree(d_L2));
	checkCudaErrors(cudaFree(d_min_L1));
	checkCudaErrors(cudaFree(d_min_L2));
	checkCudaErrors(cudaFree(d_L3));
	checkCudaErrors(cudaFree(d_L4));
	checkCudaErrors(cudaFree(d_min_L3));
	checkCudaErrors(cudaFree(d_min_L4));

	checkCudaErrors(cudaFree(d_L5));
	checkCudaErrors(cudaFree(d_L6));
	checkCudaErrors(cudaFree(d_min_L5));
	checkCudaErrors(cudaFree(d_min_L6));
	checkCudaErrors(cudaFree(d_L7));
	checkCudaErrors(cudaFree(d_L8));
	checkCudaErrors(cudaFree(d_min_L7));
	checkCudaErrors(cudaFree(d_min_L8));

	checkCudaErrors(cudaStreamDestroy(stream1));
	checkCudaErrors(cudaStreamDestroy(stream2));
	checkCudaErrors(cudaStreamDestroy(stream3));
	checkCudaErrors(cudaStreamDestroy(stream4));
	checkCudaErrors(cudaStreamDestroy(stream5));
	checkCudaErrors(cudaStreamDestroy(stream6));
	checkCudaErrors(cudaStreamDestroy(stream7));
	checkCudaErrors(cudaStreamDestroy(stream8));
}


static void ccl_dfs(int row, int col, Mat &m, bool *visited, int *label, int *area, int label_cnt, int max_dis)
{
	visited[row * m.cols + col] = 1;

	if (row > 0 && fabs(m.at<float>(row, col) - m.at<float>(row - 1, col)) < max_dis)
	{
		if (!visited[(row - 1) * m.cols + col])
		{
			label[(row - 1) * m.cols + col] = label_cnt;
			++area[label_cnt];
			//printf("1: %d, %d; ", row-1, col);
			ccl_dfs(row - 1, col, m, visited, label, area, label_cnt, max_dis);
		}
	}
	if (row < m.rows - 1 && fabs(m.at<float>(row, col) - m.at<float>(row + 1, col)) < max_dis)
	{
		if (!visited[(row + 1) * m.cols + col])
		{
			label[(row + 1) * m.cols + col] = label_cnt;
			++area[label_cnt];
			//printf("2: %d, %d; ", row+1, col);
			ccl_dfs(row + 1, col, m, visited, label, area, label_cnt, max_dis);
		}
	}
	if (col > 0 && fabs(m.at<float>(row, col) - m.at<float>(row, col - 1)) < max_dis)
	{
		if (!visited[row * m.cols + col - 1])
		{
			label[row * m.cols + col - 1] = label_cnt;
			++area[label_cnt];
			//printf("3: %d, %d; ", row, col-1);
			ccl_dfs(row, col - 1, m, visited, label, area, label_cnt, max_dis);
		}
	}
	if (col < m.cols - 1 && fabs(m.at<float>(row, col) - m.at<float>(row, col + 1)) < max_dis)
	{
		if (!visited[row * m.cols + col + 1])
		{
			label[row * m.cols + col + 1] = label_cnt;
			++area[label_cnt];
			//printf("4: %d, %d; ", row, col+1);
			ccl_dfs(row, col + 1, m, visited, label, area, label_cnt, max_dis);
		}
	}
	return;
}


static void speckle_filter(Mat &m, int value, int max_size, int max_dis)
{
	bool *visited = new bool[m.rows * m.cols];
	int *label = new int[m.rows * m.cols];
	int *area = new int[m.rows * m.cols];
	for (int i = 0; i < m.rows * m.cols; i++)
	{
		visited[i] = 0;
		label[i] = 0;
		area[i] = 0;
	}

	int label_cnt = 0;
	for (int i = 0; i < m.rows; i++)
	{
		for (int j = 0; j < m.cols; j++)
		{
			if (visited[i * m.cols + j])  continue;

			label[i*m.cols + j] = ++label_cnt;
			++area[label_cnt];
			ccl_dfs(i, j, m, visited, label, area, label_cnt, max_dis);
		}
	}

	for (int i = 0; i < m.rows; i++)
	{
		for (int j = 0; j < m.cols; j++)
		{
			if (area[label[i*m.cols + j]] <= max_size)
			{
				m.at<float>(i, j) = value;
			}
		}
	}

	delete[] visited;
	delete[] label;
	delete[] area;
}

/*
void GPU_SGM::post_filter()
{
	// sub-pixel
#pragma omp parallel for
	for (int i = 0; i < IMG_H; i++)
	{
		for (int j = 0; j < IMG_W; j++)
		{
			int d = disp.at<uchar>(i, j);
			if (d > CU_MAX_DISP - 1)
			{
				filtered_disp.at<float>(i, j) = CU_INVALID_DISP;
			}
			else if (!d || d == CU_MAX_DISP - 1)
			{
				filtered_disp.at<float>(i, j) = d;
			}
			else
			{
				int index = i * IMG_W * CU_MAX_DISP + j * CU_MAX_DISP + d;
				float cost_d = cost[index];
				float cost_d_sub = cost[index - 1];
				float cost_d_plus = cost[index + 1];
				filtered_disp.at<float>(i, j) = MIN(d + (cost_d_sub - cost_d_plus) / (2 * (cost_d_sub + cost_d_plus - 2 * cost_d)), CU_MAX_DISP - 1);
			}
		}
	}

	// median filter
	vector<int> v;
	for (int i = CU_MEDIAN_FILTER_H / 2; i < filtered_disp.rows - CU_MEDIAN_FILTER_H / 2; i++)
	{
		for (int j = CU_MEDIAN_FILTER_W / 2; j < filtered_disp.cols - CU_MEDIAN_FILTER_W / 2; j++)
		{
			if (filtered_disp.at<float>(i, j) <= CU_MAX_DISP - 1)  continue;
			int valid_cnt = 0;
			v.clear();
			for (int m = i - CU_MEDIAN_FILTER_H / 2; m <= i + CU_MEDIAN_FILTER_H / 2; m++)
			{
				for (int n = j - CU_MEDIAN_FILTER_W / 2; n <= j + CU_MEDIAN_FILTER_W / 2; n++)
				{
					if (filtered_disp.at<float>(m, n) <= CU_MAX_DISP - 1)
					{
						v.push_back(filtered_disp.at<float>(m, n));
						valid_cnt++;
					}
				}
			}
			if (valid_cnt > CU_MEDIAN_FILTER_W * CU_MEDIAN_FILTER_H / 2)
			{
				sort(v.begin(), v.end());
				filtered_disp.at<float>(i, j) = v[valid_cnt / 2];
			}
		}
	}

	// speckle_filter
	speckle_filter(filtered_disp, CU_INVALID_DISP, CU_SPECKLE_SIZE, CU_SPECKLE_DIS);
}
*/


__global__ void warmup()
{}


void GPU_SGM::Process(Mat &ll, Mat &rr, uchar *disp, float *cost)
{
	cudaSetDevice(0);
	warmup << <512, 512 >> >();  // warm up 

	cudaMemcpyAsync(d_ll, ll.data, IMG_H* IMG_W * sizeof(uchar), cudaMemcpyHostToDevice, stream1);
	cudaMemcpyAsync(d_rr, rr.data, IMG_H* IMG_W * sizeof(uchar), cudaMemcpyHostToDevice, stream2);
	cudaStreamSynchronize(stream1);
	cudaStreamSynchronize(stream2);

	double be = get_cur_ms();

	dim3 grid, block;
	grid.x = (IMG_W - 1) / 32 + 1;
	grid.y = (IMG_H - 1) / 32 + 1;
	block.x = 32;
	block.y = 32;
	cu_Build_cost_table << <grid, block, 0, stream1 >> > (d_ll, d_rr, d_cost_table_l, d_cost_table_r, IMG_W, IMG_H, CU_WIN_W, CU_WIN_H);
	cu_Build_dsi_from_table << <grid, block, 0, stream1 >> > (d_cost_table_l, d_cost_table_r, d_cost, IMG_W, IMG_H, CU_MAX_DISP);

	cudaDeviceSynchronize();
	printf("build cost takes %lf ms\n", get_cur_ms() - be);

	be = get_cur_ms();

	//cu_cost_filter_new << <grid, block, 0, stream1 >> > (d_cost, IMG_W, IMG_H, CU_MAX_DISP, CU_COST_WIN_W, CU_COST_WIN_H);

	grid.x = (IMG_W - 1) / 32 + 1;
	grid.y = (CU_MAX_DISP - 1) / 32 + 1;
	cu_cost_horizontal_filter << <grid, block, 0, stream1 >> > (d_cost, IMG_W, IMG_H, CU_MAX_DISP, CU_COST_WIN_W);
	cu_cost_vertical_filter << <grid, block, 0, stream1 >> > (d_cost, IMG_W, IMG_H, CU_MAX_DISP, CU_COST_WIN_H);

	//cu_cost_horizontal_filter_new << <grid, block, 0, stream1 >> > (d_cost, d_L1, IMG_W, IMG_H, CU_MAX_DISP, CU_COST_WIN_W);
	//cu_cost_vertical_filter_new << <grid, block, 0, stream2 >> > (d_cost, d_L2, IMG_W, IMG_H, CU_MAX_DISP, CU_COST_WIN_H);
	//cudaStreamSynchronize(stream1);
	//cudaStreamSynchronize(stream2);
	//cu_cost_filter << <grid, block, 0, stream1 >> > (d_cost, d_L1, d_L2, IMG_W, IMG_H, CU_MAX_DISP);

	cudaDeviceSynchronize();
	printf("cost filter takes %lf ms\n", get_cur_ms() - be);

	be = get_cur_ms();

	grid.x = 512;
	grid.y = 512;
	dim3 dp_grid, dp_block;
	dp_grid.x = IMG_W;
	dp_grid.y = 1;
	dp_block.x = CU_MAX_DISP;  // for dp syncronize
	dp_block.y = 1;

	cu_dp_L1 << <dp_grid, dp_block, 0, stream1 >> > (d_cost, d_L1, d_min_L1, IMG_W, IMG_H, CU_MAX_DISP, P1, P2);
	cu_dp_L2 << <dp_grid, dp_block, 0, stream2 >> > (d_cost, d_L2, d_min_L2, IMG_W, IMG_H, CU_MAX_DISP, P1, P2);
	cu_dp_L3 << <dp_grid, dp_block, 0, stream3 >> > (d_cost, d_L3, d_min_L3, IMG_W, IMG_H, CU_MAX_DISP, P1, P2);
	cu_dp_L4 << <dp_grid, dp_block, 0, stream4 >> > (d_cost, d_L4, d_min_L4, IMG_W, IMG_H, CU_MAX_DISP, P1, P2);
	if (CU_USE_8_PATH)
	{
		for (int i = 0; i < IMG_H; i++)
		{
			cu_dp_L5 << <dp_grid, dp_block, 0, stream5 >> > (d_cost, d_L5, d_min_L5, i, IMG_W, IMG_H, CU_MAX_DISP, P1, P2);
			cu_dp_L6 << <dp_grid, dp_block, 0, stream6 >> > (d_cost, d_L6, d_min_L6, i, IMG_W, IMG_H, CU_MAX_DISP, P1, P2);
			cu_dp_L7 << <dp_grid, dp_block, 0, stream7 >> > (d_cost, d_L7, d_min_L7, IMG_H - 1 - i, IMG_W, IMG_H, CU_MAX_DISP, P1, P2);
			cu_dp_L8 << <dp_grid, dp_block, 0, stream8 >> > (d_cost, d_L8, d_min_L8, IMG_H - 1 - i, IMG_W, IMG_H, CU_MAX_DISP, P1, P2);
		}
	}
	cudaStreamSynchronize(stream1);
	cudaStreamSynchronize(stream2);
	cudaStreamSynchronize(stream3);
	cudaStreamSynchronize(stream4);
	cudaStreamSynchronize(stream5);
	cudaStreamSynchronize(stream6);
	cudaStreamSynchronize(stream7);
	cudaStreamSynchronize(stream8);
	aggregation << <grid, block, 0, stream1 >> > (d_cost, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6, d_L7, d_L8, IMG_W, IMG_H, CU_MAX_DISP);
	cudaStreamSynchronize(stream1);

	grid.x = (IMG_W - 1) / 32 + 1;
	grid.y = (IMG_H - 1) / 32 + 1;
	wta<<<grid, block, 0, stream1>>>(d_cost, d_disp, IMG_W, IMG_H, CU_MAX_DISP, CU_UNIQUE_RATIO, CU_INVALID_DISP);
	cudaMemcpyAsync(cost, d_cost, IMG_H * IMG_W * CU_MAX_DISP * sizeof(float), cudaMemcpyDeviceToHost, stream2);
	cudaMemcpyAsync(disp, d_disp, IMG_H * IMG_W * sizeof(uchar), cudaMemcpyDeviceToHost, stream1);

	cudaDeviceSynchronize();
	printf("dp takes %lf ms\n", get_cur_ms() - be);
}