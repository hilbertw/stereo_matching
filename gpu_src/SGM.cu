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

	checkCudaErrors(cudaMalloc((void**)&d_img_l, IMG_H* IMG_W * sizeof(uchar)));
	checkCudaErrors(cudaMalloc((void**)&d_img_r, IMG_H * IMG_W * sizeof(uchar)));
	checkCudaErrors(cudaMalloc((void**)&d_disp, IMG_H * IMG_W * sizeof(uchar)));
	checkCudaErrors(cudaMalloc((void**)&d_filtered_disp, IMG_H * IMG_W * sizeof(float)));

	checkCudaErrors(cudaMalloc((void**)&d_cost_table_l, IMG_H * IMG_W * sizeof(uint64_t)));
	checkCudaErrors(cudaMalloc((void**)&d_cost_table_r, IMG_H * IMG_W * sizeof(uint64_t)));
	checkCudaErrors(cudaMalloc((void**)&d_cost, IMG_H * IMG_W * MAX_DISP * sizeof(float)));

	checkCudaErrors(cudaMalloc((void**)&d_L1, IMG_H * IMG_W * MAX_DISP * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&d_L2, IMG_H * IMG_W * MAX_DISP * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&d_min_L1, IMG_H * IMG_W * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&d_min_L2, IMG_H * IMG_W * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&d_L3, IMG_H * IMG_W * MAX_DISP * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&d_L4, IMG_H * IMG_W * MAX_DISP * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&d_min_L3, IMG_H * IMG_W * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&d_min_L4, IMG_H * IMG_W * sizeof(float)));

	checkCudaErrors(cudaMalloc((void**)&d_L5, IMG_H * IMG_W * MAX_DISP * sizeof(short)));
	checkCudaErrors(cudaMalloc((void**)&d_L6, IMG_H * IMG_W * MAX_DISP * sizeof(short)));
	checkCudaErrors(cudaMalloc((void**)&d_min_L5, IMG_H * IMG_W * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&d_min_L6, IMG_H * IMG_W * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&d_L7, IMG_H * IMG_W * MAX_DISP * sizeof(short)));
	checkCudaErrors(cudaMalloc((void**)&d_L8, IMG_H * IMG_W * MAX_DISP * sizeof(short)));
	checkCudaErrors(cudaMalloc((void**)&d_min_L7, IMG_H * IMG_W * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&d_min_L8, IMG_H * IMG_W * sizeof(float)));

	P1 = 10;
	P2 = 100;

	checkCudaErrors(cudaMalloc((void**)&d_label, IMG_H * IMG_W * sizeof(int)));
	checkCudaErrors(cudaMalloc((void**)&d_area, IMG_H * IMG_W * sizeof(int)));

	disp.create(IMG_H, IMG_W, CV_8UC1);
	filtered_disp.create(IMG_H, IMG_W, CV_32FC1);
	colored_disp.create(IMG_H, IMG_W, CV_8UC3);

	disp_cnt = 0;
}



GPU_SGM::~GPU_SGM()
{
	checkCudaErrors(cudaFree(d_img_l));
	checkCudaErrors(cudaFree(d_img_r));
	checkCudaErrors(cudaFree(d_disp));
	checkCudaErrors(cudaFree(d_filtered_disp));

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

	checkCudaErrors(cudaFree(d_label));
	checkCudaErrors(cudaFree(d_area));

	checkCudaErrors(cudaStreamDestroy(stream1));
	checkCudaErrors(cudaStreamDestroy(stream2));
	checkCudaErrors(cudaStreamDestroy(stream3));
	checkCudaErrors(cudaStreamDestroy(stream4));
	checkCudaErrors(cudaStreamDestroy(stream5));
	checkCudaErrors(cudaStreamDestroy(stream6));
	checkCudaErrors(cudaStreamDestroy(stream7));
	checkCudaErrors(cudaStreamDestroy(stream8));
}


void GPU_SGM::process(Mat &img_l, Mat &img_r)
{
	this->img_l = img_l;
	this->img_r = img_r;
	cudaSetDevice(0);

	cudaMemcpyAsync(d_img_l, img_l.data, IMG_H* IMG_W * sizeof(uchar), cudaMemcpyHostToDevice, stream1);
	cudaMemcpyAsync(d_img_r, img_r.data, IMG_H* IMG_W * sizeof(uchar), cudaMemcpyHostToDevice, stream2);
	cudaStreamSynchronize(stream1);
	cudaStreamSynchronize(stream2);

	double be = get_cur_ms();

	dim3 grid, block;
	grid.x = (IMG_W - 1) / 32 + 1;
	grid.y = (IMG_H - 1) / 32 + 1;
	block.x = 32;
	block.y = 32;
	cu_build_cost_table << <grid, block, 0, stream1 >> > (d_img_l, d_img_r, d_cost_table_l, d_cost_table_r, IMG_W, IMG_H, CU_WIN_W, CU_WIN_H);
	cu_build_dsi_from_table << <grid, block, 0, stream1 >> > (d_cost_table_l, d_cost_table_r, d_cost, IMG_W, IMG_H, MAX_DISP);

	cudaDeviceSynchronize();
	printf("build cost takes %lf ms\n", get_cur_ms() - be);

	be = get_cur_ms();

	grid.x = (IMG_W - 1) / 32 + 1;
	grid.y = (MAX_DISP - 1) / 32 + 1;
	cu_cost_horizontal_filter << <grid, block, 0, stream1 >> > (d_cost, IMG_W, IMG_H, MAX_DISP, CU_COST_WIN_W);
	cu_cost_vertical_filter << <grid, block, 0, stream1 >> > (d_cost, IMG_W, IMG_H, MAX_DISP, CU_COST_WIN_H);

	//cu_cost_horizontal_filter_new << <grid, block, 0, stream1 >> > (d_cost, d_L1, IMG_W, IMG_H, MAX_DISP, CU_COST_WIN_W);
	//cu_cost_vertical_filter_new << <grid, block, 0, stream2 >> > (d_cost, d_L2, IMG_W, IMG_H, MAX_DISP, CU_COST_WIN_H);
	//cudaStreamSynchronize(stream1);
	//cudaStreamSynchronize(stream2);
	//cu_cost_filter << <grid, block, 0, stream1 >> > (d_cost, d_L1, d_L2, IMG_W, IMG_H, MAX_DISP);

	cudaDeviceSynchronize();
	printf("cost filter takes %lf ms\n", get_cur_ms() - be);

	be = get_cur_ms();

	dim3 dp_grid, dp_block;
	dp_grid.x = IMG_W;
	dp_grid.y = 1;
	dp_block.x = MAX_DISP;  // for dp syncronize
	dp_block.y = 1;

	cu_dp_L1 << <dp_grid, dp_block, 0, stream1 >> > (d_cost, d_L1, d_min_L1, IMG_W, IMG_H, MAX_DISP, P1, P2);
	cu_dp_L2 << <dp_grid, dp_block, 0, stream2 >> > (d_cost, d_L2, d_min_L2, IMG_W, IMG_H, MAX_DISP, P1, P2);
	cu_dp_L3 << <dp_grid, dp_block, 0, stream3 >> > (d_cost, d_L3, d_min_L3, IMG_W, IMG_H, MAX_DISP, P1, P2);
	cu_dp_L4 << <dp_grid, dp_block, 0, stream4 >> > (d_cost, d_L4, d_min_L4, IMG_W, IMG_H, MAX_DISP, P1, P2);
	if (CU_USE_8_PATH)
	{
		//for (int i = 0; i < IMG_H; i++)
		//{
		//	cu_dp_L5 << <dp_grid, dp_block, 0, stream5 >> > (d_cost, d_L5, d_min_L5, i, IMG_W, IMG_H, MAX_DISP, P1, P2);
		//	cu_dp_L6 << <dp_grid, dp_block, 0, stream6 >> > (d_cost, d_L6, d_min_L6, i, IMG_W, IMG_H, MAX_DISP, P1, P2);
		//	cu_dp_L7 << <dp_grid, dp_block, 0, stream7 >> > (d_cost, d_L7, d_min_L7, IMG_H - 1 - i, IMG_W, IMG_H, MAX_DISP, P1, P2);
		//	cu_dp_L8 << <dp_grid, dp_block, 0, stream8 >> > (d_cost, d_L8, d_min_L8, IMG_H - 1 - i, IMG_W, IMG_H, MAX_DISP, P1, P2);
		//}

		// use truncated dp to approximate the original method
		cu_dp_L5_truncated << <dp_grid, dp_block, 0, stream5 >> > (d_cost, d_L5, d_min_L5, IMG_W, IMG_H, MAX_DISP, P1, P2);
		cu_dp_L6_truncated << <dp_grid, dp_block, 0, stream6 >> > (d_cost, d_L6, d_min_L6, IMG_W, IMG_H, MAX_DISP, P1, P2);
		cu_dp_L7_truncated << <dp_grid, dp_block, 0, stream7 >> > (d_cost, d_L7, d_min_L7, IMG_W, IMG_H, MAX_DISP, P1, P2);
		cu_dp_L8_truncated << <dp_grid, dp_block, 0, stream8 >> > (d_cost, d_L8, d_min_L8, IMG_W, IMG_H, MAX_DISP, P1, P2);
	}
	cudaDeviceSynchronize();
	printf("dp takes %lf ms\n", get_cur_ms() - be);

	be = get_cur_ms();

	grid.x = 512;
	grid.y = 512;
	aggregation << <grid, block, 0, stream1 >> > (d_cost, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6, d_L7, d_L8, IMG_W, IMG_H, MAX_DISP);
	grid.x = (IMG_W - 1) / 32 + 1;
	grid.y = (IMG_H - 1) / 32 + 1;
	wta << <grid, block, 0, stream1 >> >(d_cost, d_disp, IMG_W, IMG_H, MAX_DISP, CU_UNIQUE_RATIO, INVALID_DISP);

	cudaDeviceSynchronize();
	printf("wta takes %lf ms\n", get_cur_ms() - be);

	be = get_cur_ms();

	cu_subpixel << <grid, block, 0, stream1 >> > (d_cost, d_disp, d_filtered_disp, IMG_W, IMG_H, MAX_DISP, INVALID_DISP);
	cu_median_filter << <grid, block, 0, stream1 >> > (d_filtered_disp, IMG_W, IMG_H, MAX_DISP, CU_MEDIAN_FILTER_W, CU_MEDIAN_FILTER_H);
	cu_speckle_filter_init << <grid, block, 0, stream2 >> > (d_label, d_area, IMG_W, IMG_H);
	cudaStreamSynchronize(stream1);
	cudaStreamSynchronize(stream2);

	cu_speckle_filter_union_find << <grid, block, 0, stream1 >> > (d_filtered_disp, d_label, d_area, IMG_W, IMG_H, CU_SPECKLE_DIS);
	cu_speckle_filter_sum_up << <grid, block, 0, stream1 >> > (d_label, d_area, IMG_W, IMG_H);
	cu_speckle_filter_end << <grid, block, 0, stream1 >> > (d_filtered_disp, d_label, d_area, IMG_W, IMG_H, INVALID_DISP, CU_SPECKLE_SIZE);

	cudaDeviceSynchronize();
	printf("cuda post_filter takes %lf ms\n", get_cur_ms() - be);

	cudaMemcpyAsync(filtered_disp.data, d_filtered_disp, IMG_H * IMG_W * sizeof(float), cudaMemcpyDeviceToHost, stream1);
	//cudaMemcpyAsync(disp.data, d_disp, IMG_H * IMG_W * sizeof(uchar), cudaMemcpyDeviceToHost, stream2);
}


void GPU_SGM::show_disp()
{
	// left border invalid
	for (int i = 0; i < filtered_disp.rows; i++)
	{
		float *ptr = filtered_disp.ptr<float>(i);
		for (int j = 0; j < MAX_DISP / SCALE; j++)
		{
			ptr[j] = INVALID_DISP;
		}
	}

	// convert to RGB for better observation
	colormap();

	Mat debug_view, tmp;

	debug_view = debug_view.zeros(IMG_H * 2, IMG_W, CV_8UC3);
	tmp = debug_view(Rect(0, 0, IMG_W, IMG_H));
	cvtColor(img_l, img_l, CV_GRAY2BGR);
	img_l.copyTo(tmp);
	tmp = debug_view(Rect(0, IMG_H - 1, IMG_W, IMG_H));
	colored_disp.copyTo(tmp);

	namedWindow("disp_map", 1);
	imshow("disp_map", debug_view);
    imwrite(num2str(disp_cnt++) + "_disp.png", debug_view);

    // waitKey(-1);
	//destroyWindow("disp_map");

}


void GPU_SGM::colormap()
{
	float disp_value = 0;
	for (int i = 0; i < filtered_disp.rows; i++)
	{
		for (int j = 0; j < filtered_disp.cols; j++)
		{
			disp_value = filtered_disp.at<float>(i, j);
			//disp_value = disp.at<uchar>(i, j);
			if (disp_value > MAX_DISP - 1)
			{
				colored_disp.at<Vec3b>(i, j)[0] = 0;
				colored_disp.at<Vec3b>(i, j)[1] = 0;
				colored_disp.at<Vec3b>(i, j)[2] = 0;
			}
			else
			{
				disp_value *= (256 / (MAX_DISP));
				if (disp_value <= 51)
				{
					colored_disp.at<Vec3b>(i, j)[0] = 255;
					colored_disp.at<Vec3b>(i, j)[1] = disp_value * 5;
					colored_disp.at<Vec3b>(i, j)[2] = 0;
				}
				else if (disp_value <= 102)
				{
					disp_value -= 51;
					colored_disp.at<Vec3b>(i, j)[0] = 255 - disp_value * 5;
					colored_disp.at<Vec3b>(i, j)[1] = 255;
					colored_disp.at<Vec3b>(i, j)[2] = 0;
				}
				else if (disp_value <= 153)
				{
					disp_value -= 102;
					colored_disp.at<Vec3b>(i, j)[0] = 0;
					colored_disp.at<Vec3b>(i, j)[1] = 255;
					colored_disp.at<Vec3b>(i, j)[2] = disp_value * 5;
				}
				else if (disp_value <= 204)
				{
					disp_value -= 153;
					colored_disp.at<Vec3b>(i, j)[0] = 0;
					colored_disp.at<Vec3b>(i, j)[1] = 255 - uchar(128.0*disp_value / 51.0 + 0.5);
					colored_disp.at<Vec3b>(i, j)[2] = 255;
				}
				else
				{
					disp_value -= 204;
					colored_disp.at<Vec3b>(i, j)[0] = 0;
					colored_disp.at<Vec3b>(i, j)[1] = 127 - uchar(127.0*disp_value / 51.0 + 0.5);
					colored_disp.at<Vec3b>(i, j)[2] = 255;
				}
			}
		}
	}
}
