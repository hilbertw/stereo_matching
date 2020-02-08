#include "../inc/SGM.cuh"


GPU_SGM::GPU_SGM(int h, int w, int s, int d)
{
    img_h = h/s;
    img_w = w/s;
    scale = s;
    max_disp = d;
    invalid_disp = d+1;

	cudaSetDevice(0);
	checkCudaErrors(cudaStreamCreate(&stream1));
	checkCudaErrors(cudaStreamCreate(&stream2));
	checkCudaErrors(cudaStreamCreate(&stream3));
	checkCudaErrors(cudaStreamCreate(&stream4));
	checkCudaErrors(cudaStreamCreate(&stream5));
	checkCudaErrors(cudaStreamCreate(&stream6));
	checkCudaErrors(cudaStreamCreate(&stream7));
	checkCudaErrors(cudaStreamCreate(&stream8));

    checkCudaErrors(cudaMalloc((void**)&d_img_l, img_h* img_w * sizeof(uchar)));
    checkCudaErrors(cudaMalloc((void**)&d_img_r, img_h * img_w * sizeof(uchar)));
    checkCudaErrors(cudaMalloc((void**)&d_disp, img_h * img_w * sizeof(uchar)));
    checkCudaErrors(cudaMalloc((void**)&d_filtered_disp, img_h * img_w * sizeof(float)));

    checkCudaErrors(cudaMalloc((void**)&d_cost_table_l, img_h * img_w * sizeof(uint64_t)));
    checkCudaErrors(cudaMalloc((void**)&d_cost_table_r, img_h * img_w * sizeof(uint64_t)));
    checkCudaErrors(cudaMalloc((void**)&d_cost, img_h * img_w * max_disp * sizeof(float)));

    checkCudaErrors(cudaMalloc((void**)&d_L1, img_h * img_w * max_disp * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&d_L2, img_h * img_w * max_disp * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&d_min_L1, img_h * img_w * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&d_min_L2, img_h * img_w * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&d_L3, img_h * img_w * max_disp * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&d_L4, img_h * img_w * max_disp * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&d_min_L3, img_h * img_w * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&d_min_L4, img_h * img_w * sizeof(float)));

    checkCudaErrors(cudaMalloc((void**)&d_L5, img_h * img_w * max_disp * sizeof(short)));
    checkCudaErrors(cudaMalloc((void**)&d_L6, img_h * img_w * max_disp * sizeof(short)));
    checkCudaErrors(cudaMalloc((void**)&d_min_L5, img_h * img_w * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&d_min_L6, img_h * img_w * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&d_L7, img_h * img_w * max_disp * sizeof(short)));
    checkCudaErrors(cudaMalloc((void**)&d_L8, img_h * img_w * max_disp * sizeof(short)));
    checkCudaErrors(cudaMalloc((void**)&d_min_L7, img_h * img_w * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&d_min_L8, img_h * img_w * sizeof(float)));

	P1 = 10;
	P2 = 100;

    checkCudaErrors(cudaMalloc((void**)&d_label, img_h * img_w * sizeof(int)));
    checkCudaErrors(cudaMalloc((void**)&d_area, img_h * img_w * sizeof(int)));

    disp.create(img_h, img_w, CV_8UC1);
    filtered_disp.create(img_h, img_w, CV_32FC1);
    colored_disp.create(img_h, img_w, CV_8UC3);
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
    assert (img_l.rows == img_r.rows);
    assert (img_l.cols == img_r.cols);
    assert (img_l.channels() == img_r.channels());
    assert (img_l.type() == img_r.type());
    assert (img_l.type() == CV_8UC1);

    if (scale > 1)
    {
        this->img_l.create(img_h, img_w, CV_8UC1);
        this->img_r.create(img_h, img_w, CV_8UC1);
        for (int i=0; i<img_h; ++i)
        {
            const uchar *ptr_l = img_l.ptr<uchar>(i*scale);
            const uchar *ptr_r = img_r.ptr<uchar>(i*scale);
            uchar *ptr_l_new = this->img_l.ptr<uchar>(i);
            uchar *ptr_r_new = this->img_r.ptr<uchar>(i);
            for (int j=0; j<img_w; ++j)
            {
                ptr_l_new[j] = ptr_l[j*scale];
                ptr_r_new[j] = ptr_r[j*scale];
            }
        }
    }
    else
    {
        this->img_l = img_l;
        this->img_r = img_r;
    }

	cudaSetDevice(0);

    cudaMemcpyAsync(d_img_l, this->img_l.data, img_h* img_w * sizeof(uchar), cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(d_img_r, this->img_r.data, img_h* img_w * sizeof(uchar), cudaMemcpyHostToDevice, stream2);
	cudaStreamSynchronize(stream1);
	cudaStreamSynchronize(stream2);

	double be = get_cur_ms();

	dim3 grid, block;
    grid.x = (img_w - 1) / 32 + 1;
    grid.y = (img_h - 1) / 32 + 1;
	block.x = 32;
	block.y = 32;
    cu_build_cost_table << <grid, block, 0, stream1 >> > (d_img_l, d_img_r, d_cost_table_l, d_cost_table_r, img_w, img_h, CU_WIN_W/scale, CU_WIN_H/scale);
    cu_build_dsi_from_table << <grid, block, 0, stream1 >> > (d_cost_table_l, d_cost_table_r, d_cost, img_w, img_h, scale, max_disp);

	cudaDeviceSynchronize();
	printf("build cost takes %lf ms\n", get_cur_ms() - be);

	be = get_cur_ms();

    grid.x = (img_w - 1) / 32 + 1;
    grid.y = (max_disp - 1) / 32 + 1;
    cu_cost_horizontal_filter << <grid, block, 0, stream1 >> > (d_cost, img_w, img_h, max_disp, CU_COST_WIN_W/scale);
    cu_cost_vertical_filter << <grid, block, 0, stream1 >> > (d_cost, img_w, img_h, max_disp, CU_COST_WIN_H/scale);

    //cu_cost_horizontal_filter_new << <grid, block, 0, stream1 >> > (d_cost, d_L1, img_w, img_h, max_disp, CU_COST_WIN_W/scale);
    //cu_cost_vertical_filter_new << <grid, block, 0, stream2 >> > (d_cost, d_L2, img_w, img_h, max_disp, CU_COST_WIN_H/scale);
	//cudaStreamSynchronize(stream1);
	//cudaStreamSynchronize(stream2);
    //cu_cost_filter << <grid, block, 0, stream1 >> > (d_cost, d_L1, d_L2, img_w, img_h, max_disp);

	cudaDeviceSynchronize();
	printf("cost filter takes %lf ms\n", get_cur_ms() - be);

	be = get_cur_ms();

	dim3 dp_grid, dp_block;
    dp_grid.x = img_w;
	dp_grid.y = 1;
    dp_block.x = max_disp;  // for dp syncronize
	dp_block.y = 1;

    cu_dp_L1 << <dp_grid, dp_block, 0, stream1 >> > (d_cost, d_L1, d_min_L1, img_w, img_h, max_disp, P1, P2);
    cu_dp_L2 << <dp_grid, dp_block, 0, stream2 >> > (d_cost, d_L2, d_min_L2, img_w, img_h, max_disp, P1, P2);
    cu_dp_L3 << <dp_grid, dp_block, 0, stream3 >> > (d_cost, d_L3, d_min_L3, img_w, img_h, max_disp, P1, P2);
    cu_dp_L4 << <dp_grid, dp_block, 0, stream4 >> > (d_cost, d_L4, d_min_L4, img_w, img_h, max_disp, P1, P2);
	if (CU_USE_8_PATH)
	{
        //for (int i = 0; i < img_h; i++)
		//{
        //	cu_dp_L5 << <dp_grid, dp_block, 0, stream5 >> > (d_cost, d_L5, d_min_L5, i, img_w, img_h, max_disp, P1, P2);
        //	cu_dp_L6 << <dp_grid, dp_block, 0, stream6 >> > (d_cost, d_L6, d_min_L6, i, img_w, img_h, max_disp, P1, P2);
        //	cu_dp_L7 << <dp_grid, dp_block, 0, stream7 >> > (d_cost, d_L7, d_min_L7, img_h - 1 - i, img_w, img_h, max_disp, P1, P2);
        //	cu_dp_L8 << <dp_grid, dp_block, 0, stream8 >> > (d_cost, d_L8, d_min_L8, img_h - 1 - i, img_w, img_h, max_disp, P1, P2);
		//}

		// use truncated dp to approximate the original method
        cu_dp_L5_truncated << <dp_grid, dp_block, 0, stream5 >> > (d_cost, d_L5, d_min_L5, img_w, img_h, max_disp, P1, P2);
        cu_dp_L6_truncated << <dp_grid, dp_block, 0, stream6 >> > (d_cost, d_L6, d_min_L6, img_w, img_h, max_disp, P1, P2);
        cu_dp_L7_truncated << <dp_grid, dp_block, 0, stream7 >> > (d_cost, d_L7, d_min_L7, img_w, img_h, max_disp, P1, P2);
        cu_dp_L8_truncated << <dp_grid, dp_block, 0, stream8 >> > (d_cost, d_L8, d_min_L8, img_w, img_h, max_disp, P1, P2);
	}
	cudaDeviceSynchronize();
	printf("dp takes %lf ms\n", get_cur_ms() - be);

	be = get_cur_ms();

	grid.x = 512;
	grid.y = 512;
    aggregation << <grid, block, 0, stream1 >> > (d_cost, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6, d_L7, d_L8, img_w, img_h, max_disp);
    grid.x = (img_w - 1) / 32 + 1;
    grid.y = (img_h - 1) / 32 + 1;
    wta << <grid, block, 0, stream1 >> >(d_cost, d_disp, img_w, img_h, max_disp, CU_UNIQUE_RATIO, invalid_disp);

	cudaDeviceSynchronize();
	printf("wta takes %lf ms\n", get_cur_ms() - be);

	be = get_cur_ms();

    cu_subpixel << <grid, block, 0, stream1 >> > (d_cost, d_disp, d_filtered_disp, img_w, img_h, max_disp, invalid_disp);
    cu_median_filter << <grid, block, 0, stream1 >> > (d_filtered_disp, img_w, img_h, max_disp, CU_MEDIAN_FILTER_W, CU_MEDIAN_FILTER_H);
    cu_speckle_filter_init << <grid, block, 0, stream2 >> > (d_label, d_area, img_w, img_h);
	cudaStreamSynchronize(stream1);
	cudaStreamSynchronize(stream2);

    cu_speckle_filter_union_find << <grid, block, 0, stream1 >> > (d_filtered_disp, d_label, d_area, img_w, img_h, CU_SPECKLE_DIS);
    cu_speckle_filter_sum_up << <grid, block, 0, stream1 >> > (d_label, d_area, img_w, img_h);
    cu_speckle_filter_end << <grid, block, 0, stream1 >> > (d_filtered_disp, d_label, d_area, img_w, img_h, invalid_disp, CU_SPECKLE_SIZE/scale);

	cudaDeviceSynchronize();
	printf("cuda post_filter takes %lf ms\n", get_cur_ms() - be);

    cudaMemcpyAsync(filtered_disp.data, d_filtered_disp, img_h * img_w * sizeof(float), cudaMemcpyDeviceToHost, stream1);
    //cudaMemcpyAsync(disp.data, d_disp, img_h * img_w * sizeof(uchar), cudaMemcpyDeviceToHost, stream2);
}


void GPU_SGM::show_disp(Mat &debug_view)
{
	// left border invalid
	for (int i = 0; i < filtered_disp.rows; i++)
	{
		float *ptr = filtered_disp.ptr<float>(i);
        for (int j = 0; j < max_disp / scale; j++)
		{
            ptr[j] = invalid_disp;
		}
	}

	// convert to RGB for better observation
	colormap();

	Mat tmp;

    debug_view = debug_view.zeros(img_h * 2, img_w, CV_8UC3);
    tmp = debug_view(Rect(0, 0, img_w, img_h));
	cvtColor(img_l, img_l, CV_GRAY2BGR);
	img_l.copyTo(tmp);
    tmp = debug_view(Rect(0, img_h - 1, img_w, img_h));
	colored_disp.copyTo(tmp);
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
            if (disp_value > max_disp - 1)
			{
				colored_disp.at<Vec3b>(i, j)[0] = 0;
				colored_disp.at<Vec3b>(i, j)[1] = 0;
				colored_disp.at<Vec3b>(i, j)[2] = 0;
			}
			else
			{
                disp_value *= (256 / (max_disp));
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
