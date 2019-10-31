#include "../cpu_inc/SGM.h"


SGM::SGM(int h, int w, int s, int d)
    : Solver(h, w, s, d)
{
    L1 = new float[img_h * img_w * max_disp];
    L2 = new float[img_h * img_w * max_disp];
    L3 = new float[img_h * img_w * max_disp];
    L4 = new float[img_h * img_w * max_disp];
    min_L1 = new float[img_h * img_w];
    min_L2 = new float[img_h * img_w];
    min_L3 = new float[img_h * img_w];
    min_L4 = new float[img_h * img_w];
	if (USE_8_PATH)
	{
        L5 = new float[img_h * img_w * max_disp];
        L6 = new float[img_h * img_w * max_disp];
        L7 = new float[img_h * img_w * max_disp];
        L8 = new float[img_h * img_w * max_disp];
        min_L5 = new float[img_h * img_w];
        min_L6 = new float[img_h * img_w];
        min_L7 = new float[img_h * img_w];
        min_L8 = new float[img_h * img_w];
	}

	P1 = 10;
	P2 = 100;
}


void SGM::process(Mat &img_l, Mat &img_r)
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

    printf("process image size: %d, %d\n", this->img_l.rows, this->img_l.cols);

	double be = get_cur_ms();
	build_cost_table();
	build_dsi_from_table();
	printf("build cost takes %lf ms\n", get_cur_ms() - be);

	be = get_cur_ms();
    cost_horizontal_filter(COST_WIN_W/scale);
    cost_vertical_filter(COST_WIN_H/scale);
	printf("cost filter takes %lf ms\n", get_cur_ms() - be);

	be = get_cur_ms();

    //find_table_mean_max();
    //find_dsi_mean_max();


    // build L1: left -> right
#pragma omp parallel for
    for (int i = 0; i < img_h; ++i)
	{
        for (int j = 0; j < img_w; ++j)
        {
            int index_L1_prev = i * img_w * max_disp + (j - 1) * max_disp;

			// DP
			float minL1 = FLT_MAX;
            int bias = i * img_w * max_disp + j * max_disp;

            for (int d = 0; d < max_disp; ++d)
			{
                int index = bias + d;
                uchar d_sub_1 = MAX(d - 1, 0);
                uchar d_plus_1 = MIN(d + 1, max_disp - 1);

                if (j == 0)  //init
                {
                    L1[index] = cost[index];
                }
                else
                {
                    L1[index] = MIN(L1[index_L1_prev + d], L1[index_L1_prev + d_sub_1] + P1);
                    L1[index] = MIN(L1[index], L1[index_L1_prev + d_plus_1] + P1);
                    L1[index] = MIN(L1[index], min_L1[i * img_w + j - 1] + P2);
                    L1[index] += (cost[index] - min_L1[i * img_w + j - 1]);
                }
                if (L1[index] < minL1)
                {
                    minL1 = L1[index];
                }
			}
			
			// update minL1
            min_L1[i * img_w + j] = minL1;
        }
	}

	// build L2: right -> left
#pragma omp parallel for
    for (int i = 0; i < img_h; ++i)
	{
        for (int j = img_w - 1; j >=0; --j)
		{
            int index_L2_prev = i * img_w * max_disp + (j + 1) * max_disp;

			// DP
            float minL2 = FLT_MAX;
            int bias = i * img_w * max_disp + j * max_disp;

            for (int d = 0; d < max_disp; ++d)
			{
                int index = bias + d;
                uchar d_sub_1 = MAX(d - 1, 0);
                uchar d_plus_1 = MIN(d + 1, max_disp - 1);

                if (j == img_w - 1)  //init
				{
					L2[index] = cost[index];
				}
				else
				{
					L2[index] = MIN(L2[index_L2_prev + d], L2[index_L2_prev + d_sub_1] + P1);
					L2[index] = MIN(L2[index], L2[index_L2_prev + d_plus_1] + P1);
                    L2[index] = MIN(L2[index], min_L2[i * img_w + j + 1] + P2);
                    L2[index] += (cost[index] - min_L2[i * img_w + j +1]);
				}
				if (L2[index] < minL2)
				{
					minL2 = L2[index];
				}
			}

			// update minL2
            min_L2[i * img_w + j] = minL2;
		}
	}

	// build L3: top -> down
#pragma omp parallel for
    for (int j = 0; j < img_w; ++j)
    {
        for (int i = 0; i < img_h; ++i)
		{
            int index_L3_prev = (i - 1) * img_w * max_disp + j * max_disp;

			// DP
			float minL3 = FLT_MAX;
            int bias = i * img_w * max_disp + j * max_disp;

            for (int d = 0; d < max_disp; ++d)
			{
                int index = bias + d;
                uchar d_sub_1 = MAX(d - 1, 0);
                uchar d_plus_1 = MIN(d + 1, max_disp - 1);

                if (i == 0)  //init
				{
					L3[index] = cost[index];
				}
				else
				{
					L3[index] = MIN(L3[index_L3_prev + d], L3[index_L3_prev + d_sub_1] + P1);
					L3[index] = MIN(L3[index], L3[index_L3_prev + d_plus_1] + P1);
                    L3[index] = MIN(L3[index], min_L3[(i - 1) * img_w + j ] + P2);
                    L3[index] += (cost[index] - min_L3[(i - 1) * img_w + j]);
				}
				if (L3[index] < minL3)
				{
					minL3 = L3[index];
				}
			}

			// update minL3
            min_L3[i * img_w + j] = minL3;
		}
	}

	// build L4: down -> top
#pragma omp parallel for
    for (int j = 0; j < img_w; ++j)
	{
        for (int i = img_h - 1; i >=0; --i)
		{
            int index_L4_prev = (i + 1) * img_w * max_disp + j * max_disp;

			// DP
            float minL4 = FLT_MAX;
            int bias = i * img_w * max_disp + j * max_disp;

            for (int d = 0; d < max_disp; ++d)
			{
                int index = bias + d;
                uchar d_sub_1 = MAX(d - 1, 0);
                uchar d_plus_1 = MIN(d + 1, max_disp - 1);

                if (i == img_h - 1)  //init
				{
					L4[index] = cost[index];
				}
				else
				{
					L4[index] = MIN(L4[index_L4_prev + d], L4[index_L4_prev + d_sub_1] + P1);
					L4[index] = MIN(L4[index], L4[index_L4_prev + d_plus_1] + P1);
                    L4[index] = MIN(L4[index], min_L4[(i + 1) * img_w + j] + P2);
                    L4[index] += (cost[index] - min_L4[(i + 1) * img_w + j]);
				}
				if (L4[index] < minL4)
				{
					minL4 = L4[index];
				}
			}

			// update minL4
            min_L4[i * img_w + j] = minL4;
		}
	}

    if (USE_8_PATH)
    {

        // build L5: lefttop -> rightdown
        // build L6: righttop -> leftdown

        for (int i = 0; i < img_h; ++i)
        {
#pragma omp parallel for
            for (int j = 0; j < img_w; ++j)
            {
                int index_L5_prev = (i - 1) * img_w * max_disp + (j - 1) * max_disp;
                int index_L6_prev = (i - 1) * img_w * max_disp + (j + 1) * max_disp;

                // DP
                float minL5 = FLT_MAX;
                float minL6 = FLT_MAX;
                int bias = i * img_w * max_disp + j * max_disp;

                for (int d = 0; d < max_disp; ++d)
                {
                    int index = bias + d;
                    uchar d_sub_1 = MAX(d - 1, 0);
                    uchar d_plus_1 = MIN(d + 1, max_disp - 1);

                    if (i == 0 || j == 0)  //init
                    {
                        L5[index] = cost[index];
                    }
                    else
                    {
                        L5[index] = MIN(L5[index_L5_prev + d], L5[index_L5_prev + d_sub_1] + P1);
                        L5[index] = MIN(L5[index], L5[index_L5_prev + d_plus_1] + P1);
                        L5[index] = MIN(L5[index], min_L5[(i - 1) * img_w + j - 1] + P2);
                        L5[index] += (cost[index] - min_L5[(i - 1) * img_w + j - 1]);
                    }
                    if (L5[index] < minL5)
                    {
                        minL5 = L5[index];
                    }

                    if (i == 0 || j == img_w - 1)  //init
                    {
                        L6[index] = cost[index];
                    }
                    else
                    {
                        L6[index] = MIN(L6[index_L6_prev + d], L6[index_L6_prev + d_sub_1] + P1);
                        L6[index] = MIN(L6[index], L6[index_L6_prev + d_plus_1] + P1);
                        L6[index] = MIN(L6[index], min_L6[(i - 1) * img_w + j + 1] + P2);
                        L6[index] += (cost[index] - min_L6[(i - 1) * img_w + j + 1]);
                    }
                    if (L6[index] < minL6)
                    {
                        minL6 = L6[index];
                    }
                }

                // update minL5
                min_L5[i * img_w + j] = minL5;

                // update minL6
                min_L6[i * img_w + j] = minL6;
            }
        }


        // build L7: leftdown -> righttop
        // build L8: rightdown -> lefttop

        for (int i = img_h - 1; i >=0; --i)
        {
#pragma omp parallel for
            for (int j = 0; j < img_w; ++j)
            {
                int index_L7_prev = (i + 1) * img_w * max_disp + (j - 1) * max_disp;
                int index_L8_prev = (i + 1) * img_w * max_disp + (j + 1) * max_disp;

                // DP
                float minL7 = FLT_MAX;
                float minL8 = FLT_MAX;
                int bias = i * img_w * max_disp + j * max_disp;

                for (int d = 0; d < max_disp; ++d)
                {
                    int index = bias + d;
                    uchar d_sub_1 = MAX(d - 1, 0);
                    uchar d_plus_1 = MIN(d + 1, max_disp - 1);

                    if (i == img_h - 1 || j == 0)  //init
                    {
                        L7[index] = cost[index];
                    }
                    else
                    {
                        L7[index] = MIN(L7[index_L7_prev + d], L7[index_L7_prev + d_sub_1] + P1);
                        L7[index] = MIN(L7[index], L7[index_L7_prev + d_plus_1] + P1);
                        L7[index] = MIN(L7[index], min_L7[(i + 1) * img_w + j - 1] + P2);
                        L7[index] += (cost[index] - min_L7[(i + 1) * img_w + j - 1]);
                    }
                    if (L7[index] < minL7)
                    {
                        minL7 = L7[index];
                    }

                    if (i == img_h - 1 || j == img_w - 1)  //init
                    {
                        L8[index] = cost[index];
                    }
                    else
                    {
                        L8[index] = MIN(L8[index_L8_prev + d], L8[index_L8_prev + d_sub_1] + P1);
                        L8[index] = MIN(L8[index], L8[index_L8_prev + d_plus_1] + P1);
                        L8[index] = MIN(L8[index], min_L8[(i + 1) * img_w + j + 1] + P2);
                        L8[index] += (cost[index] - min_L8[(i + 1) * img_w + j + 1]);
                    }
                    if (L8[index] < minL8)
                    {
                        minL8 = L8[index];
                    }
                }

                // update minL7
                min_L7[i * img_w + j] = minL7;

                // update minL8
                min_L8[i * img_w + j] = minL8;
            }
        }
    }
	
	// cost aggregation
	uchar *ptr = NULL;
	float min_cost = FLT_MAX, sec_min_cost = FLT_MAX;
    uchar min_d = invalid_disp, sec_min_d = invalid_disp;
    for (int i = 0; i < img_h; ++i)
	{
		ptr = disp.ptr<uchar>(i);
        for (int j = 0; j < img_w; ++j)
		{
			min_cost = FLT_MAX;
            int index_bias = i * img_w * max_disp + j * max_disp;
            for (int d = 0; d < max_disp; ++d)
			{
                int index = index_bias + d;
				cost[index] = L1[index] + L2[index] + L3[index] + L4[index];
				if (USE_8_PATH)
				{
					cost[index] += (L5[index] + L6[index] + L7[index] + L8[index]);
				}
				// wta
				if (cost[index] < min_cost)
				{
					min_cost = cost[index];
					min_d = d;
				}
			}
			// unique check
			sec_min_cost = FLT_MAX;
            for (int d = 0; d < max_disp; ++d)
			{
                int index = index_bias + d;
				if (cost[index] < sec_min_cost && cost[index] != min_cost)
				{
					sec_min_cost = cost[index];
					sec_min_d = d;
				}
			}
			if (min_cost / sec_min_cost > UNIQUE_RATIO && abs(min_d - sec_min_d) > 1)
			{
                ptr[j] = invalid_disp;
			}
			else
			{
				ptr[j] = min_d;
			}
		}
	}
	ptr = NULL;

	printf("dp takes %lf ms\n", get_cur_ms() - be);

//    if (sky_mask.empty() || !sky_mask.data)
//    {
//        printf("no sky mask\n");
//    }
//    else
//    {
//        printf("sky mask found\n");
//        for (int i = 0; i < img_h/2+1; ++i)
//        {
//            uchar *ptr = disp.ptr<uchar>(i);
//            const uchar *ptr_sky = sky_mask.ptr<uchar>(i);

//            for (int j = 0; j < img_w; ++j)
//            {
//                if (ptr_sky[j] == 255)
//                    ptr[j] = 0;
//            }
//        }
//    }

    compute_subpixel(disp, filtered_disp);


    /* ------------------------------*/

    be = get_cur_ms();
    build_dsi_from_table_beta();
    printf("build cost beta takes %lf ms\n", get_cur_ms() - be);

    be = get_cur_ms();
    cost_horizontal_filter(COST_WIN_W/scale);
    cost_vertical_filter(COST_WIN_H/scale);
    printf("cost filter beta takes %lf ms\n", get_cur_ms() - be);

    be = get_cur_ms();

    // build L1: left -> right
#pragma omp parallel for
    for (int i = 0; i < img_h; ++i)
    {
        for (int j = 0; j < img_w; ++j)
        {
            int index_L1_prev = i * img_w * max_disp + (j - 1) * max_disp;

            // DP
            float minL1 = FLT_MAX;
            int bias = i * img_w * max_disp + j * max_disp;

            for (int d = 0; d < max_disp; ++d)
            {
                int index = bias + d;
                uchar d_sub_1 = MAX(d - 1, 0);
                uchar d_plus_1 = MIN(d + 1, max_disp - 1);

                if (j == 0)  //init
                {
                    L1[index] = cost[index];
                }
                else
                {
                    L1[index] = MIN(L1[index_L1_prev + d], L1[index_L1_prev + d_sub_1] + P1);
                    L1[index] = MIN(L1[index], L1[index_L1_prev + d_plus_1] + P1);
                    L1[index] = MIN(L1[index], min_L1[i * img_w + j - 1] + P2);
                    L1[index] += (cost[index] - min_L1[i * img_w + j - 1]);
                }
                if (L1[index] < minL1)
                {
                    minL1 = L1[index];
                }
            }

            // update minL1
            min_L1[i * img_w + j] = minL1;
        }
    }

    // build L2: right -> left
#pragma omp parallel for
    for (int i = 0; i < img_h; ++i)
    {
        for (int j = img_w - 1; j >=0; --j)
        {
            int index_L2_prev = i * img_w * max_disp + (j + 1) * max_disp;

            // DP
            float minL2 = FLT_MAX;
            int bias = i * img_w * max_disp + j * max_disp;

            for (int d = 0; d < max_disp; ++d)
            {
                int index = bias + d;
                uchar d_sub_1 = MAX(d - 1, 0);
                uchar d_plus_1 = MIN(d + 1, max_disp - 1);

                if (j == img_w - 1)  //init
                {
                    L2[index] = cost[index];
                }
                else
                {
                    L2[index] = MIN(L2[index_L2_prev + d], L2[index_L2_prev + d_sub_1] + P1);
                    L2[index] = MIN(L2[index], L2[index_L2_prev + d_plus_1] + P1);
                    L2[index] = MIN(L2[index], min_L2[i * img_w + j + 1] + P2);
                    L2[index] += (cost[index] - min_L2[i * img_w + j +1]);
                }
                if (L2[index] < minL2)
                {
                    minL2 = L2[index];
                }
            }

            // update minL2
            min_L2[i * img_w + j] = minL2;
        }
    }

    // build L3: top -> down
#pragma omp parallel for
    for (int j = 0; j < img_w; ++j)
    {
        for (int i = 0; i < img_h; ++i)
        {
            int index_L3_prev = (i - 1) * img_w * max_disp + j * max_disp;

            // DP
            float minL3 = FLT_MAX;
            int bias = i * img_w * max_disp + j * max_disp;

            for (int d = 0; d < max_disp; ++d)
            {
                int index = bias + d;
                uchar d_sub_1 = MAX(d - 1, 0);
                uchar d_plus_1 = MIN(d + 1, max_disp - 1);

                if (i == 0)  //init
                {
                    L3[index] = cost[index];
                }
                else
                {
                    L3[index] = MIN(L3[index_L3_prev + d], L3[index_L3_prev + d_sub_1] + P1);
                    L3[index] = MIN(L3[index], L3[index_L3_prev + d_plus_1] + P1);
                    L3[index] = MIN(L3[index], min_L3[(i - 1) * img_w + j ] + P2);
                    L3[index] += (cost[index] - min_L3[(i - 1) * img_w + j]);
                }
                if (L3[index] < minL3)
                {
                    minL3 = L3[index];
                }
            }

            // update minL3
            min_L3[i * img_w + j] = minL3;
        }
    }

    // build L4: down -> top
#pragma omp parallel for
    for (int j = 0; j < img_w; ++j)
    {
        for (int i = img_h - 1; i >=0; --i)
        {
            int index_L4_prev = (i + 1) * img_w * max_disp + j * max_disp;

            // DP
            float minL4 = FLT_MAX;
            int bias = i * img_w * max_disp + j * max_disp;

            for (int d = 0; d < max_disp; ++d)
            {
                int index = bias + d;
                uchar d_sub_1 = MAX(d - 1, 0);
                uchar d_plus_1 = MIN(d + 1, max_disp - 1);

                if (i == img_h - 1)  //init
                {
                    L4[index] = cost[index];
                }
                else
                {
                    L4[index] = MIN(L4[index_L4_prev + d], L4[index_L4_prev + d_sub_1] + P1);
                    L4[index] = MIN(L4[index], L4[index_L4_prev + d_plus_1] + P1);
                    L4[index] = MIN(L4[index], min_L4[(i + 1) * img_w + j] + P2);
                    L4[index] += (cost[index] - min_L4[(i + 1) * img_w + j]);
                }
                if (L4[index] < minL4)
                {
                    minL4 = L4[index];
                }
            }

            // update minL4
            min_L4[i * img_w + j] = minL4;
        }
    }

    if (USE_8_PATH)
    {

        // build L5: lefttop -> rightdown
        // build L6: righttop -> leftdown

        for (int i = 0; i < img_h; ++i)
        {
#pragma omp parallel for
            for (int j = 0; j < img_w; ++j)
            {
                int index_L5_prev = (i - 1) * img_w * max_disp + (j - 1) * max_disp;
                int index_L6_prev = (i - 1) * img_w * max_disp + (j + 1) * max_disp;

                // DP
                float minL5 = FLT_MAX;
                float minL6 = FLT_MAX;
                int bias = i * img_w * max_disp + j * max_disp;

                for (int d = 0; d < max_disp; ++d)
                {
                    int index = bias + d;
                    uchar d_sub_1 = MAX(d - 1, 0);
                    uchar d_plus_1 = MIN(d + 1, max_disp - 1);

                    if (i == 0 || j == 0)  //init
                    {
                        L5[index] = cost[index];
                    }
                    else
                    {
                        L5[index] = MIN(L5[index_L5_prev + d], L5[index_L5_prev + d_sub_1] + P1);
                        L5[index] = MIN(L5[index], L5[index_L5_prev + d_plus_1] + P1);
                        L5[index] = MIN(L5[index], min_L5[(i - 1) * img_w + j - 1] + P2);
                        L5[index] += (cost[index] - min_L5[(i - 1) * img_w + j - 1]);
                    }
                    if (L5[index] < minL5)
                    {
                        minL5 = L5[index];
                    }

                    if (i == 0 || j == img_w - 1)  //init
                    {
                        L6[index] = cost[index];
                    }
                    else
                    {
                        L6[index] = MIN(L6[index_L6_prev + d], L6[index_L6_prev + d_sub_1] + P1);
                        L6[index] = MIN(L6[index], L6[index_L6_prev + d_plus_1] + P1);
                        L6[index] = MIN(L6[index], min_L6[(i - 1) * img_w + j + 1] + P2);
                        L6[index] += (cost[index] - min_L6[(i - 1) * img_w + j + 1]);
                    }
                    if (L6[index] < minL6)
                    {
                        minL6 = L6[index];
                    }
                }

                // update minL5
                min_L5[i * img_w + j] = minL5;

                // update minL6
                min_L6[i * img_w + j] = minL6;
            }
        }


        // build L7: leftdown -> righttop
        // build L8: rightdown -> lefttop

        for (int i = img_h - 1; i >=0; --i)
        {
#pragma omp parallel for
            for (int j = 0; j < img_w; ++j)
            {
                int index_L7_prev = (i + 1) * img_w * max_disp + (j - 1) * max_disp;
                int index_L8_prev = (i + 1) * img_w * max_disp + (j + 1) * max_disp;

                // DP
                float minL7 = FLT_MAX;
                float minL8 = FLT_MAX;
                int bias = i * img_w * max_disp + j * max_disp;

                for (int d = 0; d < max_disp; ++d)
                {
                    int index = bias + d;
                    uchar d_sub_1 = MAX(d - 1, 0);
                    uchar d_plus_1 = MIN(d + 1, max_disp - 1);

                    if (i == img_h - 1 || j == 0)  //init
                    {
                        L7[index] = cost[index];
                    }
                    else
                    {
                        L7[index] = MIN(L7[index_L7_prev + d], L7[index_L7_prev + d_sub_1] + P1);
                        L7[index] = MIN(L7[index], L7[index_L7_prev + d_plus_1] + P1);
                        L7[index] = MIN(L7[index], min_L7[(i + 1) * img_w + j - 1] + P2);
                        L7[index] += (cost[index] - min_L7[(i + 1) * img_w + j - 1]);
                    }
                    if (L7[index] < minL7)
                    {
                        minL7 = L7[index];
                    }

                    if (i == img_h - 1 || j == img_w - 1)  //init
                    {
                        L8[index] = cost[index];
                    }
                    else
                    {
                        L8[index] = MIN(L8[index_L8_prev + d], L8[index_L8_prev + d_sub_1] + P1);
                        L8[index] = MIN(L8[index], L8[index_L8_prev + d_plus_1] + P1);
                        L8[index] = MIN(L8[index], min_L8[(i + 1) * img_w + j + 1] + P2);
                        L8[index] += (cost[index] - min_L8[(i + 1) * img_w + j + 1]);
                    }
                    if (L8[index] < minL8)
                    {
                        minL8 = L8[index];
                    }
                }

                // update minL7
                min_L7[i * img_w + j] = minL7;

                // update minL8
                min_L8[i * img_w + j] = minL8;
            }
        }
    }

    // cost aggregation   
    ptr = NULL;
    min_cost = FLT_MAX, sec_min_cost = FLT_MAX;
    min_d = invalid_disp, sec_min_d = invalid_disp;
    for (int i = 0; i < img_h; ++i)
    {
        ptr = disp_beta.ptr<uchar>(i);
        for (int j = 0; j < img_w; ++j)
        {
            min_cost = FLT_MAX;
            int index_bias = i * img_w * max_disp + j * max_disp;
            for (int d = 0; d < max_disp; ++d)
            {
                int index = index_bias + d;
                cost[index] = L1[index] + L2[index] + L3[index] + L4[index];
                if (USE_8_PATH)
                {
                    cost[index] += (L5[index] + L6[index] + L7[index] + L8[index]);
                }
                // wta
                if (cost[index] < min_cost)
                {
                    min_cost = cost[index];
                    min_d = d;
                }
            }
            // unique check
            sec_min_cost = FLT_MAX;
            for (int d = 0; d < max_disp; ++d)
            {
                int index = index_bias + d;
                if (cost[index] < sec_min_cost && cost[index] != min_cost)
                {
                    sec_min_cost = cost[index];
                    sec_min_d = d;
                }
            }
            if (min_cost / sec_min_cost > UNIQUE_RATIO && abs(min_d - sec_min_d) > 1)
            {
                ptr[j] = invalid_disp;
            }
            else
            {
                ptr[j] = min_d;
            }
        }
    }
    ptr = NULL;

    printf("dp beta takes %lf ms\n", get_cur_ms() - be);

    compute_subpixel(disp_beta, filtered_disp_beta);

    // lr_check based on subpixel disparity
#pragma omp parallel for
    for (int i = 0; i < img_h; ++i)
    {
        for (int j = 0; j < img_w; ++j)
        {
            float &dl = filtered_disp.at<float>(i,j);
            if (j>=dl)
            {
                const float &dr = filtered_disp_beta.at<float>(i,j-dl/scale);

                if (fabs(dl-dr) > LR_CHECK_DIS)
                    dl = invalid_disp;
            }
        }
    }

    // median filter + speckle filter
    post_filter();

}


void SGM::process(Mat &img_l, Mat &img_r, Mat &sky_mask, Mat &sky_mask_beta)
{
    this->sky_mask = sky_mask;
    this->sky_mask_beta = sky_mask_beta;
    process(img_l, img_r);
}


SGM::~SGM()
{
	delete[] L1;
	delete[] L2;
	delete[] L3;
	delete[] L4;
	delete[] min_L1;
	delete[] min_L2;
	delete[] min_L3;
	delete[] min_L4;
	if (USE_8_PATH)
	{
		delete[] L5;
		delete[] L6;
		delete[] L7;
		delete[] L8;
		delete[] min_L5;
		delete[] min_L6;
		delete[] min_L7;
		delete[] min_L8;
	}
}
