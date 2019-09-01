#include "../cpu_inc/SGM.h"


SGM::SGM() : Solver()
{
	L1 = new float[IMG_H * IMG_W * MAX_DISP];
	L2 = new float[IMG_H * IMG_W * MAX_DISP];
	L3 = new float[IMG_H * IMG_W * MAX_DISP];
	L4 = new float[IMG_H * IMG_W * MAX_DISP];
	min_L1 = new float[IMG_H * IMG_W];
	min_L2 = new float[IMG_H * IMG_W];
	min_L3 = new float[IMG_H * IMG_W];
	min_L4 = new float[IMG_H * IMG_W];
	if (USE_8_PATH)
	{
		L5 = new float[IMG_H * IMG_W * MAX_DISP];
		L6 = new float[IMG_H * IMG_W * MAX_DISP];
		L7 = new float[IMG_H * IMG_W * MAX_DISP];
		L8 = new float[IMG_H * IMG_W * MAX_DISP];
		min_L5 = new float[IMG_H * IMG_W];
		min_L6 = new float[IMG_H * IMG_W];
		min_L7 = new float[IMG_H * IMG_W];
		min_L8 = new float[IMG_H * IMG_W];
	}

	P1 = 10;
	P2 = 100;
}


void SGM::process(Mat &img_l, Mat &img_r)
{
	this->img_l = img_l;
	this->img_r = img_r;

	double be = get_cur_ms();
	build_cost_table();
	build_dsi_from_table();
	printf("build cost takes %lf ms\n", get_cur_ms() - be);

	be = get_cur_ms();
    cost_horizontal_filter(COST_WIN_W);
    cost_vertical_filter(COST_WIN_H);
	printf("cost filter takes %lf ms\n", get_cur_ms() - be);

	be = get_cur_ms();

    //find_table_mean_max();
    //find_dsi_mean_max();


    // build L1: left -> right
#pragma omp parallel for
    for (int i = 0; i < IMG_H; ++i)
	{
        for (int j = 0; j < IMG_W; ++j)
        {
            int index_L1_prev = i * IMG_W * MAX_DISP + (j - 1) * MAX_DISP;

			// DP
			float minL1 = FLT_MAX;
            int bias = i * IMG_W * MAX_DISP + j * MAX_DISP;

            for (int d = 0; d < MAX_DISP; ++d)
			{
                int index = bias + d;
                uchar d_sub_1 = MAX(d - 1, 0);
                uchar d_plus_1 = MIN(d + 1, MAX_DISP - 1);

                if (j == 0)  //init
                {
                    L1[index] = cost[index];
                }
                else
                {
                    L1[index] = MIN(L1[index_L1_prev + d], L1[index_L1_prev + d_sub_1] + P1);
                    L1[index] = MIN(L1[index], L1[index_L1_prev + d_plus_1] + P1);
                    L1[index] = MIN(L1[index], min_L1[i * IMG_W + j - 1] + P2);
                    L1[index] += (cost[index] - min_L1[i * IMG_W + j - 1]);
                }
                if (L1[index] < minL1)
                {
                    minL1 = L1[index];
                }
			}
			
			// update minL1
			min_L1[i * IMG_W + j] = minL1;
        }
	}

	// build L2: right -> left
#pragma omp parallel for
    for (int i = 0; i < IMG_H; ++i)
	{
        for (int j = IMG_W - 1; j >=0; --j)
		{
            int index_L2_prev = i * IMG_W * MAX_DISP + (j + 1) * MAX_DISP;

			// DP
            float minL2 = FLT_MAX;
            int bias = i * IMG_W * MAX_DISP + j * MAX_DISP;

            for (int d = 0; d < MAX_DISP; ++d)
			{
                int index = bias + d;
                uchar d_sub_1 = MAX(d - 1, 0);
                uchar d_plus_1 = MIN(d + 1, MAX_DISP - 1);

                if (j == IMG_W - 1)  //init
				{
					L2[index] = cost[index];
				}
				else
				{
					L2[index] = MIN(L2[index_L2_prev + d], L2[index_L2_prev + d_sub_1] + P1);
					L2[index] = MIN(L2[index], L2[index_L2_prev + d_plus_1] + P1);
					L2[index] = MIN(L2[index], min_L2[i * IMG_W + j + 1] + P2);
					L2[index] += (cost[index] - min_L2[i * IMG_W + j +1]);
				}
				if (L2[index] < minL2)
				{
					minL2 = L2[index];
				}
			}

			// update minL2
			min_L2[i * IMG_W + j] = minL2;
		}
	}

	// build L3: top -> down
#pragma omp parallel for
    for (int j = 0; j < IMG_W; ++j)
    {
        for (int i = 0; i < IMG_H; ++i)
		{
            int index_L3_prev = (i - 1) * IMG_W * MAX_DISP + j * MAX_DISP;

			// DP
			float minL3 = FLT_MAX;
            int bias = i * IMG_W * MAX_DISP + j * MAX_DISP;

            for (int d = 0; d < MAX_DISP; ++d)
			{
                int index = bias + d;
                uchar d_sub_1 = MAX(d - 1, 0);
                uchar d_plus_1 = MIN(d + 1, MAX_DISP - 1);

                if (i == 0)  //init
				{
					L3[index] = cost[index];
				}
				else
				{
					L3[index] = MIN(L3[index_L3_prev + d], L3[index_L3_prev + d_sub_1] + P1);
					L3[index] = MIN(L3[index], L3[index_L3_prev + d_plus_1] + P1);
					L3[index] = MIN(L3[index], min_L3[(i - 1) * IMG_W + j ] + P2);
					L3[index] += (cost[index] - min_L3[(i - 1) * IMG_W + j]);
				}
				if (L3[index] < minL3)
				{
					minL3 = L3[index];
				}
			}

			// update minL3
			min_L3[i * IMG_W + j] = minL3;
		}
	}

	// build L4: down -> top
#pragma omp parallel for
    for (int j = 0; j < IMG_W; ++j)
	{
        for (int i = IMG_H - 1; i >=0; --i)
		{
            int index_L4_prev = (i + 1) * IMG_W * MAX_DISP + j * MAX_DISP;

			// DP
            float minL4 = FLT_MAX;
            int bias = i * IMG_W * MAX_DISP + j * MAX_DISP;

            for (int d = 0; d < MAX_DISP; ++d)
			{
                int index = bias + d;
                uchar d_sub_1 = MAX(d - 1, 0);
                uchar d_plus_1 = MIN(d + 1, MAX_DISP - 1);

                if (i == IMG_H - 1)  //init
				{
					L4[index] = cost[index];
				}
				else
				{
					L4[index] = MIN(L4[index_L4_prev + d], L4[index_L4_prev + d_sub_1] + P1);
					L4[index] = MIN(L4[index], L4[index_L4_prev + d_plus_1] + P1);
					L4[index] = MIN(L4[index], min_L4[(i + 1) * IMG_W + j] + P2);
					L4[index] += (cost[index] - min_L4[(i + 1) * IMG_W + j]);
				}
				if (L4[index] < minL4)
				{
					minL4 = L4[index];
				}
			}

			// update minL4
			min_L4[i * IMG_W + j] = minL4;
		}
	}

    if (USE_8_PATH)
    {

        // build L5: lefttop -> rightdown
        // build L6: righttop -> leftdown

        for (int i = 0; i < IMG_H; ++i)
        {
#pragma omp parallel for
            for (int j = 0; j < IMG_W; ++j)
            {
                int index_L5_prev = (i - 1) * IMG_W * MAX_DISP + (j - 1) * MAX_DISP;
                int index_L6_prev = (i - 1) * IMG_W * MAX_DISP + (j + 1) * MAX_DISP;

                // DP
                float minL5 = FLT_MAX;
                float minL6 = FLT_MAX;
                int bias = i * IMG_W * MAX_DISP + j * MAX_DISP;

                for (int d = 0; d < MAX_DISP; ++d)
                {
                    int index = bias + d;
                    uchar d_sub_1 = MAX(d - 1, 0);
                    uchar d_plus_1 = MIN(d + 1, MAX_DISP - 1);

                    if (i == 0 || j == 0)  //init
                    {
                        L5[index] = cost[index];
                    }
                    else
                    {
                        L5[index] = MIN(L5[index_L5_prev + d], L5[index_L5_prev + d_sub_1] + P1);
                        L5[index] = MIN(L5[index], L5[index_L5_prev + d_plus_1] + P1);
                        L5[index] = MIN(L5[index], min_L5[(i - 1) * IMG_W + j - 1] + P2);
                        L5[index] += (cost[index] - min_L5[(i - 1) * IMG_W + j - 1]);
                    }
                    if (L5[index] < minL5)
                    {
                        minL5 = L5[index];
                    }

                    if (i == 0 || j == IMG_W - 1)  //init
                    {
                        L6[index] = cost[index];
                    }
                    else
                    {
                        L6[index] = MIN(L6[index_L6_prev + d], L6[index_L6_prev + d_sub_1] + P1);
                        L6[index] = MIN(L6[index], L6[index_L6_prev + d_plus_1] + P1);
                        L6[index] = MIN(L6[index], min_L6[(i - 1) * IMG_W + j + 1] + P2);
                        L6[index] += (cost[index] - min_L6[(i - 1) * IMG_W + j + 1]);
                    }
                    if (L6[index] < minL6)
                    {
                        minL6 = L6[index];
                    }
                }

                // update minL5
                min_L5[i * IMG_W + j] = minL5;

                // update minL6
                min_L6[i * IMG_W + j] = minL6;
            }
        }


        // build L7: leftdown -> righttop
        // build L8: rightdown -> lefttop

        for (int i = IMG_H - 1; i >=0; --i)
        {
#pragma omp parallel for
            for (int j = 0; j < IMG_W; ++j)
            {
                int index_L7_prev = (i + 1) * IMG_W * MAX_DISP + (j - 1) * MAX_DISP;
                int index_L8_prev = (i + 1) * IMG_W * MAX_DISP + (j + 1) * MAX_DISP;

                // DP
                float minL7 = FLT_MAX;
                float minL8 = FLT_MAX;
                int bias = i * IMG_W * MAX_DISP + j * MAX_DISP;

                for (int d = 0; d < MAX_DISP; ++d)
                {
                    int index = bias + d;
                    uchar d_sub_1 = MAX(d - 1, 0);
                    uchar d_plus_1 = MIN(d + 1, MAX_DISP - 1);

                    if (i == IMG_H - 1 || j == 0)  //init
                    {
                        L7[index] = cost[index];
                    }
                    else
                    {
                        L7[index] = MIN(L7[index_L7_prev + d], L7[index_L7_prev + d_sub_1] + P1);
                        L7[index] = MIN(L7[index], L7[index_L7_prev + d_plus_1] + P1);
                        L7[index] = MIN(L7[index], min_L7[(i + 1) * IMG_W + j - 1] + P2);
                        L7[index] += (cost[index] - min_L7[(i + 1) * IMG_W + j - 1]);
                    }
                    if (L7[index] < minL7)
                    {
                        minL7 = L7[index];
                    }

                    if (i == IMG_H - 1 || j == IMG_W - 1)  //init
                    {
                        L8[index] = cost[index];
                    }
                    else
                    {
                        L8[index] = MIN(L8[index_L8_prev + d], L8[index_L8_prev + d_sub_1] + P1);
                        L8[index] = MIN(L8[index], L8[index_L8_prev + d_plus_1] + P1);
                        L8[index] = MIN(L8[index], min_L8[(i + 1) * IMG_W + j + 1] + P2);
                        L8[index] += (cost[index] - min_L8[(i + 1) * IMG_W + j + 1]);
                    }
                    if (L8[index] < minL8)
                    {
                        minL8 = L8[index];
                    }
                }

                // update minL7
                min_L7[i * IMG_W + j] = minL7;

                // update minL8
                min_L8[i * IMG_W + j] = minL8;
            }
        }
    }
	
	// cost aggregation
	uchar *ptr = NULL;
	float min_cost = FLT_MAX, sec_min_cost = FLT_MAX;
	uchar min_d = INVALID_DISP, sec_min_d = INVALID_DISP;
    for (int i = 0; i < IMG_H; ++i)
	{
		ptr = disp.ptr<uchar>(i);
        for (int j = 0; j < IMG_W; ++j)
		{
			min_cost = FLT_MAX;
            int index_bias = i * IMG_W * MAX_DISP + j * MAX_DISP;
            for (int d = 0; d < MAX_DISP; ++d)
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
            for (int d = 0; d < MAX_DISP; ++d)
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
				ptr[j] = INVALID_DISP;
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
//        for (int i = 0; i < IMG_H/2+1; ++i)
//        {
//            uchar *ptr = disp.ptr<uchar>(i);
//            const uchar *ptr_sky = sky_mask.ptr<uchar>(i);

//            for (int j = 0; j < IMG_W; ++j)
//            {
//                if (ptr_sky[j] == 255)
//                    ptr[j] = 0;
//            }
//        }
//    }

	post_filter();
}


void SGM::process(Mat &img_l, Mat &img_r, Mat &sky_mask)
{
    this->sky_mask = sky_mask;
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
