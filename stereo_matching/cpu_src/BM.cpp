#include "../cpu_inc/BM.h"


BM::BM() : Solver()
{}


void BM::process(Mat &img_l, Mat &img_r)
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

	uchar *ptr = NULL;
	float min_cost = FLT_MAX, sec_min_cost = FLT_MAX;
	uchar min_d = INVALID_DISP, sec_min_d = INVALID_DISP;
	for (int i = 0; i < IMG_H; i++)
	{
		ptr = disp.ptr<uchar>(i);
		for (int j = 0; j < IMG_W; j++)
		{
			min_cost = FLT_MAX;
			min_d = INVALID_DISP;
			for (int d = 0; d < MAX_DISP; d++)
			{
				int index = i * IMG_W * MAX_DISP + j * MAX_DISP + d;
				if (cost[index] < min_cost)
				{
					min_cost = cost[index];
					min_d = d;
				}
			}
			// unique check
			sec_min_cost = FLT_MAX;
			for (int d = 0; d < MAX_DISP; d++)
			{
				int index = i * IMG_W * MAX_DISP + j * MAX_DISP + d;
				if (cost[index] < sec_min_cost && cost[index] != min_cost)
				{
					sec_min_cost = cost[index];
					sec_min_d = d;
				}
			}
			if (min_cost / sec_min_cost > UNIQUE_RATIO && abs(min_d - sec_min_d) > 2)
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

	printf("matching takes %lf ms\n", get_cur_ms() - be);

	post_filter();
}


BM::~BM()
{}