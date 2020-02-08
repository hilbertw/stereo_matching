#include "../inc/BM.h"


BM::BM(int h, int w, int s, int d)
   : Solver(h, w, s, d)
{}


void BM::process(Mat &img_l, Mat &img_r)
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
            const uchar *ptr_l = img_l.ptr<uchar>(i);
            const uchar *ptr_r = img_r.ptr<uchar>(i);
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

	double be = get_cur_ms();
	build_cost_table();
	build_dsi_from_table();
	printf("build cost takes %lf ms\n", get_cur_ms() - be);

	be = get_cur_ms();
    cost_horizontal_filter(COST_WIN_W/scale);
    cost_vertical_filter(COST_WIN_H/scale);
	printf("cost filter takes %lf ms\n", get_cur_ms() - be);

	be = get_cur_ms();

	uchar *ptr = NULL;
	float min_cost = FLT_MAX, sec_min_cost = FLT_MAX;
    uchar min_d = invalid_disp, sec_min_d = invalid_disp;
    for (int i = 0; i < img_h; i++)
	{
        ptr = disp.ptr<uchar>(i);
        for (int j = 0; j < img_w; j++)
		{
			min_cost = FLT_MAX;
            min_d = invalid_disp;
            for (int d = 0; d < max_disp; d++)
			{
                int index = i * img_w * max_disp + j * max_disp + d;
				if (cost[index] < min_cost)
				{
					min_cost = cost[index];
					min_d = d;
				}
			}
			// unique check
			sec_min_cost = FLT_MAX;
            for (int d = 0; d < max_disp; d++)
			{
                int index = i * img_w * max_disp + j * max_disp + d;
				if (cost[index] < sec_min_cost && cost[index] != min_cost)
				{
					sec_min_cost = cost[index];
					sec_min_d = d;
				}
			}
			if (min_cost / sec_min_cost > UNIQUE_RATIO && abs(min_d - sec_min_d) > 2)
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

	printf("matching takes %lf ms\n", get_cur_ms() - be);

	post_filter();
}


void BM::process(Mat &img_l, Mat &img_r, Mat &sky_mask, Mat &sky_mask_beta)
{
    this->sky_mask = sky_mask;
    this->sky_mask_beta = sky_mask_beta;
    process(img_l, img_r);
}


BM::~BM()
{}
