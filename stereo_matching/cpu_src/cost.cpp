#include "../cpu_inc/Cost.h"


float SAD(Mat &img_l, Mat &img_r, Point l_pt, int disp, int win_h, int win_w, float* weight)
{
	uchar *ptr_l = NULL, *ptr_r = NULL;
	int y = 0, x_l = 0, x_r = 0;
	float cost = 0;
	for (int i = -win_h / 2; i <= win_h/2; i++)
	{
		y = MAX(l_pt.y + i, 0);		// check border
		y = MIN(y, img_l.rows - 1);
		ptr_l = img_l.ptr<uchar>(y);
		ptr_r = img_r.ptr<uchar>(y); 
		for (int j = -win_w / 2; j <= win_w / 2; j++)
		{
			x_l = MAX(l_pt.x + j, 0);
			x_l = MIN(x_l, img_l.cols - 1);
			x_r = MAX(x_l - disp / SCALE, 0);
			if (WEIGHTED_COST)
			{
				cost += abs(ptr_l[x_l] - ptr_r[x_r]) * weight[(i + win_h / 2) * win_w + (j + win_w / 2)];
			}
			else
			{
				cost += abs(ptr_l[x_l] - ptr_r[x_r]);
			}
		}
	}
	ptr_l = ptr_r = NULL;
	return cost / win_h / win_w;
}


float SSD(Mat &img_l, Mat &img_r, Point l_pt, int disp, int win_h, int win_w, float* weight)
{
	uchar *ptr_l = NULL, *ptr_r = NULL;
	int y = 0, x_l = 0, x_r = 0;
	float cost = 0;
	for (int i = -win_h / 2; i <= win_h / 2; i++)
	{
		y = MAX(l_pt.y + i, 0);		// check border
		y = MIN(y, img_l.rows - 1);
		ptr_l = img_l.ptr<uchar>(y);
		ptr_r = img_r.ptr<uchar>(y);
		for (int j = -win_w / 2; j <= win_w / 2; j++)
		{
			x_l = MAX(l_pt.x + j, 0);
			x_l = MIN(x_l, img_l.cols - 1);
			x_r = MAX(x_l - disp / SCALE, 0);
			if (WEIGHTED_COST)
			{
				cost += (ptr_l[x_l] - ptr_r[x_r]) * (ptr_l[x_l] - ptr_r[x_r]) * weight[(i + win_h / 2) * win_w + (j + win_w / 2)];
			}
			else
			{
				cost += (ptr_l[x_l] - ptr_r[x_r]) * (ptr_l[x_l] - ptr_r[x_r]);
			}
		}
	}
	ptr_l = ptr_r = NULL;
	return cost / win_h / win_w;		// be careful of overflow
}


int CT(Mat &img_l, Mat &img_r, Point l_pt, int disp, int win_h, int win_w, float* weight)
{
	uchar *ptr_l = NULL, *ptr_r = NULL;
	int y = 0, x_l = 0, x_r = 0;
	uint64_t ct_l = 0, ct_r = 0;
	int cost = 0;

	uchar ctr_pixel_l = img_l.at<uchar>(l_pt.y, l_pt.x);
	uchar ctr_pixel_r = img_r.at<uchar>(l_pt.y, MAX(l_pt.x - disp, 0));

	for (int i = -win_h / 2; i <= win_h / 2; i++)
	{
		y = MAX(l_pt.y + i, 0);		// check border
		y = MIN(y, img_l.rows - 1);
		ptr_l = img_l.ptr<uchar>(y);
		ptr_r = img_r.ptr<uchar>(y);
		for (int j = -win_w / 2; j <= win_w / 2; j++)
		{
			if (i == 0 && j == 0)
				continue;
			if (WEIGHTED_COST && weight[(i + win_h / 2) * win_w + (j + win_w / 2)] < 0.5)
				continue;
			x_l = MAX(l_pt.x + j, 0);
			x_l = MIN(x_l, img_l.cols - 1);
			x_r = MAX(x_l - disp / SCALE, 0);
			ct_l = (ct_l | (ptr_l[x_l] > ctr_pixel_l)) << 1;
			ct_r = (ct_r | (ptr_r[x_r] > ctr_pixel_r)) << 1;
		}
	}
	cost = hamming_cost(ct_l, ct_r);
	ptr_l = ptr_r = NULL;
	return cost;
}


void CT_pts(Mat &img_l, Mat &img_r, int u, int v, int win_h, int win_w, float* weight, uint64_t *cost_table_l, uint64_t *cost_table_r)
{
	uint64_t value_l = 0, value_r = 0;

	uchar ctr_pixel_l = img_l.at<uchar>(v, u);
	uchar ctr_pixel_r = img_r.at<uchar>(v, u);

	for (int i = -win_h / 2; i <= win_h / 2; i++)
	{
		int y = MAX(v + i, 0);		// check border
		y = MIN(y, img_l.rows - 1);
		for (int j = -win_w / 2; j <= win_w / 2; j++)
		{
			if (i == 0 && j == 0)
				continue;
			if (WEIGHTED_COST && weight[(i + win_h / 2) * win_w + (j + win_w / 2)] < 0.5)
				continue;
			int x = MAX(u + j, 0);
			x = MIN(x, img_l.cols - 1);
			value_l = (value_l | (img_l.at<uchar>(y, x) > ctr_pixel_l)) << 1;
			value_r = (value_r | (img_r.at<uchar>(y, x) > ctr_pixel_r)) << 1;
		}
	}
	cost_table_l[v*img_l.cols + u] = value_l;
	cost_table_r[v*img_l.cols + u] = value_r;
}


int hamming_cost(uint64_t ct_l, uint64_t ct_r)
{
	uint64_t not_the_same = ct_l ^ ct_r;
	// find the number of '1', log(N)
	int cnt = 0;
	while (not_the_same)
	{
		//std::cout << not_the_same << std::endl;
		cnt += (not_the_same & 1);
		not_the_same >>= 1;
	}
	return cnt;
}