#pragma once

#include "global.h"
#include "utils.h"
#include "cost.h"


const int WIN_H = 7 / SCALE;
const int WIN_W = 9 / SCALE;
const int COST_WIN_H = 3 / SCALE;
const int COST_WIN_W = 5 / SCALE;
const float UNIQUE_RATIO = 0.7;
const bool WEIGHTED_COST = 0;
const int MEDIAN_FILTER_H = 5;
const int MEDIAN_FILTER_W = 5;
const int SPECKLE_SIZE = 1000 / SCALE;
const int SPECKLE_DIS = 2;


class Solver
{
public:
	Solver();
	virtual ~Solver();

	void show_disp(Mat &debug_view); 
	virtual void process(Mat &img_l, Mat &img_r);
	void build_dsi();
	void build_cost_table();
	void build_dsi_from_table();
	float find_dsi_mean_max();
	float find_table_mean_max();
	void cost_horizontal_filter(int win_size);
	void cost_vertical_filter(int win_size);
	void fetch_cost(float *p);
	void fetch_disparity(uchar *d);
	void fetch_disparity(float *d);
	void post_filter();
	void colormap();
	Mat get_disp() const
	{
		return filtered_disp;
	}

protected:
	Mat img_l, img_r;
	Mat disp,  filtered_disp, colored_disp;
	uint64_t *cost_table_l, *cost_table_r;
	float *cost;
	float *weight;
	int disp_cnt;
};

