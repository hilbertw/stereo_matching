#pragma once

#include "global.h"
#include "utils.h"
#include "cost.h"

#include "../LKRefine/LKSubPixel.h"


const int WIN_H = 7;
const int WIN_W = 9;
const int COST_WIN_H = 3;
const int COST_WIN_W = 5;
const float UNIQUE_RATIO = 0.7;
const bool WEIGHTED_COST = 0;
const float LR_CHECK_DIS = 1;
const int MEDIAN_FILTER_H = 5;
const int MEDIAN_FILTER_W = 5;
const int SPECKLE_SIZE = 1000;
const int SPECKLE_DIS = 2;


class Solver
{
public:
    explicit Solver(int h, int w, int s, int d);
	virtual ~Solver();

	Solver(const Solver&) =delete;
	Solver& operator=(Solver&) =delete;
	
	virtual void process(Mat &img_l, Mat &img_r) =0;
    virtual void process(Mat &img_l, Mat &img_r, Mat &sky_mask, Mat &sky_mask_beta) =0;

	virtual void show_disp(Mat &debug_view); 
	virtual const Mat& get_disp() const { return filtered_disp;}

protected:
    int img_h, img_w;
    int scale;
    int max_disp, invalid_disp;

	Mat img_l, img_r;
    Mat disp, disp_beta, filtered_disp, filtered_disp_beta;
    Mat colored_disp;

	uint64_t *cost_table_l, *cost_table_r;
	float *cost;
	float *weight;

    Mat sky_mask, sky_mask_beta;

	LKSubPixelPtr lkptr;

protected:
	void build_dsi();
	void build_cost_table();
	void build_dsi_from_table();
    void build_dsi_from_table_beta();
	float find_dsi_mean_max();
	float find_table_mean_max();
	void cost_horizontal_filter(int win_size);
	void cost_vertical_filter(int win_size);
	void fetch_cost(float *p);
	void fetch_disparity(uchar *d);
	void fetch_disparity(float *d);
    void compute_subpixel(const Mat &disp, Mat &filtered_disp);
	void post_filter();
	void colormap();
};

typedef std::shared_ptr<Solver> SolverPtr;

