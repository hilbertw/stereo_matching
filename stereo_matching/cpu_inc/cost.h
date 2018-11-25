#pragma once

#include "global.h"
#include "Solver.h"


float SAD(Mat &img_l, Mat &img_r, Point l_pt, int disp, int win_h, int win_w, float* weight);
float SSD(Mat &img_l, Mat &img_r, Point l_pt, int disp, int win_h, int win_w, float* weight);

int CT(Mat &img_l, Mat &img_r, Point l_pt, int disp, int win_h, int win_w, float* weight);
int hamming_cost(uint64_t ct_l, uint64_t ct_r);

void CT_pts(Mat &img_l, Mat &img_r, int u, int v, int win_h, int win_w, float* weight,
	                  uint64_t *cost_table_l, uint64_t *cost_table_r);