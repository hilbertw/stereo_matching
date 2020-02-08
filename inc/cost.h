#pragma once

#include "global.h"
#include "Solver.h"


float SAD(const Mat &img_l, const Mat &img_r, const Point &l_pt, int disp, int win_h, int win_w, float* weight, int scale=1);
float SSD(const Mat &img_l, const Mat &img_r, const Point &l_pt, int disp, int win_h, int win_w, float* weight, int scale=1);

int CT(const Mat &img_l, const Mat &img_r, const Point &l_pt, int disp, int win_h, int win_w, float* weight, int scale=1);
int hamming_cost(uint64_t ct_l, uint64_t ct_r);

void CT_pts(const Mat &img_l, const Mat &img_r, int u, int v, int win_h, int win_w, float* weight,
            uint64_t *cost_table_l, uint64_t *cost_table_r);
