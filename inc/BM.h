#pragma once

#include "Solver.h"


class BM : public Solver
{
public:
    explicit BM(int h, int w, int s, int d);
	virtual ~BM();

    BM(const BM&) =delete;
    BM& operator=(const BM&) =delete;

	virtual void process(Mat &img_l, Mat &img_r);
    virtual void process(Mat &img_l, Mat &img_r, Mat &sky_mask, Mat &sky_mask_beta);
};

typedef std::shared_ptr<BM> BMSolverPtr;

