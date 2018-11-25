#pragma once

#include "Solver.h"


class GM : public Solver
{
public:
	GM();
	~GM();

	virtual void process(Mat &img_l, Mat &img_r);

};

