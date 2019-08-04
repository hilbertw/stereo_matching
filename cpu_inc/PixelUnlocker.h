#pragma once

#include "global.h"
#include "Solver.h"


class PixelUnlocker
{
  public:
    PixelUnlocker();
	~PixelUnlocker();

	Mat unlock(Mat &img_l, Mat &img_r, Mat &disp);

};

