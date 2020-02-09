#ifndef LKSUBPIXELIMPL_H_
#define LKSUBPIXELIMPL_H_

#include <iostream>
#include <tr1/memory>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include "LKSubPixel.h"


class LKSubPixelImpl : public LKSubPixel
{
  public:
    explicit LKSubPixelImpl(int h, int w, int s, int d);
    virtual ~LKSubPixelImpl();

    LKSubPixelImpl(const LKSubPixelImpl&) =delete;
    LKSubPixelImpl& operator=(const LKSubPixelImpl&) =delete;

    virtual void LKRefine(const cv::Mat &img_l,
                          const cv::Mat &img_r,
                          cv::Mat &disp_float);

  private:
    bool DispValid(float disparity) const {
      return disparity > 0 && disparity < max_disp;
    }

    bool GradValid(float gradient) const {
      return gradient > 2;  // TODO
    }

    uchar GetInterpolatedValue(const cv::Mat &img,
                               float u, float v) const;

    void LKRefineCore(cv::Mat &disp_float) const;

    static const int enable_adaptive_win = 1;
    static const int win_size = 7;
    static const int iter_num = 10;

    int img_h, img_w;
    int scale;
    int max_disp, invalid_disp;
    
    cv::Mat img_l, img_r;

    mutable int success_num;
    mutable int process_num;
};

#endif