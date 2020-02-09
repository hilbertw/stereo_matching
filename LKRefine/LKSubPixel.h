#ifndef LKSUBPIXEL_H_
#define LKSUBPIXEL_H_

#include <iostream>
#include <tr1/memory>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>

class LKSubPixel
{
  public:
    explicit LKSubPixel(int h, int w, int s, int d);
    virtual ~LKSubPixel();

    LKSubPixel(const LKSubPixel&) =delete;
    LKSubPixel& operator=(const LKSubPixel&) =delete;

    virtual void LKRefine(const cv::Mat &img_l,
                          const cv::Mat &img_r,
                          cv::Mat &disp_float) =0;
    
    static std::shared_ptr<LKSubPixel> create(int h, int w, int s, int d);
};

typedef std::shared_ptr<LKSubPixel> LKSubPixelPtr;

#endif