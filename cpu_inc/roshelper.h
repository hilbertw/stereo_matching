#pragma once

#include "global.h"


void publish_pointcloud(ros::Publisher pd_pub,
                        const std::vector<cv::Point3d> &stereo_pts,
                        const std::vector<uchar> &stereo_pixel);

void publish_rgb(image_transport::Publisher disp_pub, const cv::Mat &disp);

void publish_disp(image_transport::Publisher disp_pub, const cv::Mat &disp);
