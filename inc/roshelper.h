#pragma once

#include "global.h"


void publish_pointcloud(ros::Publisher pc_pub,
                        const std::vector<cv::Point3d> &pointcloud,
                        const std::vector<uchar> &itensity);

void publish_rgb(image_transport::Publisher img_pub, const cv::Mat &img);

void publish_float32(image_transport::Publisher img_pub, const cv::Mat &img);
