#include "../cpu_inc/roshelper.h"


// if "no tf data" in rviz, run "rosrun tf static_transform_publisher 0 0 0 0 0 0 1 map my_frame 10 &"
void publish_pointcloud(ros::Publisher pd_pub,
                        const std::vector<cv::Point3d> &stereo_pts,
                        const std::vector<uchar> &stereo_pixel)
{
    sensor_msgs::PointCloud pd;
    pd.header.stamp = ros::Time::now();
    pd.header.frame_id = "my_frame";
    pd.points.resize(stereo_pts.size());

    pd.channels.resize(1);
    pd.channels[0].name = "grey";
    pd.channels[0].values.resize(stereo_pts.size());

    for (int i = 0; i < stereo_pts.size(); ++i)
    {
        pd.points[i].x = stereo_pts[i].x;
        pd.points[i].y = stereo_pts[i].y;
        pd.points[i].z = stereo_pts[i].z;
        pd.channels[0].values[i] = stereo_pixel[i];
    }
    pd_pub.publish(pd);
}

void publish_rgb(image_transport::Publisher disp_pub, const cv::Mat &disp)
{
    sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "8UC3", disp).toImageMsg();
    disp_pub.publish(msg);
}

void publish_disp(image_transport::Publisher disp_pub, const cv::Mat &disp)
{
    sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "32FC1", disp).toImageMsg();
    disp_pub.publish(msg);
}
