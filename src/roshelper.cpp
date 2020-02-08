#include "../inc/roshelper.h"


// if "no tf data" in rviz, run "rosrun tf static_transform_publisher 0 0 0 0 0 0 1 map my_frame 10 &"
void publish_pointcloud(ros::Publisher pc_pub,
                        const std::vector<cv::Point3d> &pointcloud,
                        const std::vector<uchar> &itensity)
{
    sensor_msgs::PointCloud pc;
    pc.header.stamp = ros::Time::now();
    pc.header.frame_id = "my_frame";
    pc.points.resize(pointcloud.size());

    pc.channels.resize(1);
    pc.channels[0].name = "grey";
    pc.channels[0].values.resize(pointcloud.size());

    for (int i = 0; i < pointcloud.size(); ++i)
    {
        pc.points[i].x = pointcloud[i].x;
        pc.points[i].y = pointcloud[i].y;
        pc.points[i].z = pointcloud[i].z;
        pc.channels[0].values[i] = itensity[i];
    }
    pc_pub.publish(pc);
}

void publish_rgb(image_transport::Publisher img_pub, const cv::Mat &img)
{
    sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "8UC3", img).toImageMsg();
    img_pub.publish(msg);
}

void publish_float32(image_transport::Publisher img_pub, const cv::Mat &img)
{
    sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "32FC1", img).toImageMsg();
    img_pub.publish(msg);
}
