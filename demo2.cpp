#include "cpu_inc/global.h"
#include "cpu_inc/BM.h"
#include "cpu_inc/SGM.h"
#include "cpu_inc/utils.h"
#include "gpu_inc/SGM.cuh"
#include "gpu_inc/cost.cuh"

#include "sky_detector/imageSkyDetector.h"


std::string data_addr = "/home/hunterlew/data_stereo_flow_multiview/";
std::string res_addr = "/home/hunterlew/catkin_ws/src/stereo_matching/res2/";

struct CamIntrinsics
{
	float fx;
	float fy;
	float cx;
	float cy;
};

CamIntrinsics read_calib(std::string file_addr)
{
	std::ifstream in;
	in.open(file_addr);
	if (!in.is_open()){
		printf("reading calib file failed\n");
		assert(false);
	}
	std::string str, str_tmp;
	std::stringstream ss;
	std::getline(in, str);  // only read left cam P0
	ss.clear();
	ss.str(str);

	CamIntrinsics cam_para;
	for (int i = 0; i < 13; i++)
	{
		if (i == 1)
			ss >> cam_para.fx;
		else if (i == 3)
			ss >> cam_para.cx;
		else if (i == 6)
			ss >> cam_para.fy;
		else if (i == 7)
			ss >> cam_para.cy;
		else
			ss >> str_tmp;
	}

	return cam_para;
}

// if "no tf data" in rviz, run "rosrun tf static_transform_publisher 0 0 0 0 0 0 1 map my_frame 10"
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

int main(int argc, char **argv)
{
	ros::init(argc, argv, "stereo_node");
	ros::NodeHandle nh("~");
	ros::Publisher pd_pub = nh.advertise<sensor_msgs::PointCloud> ("/stereo_cam0/pointcloud", 1);
	image_transport::ImageTransport it(nh);
	image_transport::Publisher disp_pub = it.advertise("/stereo_cam0/disp", 1);
	image_transport::Publisher debug_pub = it.advertise("/stereo_cam0/debug_view", 1);

	Mat disp;
	Mat debug_view;

    auto sv = std::make_shared<SGM>();
//    auto g_sv = std::make_shared<GPU_SGM>();

    auto sky_det = std::make_shared<sky_detector::SkyAreaDetector>();

//     for (int i = 0; i <= 194; i++)
//     {
//        for (int j = 0; j <= 20; ++j)
//        {
    for (int i = 0; i <= 0; i++)
    {
        for (int j = 0; j <= 0; ++j)
        {
            std::string img_l_addr = data_addr+"testing/image_0/"+num2str(i)+"_"+num2strbeta(j)+".png";
            std::string img_r_addr = data_addr+"testing/image_1/"+num2str(i)+"_"+num2strbeta(j)+".png";
            std::cout << "processing " << img_l_addr << std::endl;

//            std::string img_index = "000027_07";
//            std::string img_l_addr = data_addr+"testing/image_0/"+img_index+".png";
//            std::string img_r_addr = data_addr+"testing/image_1/"+img_index+".png";

			Mat img_l = imread(img_l_addr, IMREAD_GRAYSCALE);
			Mat img_r = imread(img_r_addr, IMREAD_GRAYSCALE);

			printf("left size: %d, %d\n", img_l.rows, img_l.cols);
			printf("right size: %d, %d\n", img_r.rows, img_r.cols);

			resize(img_l, img_l, Size(IMG_W, IMG_H));
			resize(img_r, img_r, Size(IMG_W, IMG_H));
			printf("resized left size: %d, %d\n", img_l.rows, img_l.cols);
			printf("resized right size: %d, %d\n", img_r.rows, img_r.cols);

            Mat sky_mask;
            sky_det->detect(img_l, res_addr+num2str(i)+"_"+num2strbeta(j)+"_sky.png", sky_mask);
//            sky_det->detect(img_l, res_addr+"test_sky.png", sky_mask);

			printf("waiting ...\n");

			double be = get_cur_ms();
//            g_sv->process(img_l, img_r);
            sv->process(img_l, img_r, sky_mask);
			double en = get_cur_ms();
			printf("done ...\n");
			printf("time cost: %lf ms\n", en - be);

//            disp = g_sv->get_disp();
            disp = sv->get_disp();
			printf("disp size: %d, %d\n", disp.rows, disp.cols);
			publish_disp(disp_pub, disp);

//            g_sv->show_disp(debug_view);
            sv->show_disp(debug_view);
			publish_rgb(debug_pub, debug_view);

            imwrite(res_addr+num2str(i)+"_"+num2strbeta(j)+"_disp.png", debug_view);
//            imwrite(res_addr+"test.png", debug_view);

			waitKey(1);

			// read calibration
			CamIntrinsics cam_para = read_calib(data_addr+"calib/"+num2str(i)+".txt");
			printf("calib param: fx %f, fy %f, cx %f, cy %f\n",
					cam_para.fx, cam_para.fy, cam_para.cx, cam_para.cy);

			// convert to cam coords
			std::vector<cv::Point3d> stereo_pts;
			std::vector<uchar> stereo_pixel;

			float max_range = 100;
			double baseline = 0.5;
			for (int i = 0; i < disp.rows; i++)
			{
				float *ptr = disp.ptr<float>(i);
				for (int j = 0; j < disp.cols; j++)
				{
					// printf("%f, ", ptr[j]);

					if (ptr[j] == INVALID_DISP)  continue;
					
					double Zc = (cam_para.fx+cam_para.fy)/2.0 * baseline / (ptr[j]+1e-6);
					if (Zc > max_range)  continue;

					double Xc = (j - cam_para.cx/SCALE) * Zc / cam_para.fx;
					double Yc = (i - cam_para.cy/SCALE) * Zc / cam_para.fy;

					stereo_pts.push_back({Xc, Yc, Zc});
					stereo_pixel.push_back(img_l.at<uchar>(i,j));
				}
			}

			printf("pointcloud size: %zu, %zu\n", stereo_pts.size(), stereo_pixel.size());
			publish_pointcloud(pd_pub, stereo_pts, stereo_pixel);
			usleep(1000);
		}
	}
	
	ros::spin();
	cv::destroyAllWindows();
	return 0;
}
