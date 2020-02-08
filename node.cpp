#include "inc/global.h"
#include "inc/BM.h"
#include "inc/SGM.h"
#include "inc/utils.h"
#include "inc/roshelper.h"

#include "sky_detector/imageSkyDetector.h"
typedef std::shared_ptr<sky_detector::SkyAreaDetector> SkyDetPtr;

// #include "gpu_sgm/inc/SGM.cuh"

std::string data_addr = "/home/hunterlew/data_stereo_flow_multiview/";
std::string res_addr = "/home/hunterlew/catkin_ws/src/stereo_matching/res2/";


bool g_use_gpu = false;  // TODO: config in CMakelists

int g_img_w = 1240;
int g_img_h = 360;
int g_scale = 1;
int g_max_disp = 128;
int g_invalid_disp = g_max_disp+1;


int main(int argc, char **argv)
{
	ros::init(argc, argv, "stereo_node");
	ros::NodeHandle nh("~");
	ros::Publisher pd_pub = nh.advertise<sensor_msgs::PointCloud> ("/stereo_cam0/pointcloud", 1);
	image_transport::ImageTransport it(nh);
	image_transport::Publisher disp_pub = it.advertise("/stereo_cam0/disp", 1);
	image_transport::Publisher debug_pub = it.advertise("/stereo_cam0/debug_view", 1);

    nh.param("/sgm_node/img_w", g_img_w, g_img_w);
    nh.param("/sgm_node/img_h", g_img_h, g_img_h);
    nh.param("/sgm_node/scale", g_scale, g_scale);
    nh.param("/sgm_node/max_disp", g_max_disp, g_max_disp);
    g_invalid_disp = g_max_disp+1;

    printf("read config:\n");
    printf("/sgm_node/img_w: %d\n", g_img_w);
    printf("/sgm_node/img_h: %d\n", g_img_h);
    printf("/sgm_node/scale: %d\n", g_scale);
    printf("/sgm_node/max_disp: %d\n", g_max_disp);

	Mat disp;
	Mat debug_view;

	SolverPtr sv = std::make_shared<SGM>(g_img_h, g_img_w, g_scale, g_max_disp);
	// GSGMPtr gsv = std::make_shared<GPU_SGM>(g_img_h, g_img_w, g_scale, g_max_disp);
	SkyDetPtr sky_det = std::make_shared<sky_detector::SkyAreaDetector>();

//     for (int i = 0; i <= 194; i++)
//     {
//        for (int j = 0; j <= 20; ++j)
//        {
    for (int i = 17; i <= 17; i++)
    {
        for (int j = 14; j <= 14; ++j)
        {
            std::string img_l_addr = data_addr+"testing/image_0/"+num2str(i)+"_"+num2strbeta(j)+".png";
            std::string img_r_addr = data_addr+"testing/image_1/"+num2str(i)+"_"+num2strbeta(j)+".png";
			printf("processing %s\n", img_l_addr.c_str());

//            std::string img_index = "000000_09";
//            std::string img_l_addr = data_addr+"testing/image_0/"+img_index+".png";
//            std::string img_r_addr = data_addr+"testing/image_1/"+img_index+".png";

			Mat img_l = imread(img_l_addr, IMREAD_GRAYSCALE);
			Mat img_r = imread(img_r_addr, IMREAD_GRAYSCALE);

			printf("left size: %d, %d\n", img_l.rows, img_l.cols);
			printf("right size: %d, %d\n", img_r.rows, img_r.cols);

            resize(img_l, img_l, Size(g_img_w, g_img_h));
            resize(img_r, img_r, Size(g_img_w, g_img_h));
            printf("resized left size: %d, %d\n", img_l.rows, img_l.cols);
            printf("resized right size: %d, %d\n", img_r.rows, img_r.cols);

			printf("calling sky detector ...\n");

            Mat sky_mask;
            sky_det->detect(img_l, res_addr+num2str(i)+"_"+num2strbeta(j)+"_sky.png", sky_mask, g_scale);

            Mat sky_mask_beta;
            sky_det->detect(img_r, res_addr+num2str(i)+"_"+num2strbeta(j)+"_sky2.png", sky_mask_beta, g_scale);

			printf("calling sky detector end\n");

			printf("computing stereo matching waiting ...\n");

			double be = get_cur_ms();
            sv->process(img_l, img_r, sky_mask, sky_mask_beta);
			double en = get_cur_ms();
			printf("stereo matching done\n");
			printf("time cost: %lf ms\n", en - be);

			// be = get_cur_ms();
			// gsv->process(img_l, img_r);
			// en = get_cur_ms();
			// printf("stereo matching (gpu version) done\n");
			// printf("time cost: %lf ms\n", en - be);

            disp = sv->get_disp();
			publish_float32(disp_pub, disp);

            sv->show_disp(debug_view);
			publish_rgb(debug_pub, debug_view);

            imwrite(res_addr+num2str(i)+"_"+num2strbeta(j)+"_disp.png", debug_view);
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
                    if (ptr[j] == g_invalid_disp)  continue;
					
					double Zc = (cam_para.fx+cam_para.fy)/2.0 * baseline / (ptr[j]+1e-6);
					if (Zc > max_range)  continue;

                    double Xc = (j*g_scale - cam_para.cx) * Zc / cam_para.fx;
                    double Yc = (i*g_scale - cam_para.cy) * Zc / cam_para.fy;

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
