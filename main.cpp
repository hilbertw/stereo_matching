#include "cpu_inc/global.h"
#include "cpu_inc/BM.h"
#include "cpu_inc/SGM.h"
#include "cpu_inc/utils.h"
#include "gpu_inc/SGM.cuh"
#include "gpu_inc/cost.cuh"

#define OFFLINE_TEST 1

std::string example_addr = "/home/hunterlew/catkin_ws/src/stereo_matching/";

int main(int argc, char **argv)
{
	ros::init(argc, argv, "stereo_node");
	ros::NodeHandle nh("~");

	//stereo_record(1, "E:\\stereo_181202\\2\\");

#if OFFLINE_TEST
	Mat disp;
	if (!USE_GPU)
	{
        std::tr1::shared_ptr<SGM> sv(new SGM);

		for (int cnt = 0; cnt < 1; cnt++)
		{
			Mat img_l = imread(example_addr+"example/kitti_0_left.png", IMREAD_GRAYSCALE);
			Mat img_r = imread(example_addr+"example/kitti_0_right.png", IMREAD_GRAYSCALE);

            printf("left size: %d, %d\n", img_l.rows, img_l.cols);
            printf("right size: %d, %d\n", img_r.rows, img_r.cols);

            resize(img_l, img_l, Size(IMG_W, IMG_H));
            resize(img_r, img_r, Size(IMG_W, IMG_H));
            printf("resized left size: %d, %d\n", img_l.rows, img_l.cols);
            printf("resized right size: %d, %d\n", img_r.rows, img_r.cols);

			printf("waiting ...\n");

			double be = get_cur_ms();
			sv->process(img_l, img_r);
			double en = get_cur_ms();
			printf("done ...\n");
			printf("time cost: %lf ms\n", en - be);
			sv->show_disp();
		}
		disp = sv->get_disp();
	}
	else
	{
        std::tr1::shared_ptr<GPU_SGM> g_sv(new GPU_SGM);

		for (int cnt = 0; cnt < 1; cnt++)
		{
			Mat img_l = imread(example_addr+"example/kitti_0_left.png", IMREAD_GRAYSCALE);
			Mat img_r = imread(example_addr+"example/kitti_0_right.png", IMREAD_GRAYSCALE);

            printf("left size: %d, %d\n", img_l.rows, img_l.cols);
            printf("right size: %d, %d\n", img_r.rows, img_r.cols);

			resize(img_l, img_l, Size(IMG_W, IMG_H));
			resize(img_r, img_r, Size(IMG_W, IMG_H));
            printf("resized left size: %d, %d\n", img_l.rows, img_l.cols);
            printf("resized right size: %d, %d\n", img_r.rows, img_r.cols);

			printf("waiting ...\n");

			//cv::StereoSGBM sgbm;
			//sgbm.preFilterCap = 0;
			//int SADWindowSize = 11;
			//int cn = 1;
			//sgbm.SADWindowSize = SADWindowSize > 0 ? SADWindowSize : 3;
			//sgbm.P1 = 4 * cn*sgbm.SADWindowSize*sgbm.SADWindowSize;
			//sgbm.P2 = 32 * cn*sgbm.SADWindowSize*sgbm.SADWindowSize;
			//sgbm.minDisparity = 0;
			//sgbm.numberOfDisparities = 128;
			//sgbm.uniquenessRatio = 30;
			//sgbm.speckleWindowSize = 250;
			//sgbm.speckleRange = 2;
			//sgbm.disp12MaxDiff = 0;

			//Mat disp, disp8;
			//sgbm(img_l, img_r, disp);
			//disp.convertTo(disp8, CV_8U, 255 / (128 *16.));
			//namedWindow("disparity" , 1);
			//imshow("disparity" , disp8);
			//waitKey();

			double be = get_cur_ms();
			g_sv->process(img_l, img_r);
			double en = get_cur_ms();
			printf("done ...\n");
			printf("time cost: %lf ms\n", en - be);
			g_sv->show_disp();
		}
		disp = g_sv->get_disp();
		printf("disp size: %d, %d\n", disp.rows, disp.cols);
	}

	Mat rgb_l = imread(example_addr+"example/left_0.png");
	resize(rgb_l, rgb_l, Size(IMG_W, IMG_H));

	// read calibration
	std::ifstream in;
	in.open(example_addr+"example/calib_0.txt");
	if (!in.is_open()){
		printf("reading calib file failed\n");
		std::cin.get();
		return 0;
	}
    std::string str, str_tmp;
	std::stringstream ss;
	//while (std::getline(in, str)){
	//	std::cout << str << std::endl;
	//}
	std::getline(in, str);  // only read left cam P0
	ss.clear();
	ss.str(str);

	double fx, fy, cx, cy;
	for (int i = 0; i < 13; i++)
	{
		if (i == 1)
		{
			ss >> fx;
		}
		else if (i == 3)
		{
			ss >> cx;
		}
		else if (i == 6)
		{
			ss >> fy;
		}
		else if (i == 7)
		{
			ss >> cy;
		}
		else
		{
			ss >> str_tmp;
		}
	}
	printf("P0 intrinsic param: \nfx: %lf, fy: %lf, cx: %lf, cy: %lf\n", fx, fy, cx, cy);

	// convert disp -> Zc and compute Xc, Yc
	// +Zc points to front, +Xc points to right, +Yc points to down
	double baseline = 0.5;
	double *Xc = new double[disp.rows * disp.cols];
	double *Yc = new double[disp.rows * disp.cols];
	double *Zc = new double[disp.rows * disp.cols];

	double min_X = DBL_MAX, max_X = -DBL_MAX;
	double min_Y = DBL_MAX, max_Y = -DBL_MAX;
	double min_Z = DBL_MAX, max_Z = -DBL_MAX;

	for (int i = 0; i < disp.rows; i++)
	{
		float *ptr = disp.ptr<float>(i);
		for (int j = 0; j < disp.cols; j++)
		{
			if (ptr[j] == INVALID_DISP)
				Zc[i*disp.cols + j] = -1;  // invalid
			else
				Zc[i*disp.cols + j] = (fx+fy)/2.0 * baseline / (ptr[j]+1e-6);

			if (Zc[i*disp.cols + j] >= 0)
			{
				Xc[i*disp.cols + j] = (j - cx/SCALE) * Zc[i*disp.cols + j] / fx;
				Yc[i*disp.cols + j] = (i - cy/SCALE) * Zc[i*disp.cols + j] / fy;

				if (Xc[i*disp.cols + j] > max_X)  max_X = Xc[i*disp.cols + j];
				if (Xc[i*disp.cols + j] < min_X)  min_X = Xc[i*disp.cols + j];
				if (Yc[i*disp.cols + j] > max_Y)  max_Y = Yc[i*disp.cols + j];
				if (Yc[i*disp.cols + j] < min_Y)  min_Y = Yc[i*disp.cols + j];
				if (Zc[i*disp.cols + j] > max_Z)  max_Z = Zc[i*disp.cols + j];
				if (Zc[i*disp.cols + j] < min_Z)  min_Z = Zc[i*disp.cols + j];
			}
		}
	}
	// fov in camera coords
	printf("minx: %lf, maxx:%lf\n", min_X, max_X);
	printf("miny: %lf, maxy:%lf\n", min_Y, max_Y);  //  indicates the camera height from ground
	printf("minz: %lf, maxz:%lf\n", min_Z, max_Z);

/*
	// write file for meshlab visualization
	float max_range = 100;
	printf("generating data file ...\n");
	std::ofstream out;
	out.open(example_addr+"example/pt_0-new.txt");
	for (int i = 0; i < disp.rows; i++)
	{
		for (int j = 0; j < disp.cols; j++)
		{
			if (Zc[i*disp.cols + j] >= 0 && Zc[i*disp.cols + j] < max_range)
			{
				std::stringstream ss;
				std::string str;
				int R_value = rgb_l.at<Vec3b>(i, j)[2];
				int G_value = rgb_l.at<Vec3b>(i, j)[1];
				int B_value = rgb_l.at<Vec3b>(i, j)[0];
				ss << Xc[i*disp.cols + j] << ";" << Yc[i*disp.cols + j] << ";" << Zc[i*disp.cols + j] << ";"
					<< R_value << ";" << G_value << ";" << B_value << ";"  << std::endl;
				str = ss.str();
				out << str;
			}
		}
	}
	printf("generating data file finished\n");
*/

#else
	Mat disp;
	Mat frame;
	VideoCapture cap(0);
	if (!cap.isOpened())
	{
		std::cout << "reading camera error" << std::endl;
		std::cin.get();
		return -1;
	}
	cap.set(CV_CAP_PROP_FRAME_WIDTH, IMG_W*2);
	cap.set(CV_CAP_PROP_FRAME_HEIGHT, IMG_H);

	printf("initialing camera ...\n");
	Sleep(5000);
	printf("finished\n");

	if (!USE_GPU)
	{
		Solver *sv = new SGM();

		while (cap.isOpened())
		{
			cap >> frame;
			cvtColor(frame, frame, CV_BGR2GRAY);
			Mat img_l = frame(Rect(0, 0, frame.cols / 2, frame.rows));
			Mat img_r = frame(Rect(frame.cols / 2, 0, frame.cols / 2, frame.rows));
			resize(img_l, img_l, Size(IMG_W, IMG_H));
			resize(img_r, img_r, Size(IMG_W, IMG_H));
			printf("waiting ...\n");

			double be = get_cur_ms();
			sv->process(img_l, img_r);
			double en = get_cur_ms();
			printf("done ...\n");
			printf("time cost: %lf ms\n\n", en - be);
			sv->show_disp();
		}
		disp = sv->get_disp();
		delete sv;
	}
	else
	{
		GPU_SGM *g_sv = new GPU_SGM();

		while (cap.isOpened())
		{
			cap >> frame;
			cvtColor(frame, frame, CV_BGR2GRAY);
			Mat img_l = frame(Rect(0, 0, frame.cols / 2, frame.rows));
			Mat img_r = frame(Rect(frame.cols / 2, 0, frame.cols / 2, frame.rows));
			resize(img_l, img_l, Size(IMG_W, IMG_H));
			resize(img_r, img_r, Size(IMG_W, IMG_H));
			printf("waiting ...\n");

			double be = get_cur_ms();
			g_sv->process(img_l, img_r);
			double en = get_cur_ms();
			printf("done ...\n");
			printf("time cost: %lf ms\n\n", en - be);
			g_sv->show_disp();
		}
		disp = g_sv->get_disp();
		delete g_sv;
	}
#endif
	
	// ros::spin();
	cv::destroyAllWindows();
	return 0;
}
