#include "../cpu_inc/utils.h"


double get_cur_ms()
{
	return getTickCount() * 1000.f / getTickFrequency();
}

std::string num2str(int i)
{
	char ss[50];
	sprintf(ss, "%06d", i);
	return ss;
}

std::string num2strbeta(int i)
{
	char ss[20];
	sprintf(ss, "%02d", i);
	return ss;
}

void stereo_record(int camid, std::string address)
{
	int cnt = 0;
	Mat frame;
	VideoCapture cap(camid);
	if (!cap.isOpened())
	{
		std::cout << "reading camera error" << std::endl;
		std::cin.get();
		return;
	}
	cap.set(CV_CAP_PROP_FRAME_WIDTH, 1280 * 2);
	cap.set(CV_CAP_PROP_FRAME_HEIGHT, 720);

	printf("initialing camera ...\n");
    sleep(5);  // wait 5 secs
	printf("finished\n");
	namedWindow("video");

	while (cap.isOpened())
	{
		cap >> frame;
		if (frame.empty())  break;
        std::string img_name = address + num2str(cnt++) + ".png";
		imshow("video", frame);
		imwrite(img_name, frame);
		if (waitKey(10) == 13)
		{
			printf("record finished. exit...");
			destroyAllWindows();
			break;
		}
        sleep(1);
	}
}

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
