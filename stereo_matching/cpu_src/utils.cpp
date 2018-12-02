#include "../cpu_inc/utils.h"


double get_cur_ms()
{
	return getTickCount() * 1000.f / getTickFrequency();
}

string num2str(int i)
{
	char ss[50];
	sprintf(ss, "%06d", i);
	return ss;
}

void stereo_record(int camid, string address)
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
	Sleep(5000);
	printf("finished\n");
	namedWindow("video");

	while (cap.isOpened())
	{
		cap >> frame;
		if (frame.empty())  break;
		string img_name = address + num2str(cnt++) + ".png";
		imshow("video", frame);
		imwrite(img_name, frame);
		if (waitKey(10) == 13)
		{
			printf("record finished. exit...");
			destroyAllWindows();
			break;
		}
		Sleep(200);
	}
}
