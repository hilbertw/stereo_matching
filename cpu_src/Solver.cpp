#include "../cpu_inc/Solver.h"


Solver::Solver(int h, int w, int s, int d)
{
    assert (h>0 && w>0 && s>0 && d>0);

    assert (s==1 || s==2);

    assert (d==32 || d==64 || d==128);

    img_h = h/s;
    img_w = w/s;
    scale = s;
    max_disp = d;
    invalid_disp = d+1;

    disp.create(img_h, img_w, CV_8UC1);
    disp_beta.create(img_h, img_w, CV_8UC1);
    filtered_disp.create(img_h, img_w, CV_32FC1);
    filtered_disp_beta.create(img_h, img_w, CV_32FC1);
    colored_disp.create(img_h, img_w, CV_8UC3);

	// dsi
    cost = new float[img_h * img_w * max_disp];
    cost_table_l = new uint64_t[img_h * img_w];
    cost_table_r = new uint64_t[img_h * img_w];

	if (WEIGHTED_COST)
	{
        int win_h = WIN_H/scale;
        int win_w = WIN_W/scale;

        weight = new float[win_h * win_w];
#pragma omp parallel for
        for (int i = 0; i < win_h; ++i)
		{
            for (int j = 0; j < win_w; ++j)
			{
                weight[i*win_w + j] = exp((pow(i - win_h / 2, 2) + pow(j - win_w / 2, 2)) / -25.0);
                //std::cout << weight[i*win_w + j] << "\t";
			}
			//std::cout << std::endl;
		}
	}
	//std::cin.get();

    sky_mask.release();
    sky_mask_beta.release();
}


void Solver::show_disp(Mat &debug_view)
{
    /*
	// left border invalid
    for (int i = 0; i < disp.rows; ++i)
	{
		//uchar *ptr = disp.ptr<uchar>(i);
		float *ptr = filtered_disp.ptr<float>(i);

//        if (i < disp.rows / 4)
//        {
//            for(int j = 0; j < disp.cols; ++j)
//            {
//                ptr[j] = invalid_disp;
//            }
//        }

//        else
        {
            for(int j = 0; j < max_disp / scale; ++j)
            {
                ptr[j] = invalid_disp;
            }
        }
	}
    */
	
	// convert to RGB for better observation
	colormap();

	Mat tmp;

    debug_view = debug_view.zeros(img_h * 2, img_w, CV_8UC3);
    tmp = debug_view(Rect(0, 0, img_w, img_h));
	cvtColor(img_l, img_l, CV_GRAY2BGR);
	img_l.copyTo(tmp);
    tmp = debug_view(Rect(0, img_h - 1, img_w, img_h));
	colored_disp.copyTo(tmp);
}


void Solver::build_dsi()
{
    int win_h = WIN_H/scale;
    int win_w = WIN_W/scale;

#pragma omp parallel for
    for (int i = 0; i < img_h; ++i)
	{
        for (int j = 0; j < img_w; ++j)
		{
            for (int d = 0; d < max_disp; ++d)
			{
                int index = i * img_w * max_disp + j * max_disp + d;
                //cost[index] = SSD(img_l, img_r, Point(j, i), d, win_h, win_w, weight);
                cost[index] = CT(img_l, img_r, Point(j, i), d, win_h, win_w, weight);

				//std::cout << "[" << i << ", " << j << ", " << (int)d << "]:\t" <<  cost[index];
				//std::cin.get();
			}
		}
	}
}


void Solver::build_cost_table()
{

    cv::Mat img_l_filtered, img_r_filtered;
    cv::GaussianBlur(img_l, img_l_filtered, cv::Size(3,3), 2, 1);  // denoise
    cv::GaussianBlur(img_r, img_r_filtered, cv::Size(3,3), 2, 1);  // denoise

    int win_h = WIN_H/scale;
    int win_w = WIN_W/scale;

#pragma omp parallel for
    for (int i = 0; i < img_h; ++i)
	{
        for (int j = 0; j < img_w; ++j)
		{
            // TODO: esemble sad cost
//            CT_pts(img_l, img_r, j, i, win_h, win_w, weight, cost_table_l, cost_table_r);
            CT_pts(img_l_filtered, img_r_filtered, j, i, win_h, win_w, weight, cost_table_l, cost_table_r);
		}
	}
}


void Solver::build_dsi_from_table()
{

    bool has_sky_mask = !(sky_mask.empty() || !sky_mask.data);
//    has_sky_mask = false;
    if (has_sky_mask)
        printf("sky mask found\n");
    else
        printf("no sky mask\n");

#pragma omp parallel for
    for (int i = 0; i < img_h; ++i)
	{
        const uchar *ptr = NULL;
        if (has_sky_mask)
        {
            ptr = sky_mask.ptr<uchar>(i);
        }

        for (int j = 0; j < img_w; ++j)
		{
            int bias = i * img_w * max_disp + j * max_disp;
            if (has_sky_mask && ptr[j] == 255)
            {
                for (int d = 0; d < max_disp; ++d)
                {
                    int index = bias + d;
                    if (d==0)
                    {
                        cost[index] = 0;
                    }
                    else
                    {
                        cost[index] = 999999;
                    }
                }
            }
            else
            {
                for (int d = 0; d < max_disp; ++d)
                {
                    int index = bias + d;
                    int d_bk = std::max(j - d/scale, 0);

                    uint64_t ct_l = cost_table_l[i*img_w + j];
                    uint64_t ct_r = cost_table_r[i*img_w + d_bk];
                    cost[index] = hamming_cost(ct_l, ct_r);
                }
            }
		}
	}
}


void Solver::build_dsi_from_table_beta()
{

    bool has_sky_mask = !(sky_mask_beta.empty() || !sky_mask_beta.data);
//    has_sky_mask = false;
    if (has_sky_mask)
        printf("sky mask beta found\n");
    else
        printf("no sky mask beta\n");

#pragma omp parallel for
    for (int i = 0; i < img_h; ++i)
    {
        const uchar *ptr = NULL;
        if (has_sky_mask)
        {
            ptr = sky_mask_beta.ptr<uchar>(i);
        }

        for (int j = 0; j < img_w; ++j)
        {
            int bias = i * img_w * max_disp + j * max_disp;
            if (has_sky_mask && ptr[j] == 255)
            {
                for (int d = 0; d < max_disp; ++d)
                {
                    int index = bias + d;
                    if (d==0)
                    {
                        cost[index] = 0;
                    }
                    else
                    {
                        cost[index] = 999999;
                    }
                }
            }
            else
            {
                for (int d = 0; d < max_disp; ++d)
                {
                    int index = bias + d;
                    int d_bk = std::min(j + d/scale, img_w-1);

                    uint64_t ct_l = cost_table_l[i*img_w + d_bk];
                    uint64_t ct_r = cost_table_r[i*img_w + j];
                    cost[index] = hamming_cost(ct_l, ct_r);
                }
            }
        }
    }
}


float Solver::find_dsi_mean_max()
{
	double max_cost = 0, mean_cost = 0;
    for (int i = 0; i < img_h; ++i)
	{
        for (int j = 0; j < img_w; ++j)
		{
            for (int d = 0; d < max_disp; ++d)
			{
                int index = i * img_w * max_disp + j * max_disp + d;
				mean_cost += cost[index];
				if (cost[index] > max_cost)
				{
					max_cost = cost[index];
				}
			}
		}
	}
    mean_cost /= (img_h * img_w * max_disp);
	printf("max_cost: %lf, mean_cost: %lf\n", max_cost, mean_cost);
	return mean_cost;
}


float Solver::find_table_mean_max()
{
	double max_cost = 0, mean_cost = 0;
    for (int i = 0; i < img_h; ++i)
	{
        for (int j = 0; j < img_w; ++j)
		{
            int index = i * img_w + j;
			mean_cost += cost_table_r[index];
			if (cost_table_r[index] > max_cost)
			{
				max_cost = cost_table_r[index];
			}
		}
	}
    mean_cost /= (img_h * img_w * max_disp);
	printf("max_cost: %lf, mean_cost: %lf\n", max_cost, mean_cost);
	return mean_cost;
}


void Solver::cost_horizontal_filter(int win_size)
{
    const int index_step = (win_size/2+1)*max_disp;

    // for each row, smooth horizontal cost in the same disparity
    for (int i = 0; i < img_h; ++i)
	{
        for (int d = 0; d < max_disp; ++d)
		{
			float sum = 0;
            int index = i * img_w * max_disp + d;

			// initialize
            for (int j = 0; j < win_size; ++j)
			{
				sum += cost[index];
                index += max_disp;  // next index in the same disparity
			}

			// box filter
            for (int j = win_size/2; j < img_w - win_size/2; ++j)
			{
//				cost[i * img_w * max_disp + j * max_disp + d] = sum / win_size;
                cost[index - index_step] = sum / win_size;

                if (j == img_w - win_size/2 - 1)
                    break;

                sum += cost[index];
                sum -= cost[index - win_size * max_disp];
                index += max_disp;
			}
		}
	}
}


void Solver::cost_vertical_filter(int win_size)
{
    const int step = img_w * max_disp;
    const int index_step = (win_size/2+1)*step;

    // for each col, smooth vertical cost in the same disparity
    for (int j = 0; j < img_w; ++j)
	{
        for (int d = 0; d < max_disp; ++d)
		{
			float sum = 0;
            int index = j * max_disp + d;

			// initialize
            for (int i = 0; i < win_size; ++i)
			{
				sum += cost[index];
                index += step;
			}

			// box filter
            for (int i = win_size/2; i < img_h - win_size/2; ++i)
			{
//				cost[i * img_w * max_disp + j * max_disp + d] = sum / win_size;
                cost[index - index_step] = sum / win_size;

                if (i == img_h - win_size/2 - 1)
                    break;

                sum += cost[index];
                sum -= cost[index - win_size * step];
                index += step;
			}
		}
	}
}


void Solver::fetch_cost(float *p)
{
    memcpy(cost, p, img_h * img_w * max_disp * sizeof(float));
}


void Solver::fetch_disparity(uchar *d)
{
	int cnt = 0;
    for (int i = 0; i < disp.rows; ++i)
	{
        for (int j = 0; j < disp.cols; ++j)
		{
			disp.at<uchar>(i, j) = d[cnt++];
		}
	}
}


void Solver::fetch_disparity(float *d)
{
	int cnt = 0;
    for (int i = 0; i < filtered_disp.rows; ++i)
	{
        for (int j = 0; j < filtered_disp.cols; ++j)
		{
			filtered_disp.at<float>(i, j) = d[cnt++];
		}
	}
}


static void ccl_dfs(int row, int col, Mat &m, bool *visited, int *label, int *area, int label_cnt, int max_dis)
{
	visited[row * m.cols + col] = 1;

	if (row > 0 && fabs(m.at<float>(row, col) - m.at<float>(row - 1, col)) < max_dis)
	{
		if (!visited[(row - 1) * m.cols + col])
		{
			label[(row - 1) * m.cols + col] = label_cnt;
			++area[label_cnt];
			//printf("1: %d, %d; ", row-1, col);
			ccl_dfs(row - 1, col, m, visited, label, area, label_cnt, max_dis);
		}
	}
	if (row < m.rows - 1 && fabs(m.at<float>(row, col) - m.at<float>(row + 1, col)) < max_dis)
	{
		if (!visited[(row + 1) * m.cols + col])
		{
			label[(row + 1) * m.cols + col] = label_cnt;
			++area[label_cnt];
			//printf("2: %d, %d; ", row+1, col);
			ccl_dfs(row + 1, col, m, visited, label, area, label_cnt, max_dis);
		}
	}
	if (col > 0 && fabs(m.at<float>(row, col) - m.at<float>(row, col - 1)) < max_dis)
	{
		if (!visited[row * m.cols + col - 1])
		{
			label[row * m.cols + col - 1] = label_cnt;
			++area[label_cnt];
			//printf("3: %d, %d; ", row, col-1);
			ccl_dfs(row, col - 1, m, visited, label, area, label_cnt, max_dis);
		}
	}
	if (col < m.cols - 1 && fabs(m.at<float>(row, col) - m.at<float>(row, col + 1)) < max_dis)
	{
		if (!visited[row * m.cols + col + 1])
		{
			label[row * m.cols + col + 1] = label_cnt;
			++area[label_cnt];
			//printf("4: %d, %d; ", row, col+1);
			ccl_dfs(row, col + 1, m, visited, label, area, label_cnt, max_dis);
		}
	}
	return;
}


static void speckle_filter(Mat &m, int value, int max_size, int max_dis)
{
	bool *visited = new bool[m.rows * m.cols];
	int *label = new int[m.rows * m.cols];
	int *area = new int[m.rows * m.cols];
    for (int i = 0; i < m.rows * m.cols; ++i)
	{
		visited[i] = 0;
		label[i] = 0;
		area[i] = 0;
	}

	int label_cnt = 0;
    for (int i = 0; i < m.rows; ++i)
	{
        for (int j = 0; j < m.cols; ++j)
		{
			if (visited[i * m.cols + j])  continue;

			label[i*m.cols + j] = ++label_cnt;
			++area[label_cnt];
			ccl_dfs(i, j, m, visited, label, area, label_cnt, max_dis);
		}
	}

    for (int i = 0; i < m.rows; ++i)
	{
        for (int j = 0; j < m.cols; ++j)
		{
			if (area[label[i*m.cols + j]] <= max_size)
			{
				m.at<float>(i, j) = value;
			}
		}
	}

	delete[] visited;
	delete[] label;
	delete[] area;
}


static int Find(int i, int *label)
{
	while (i != label[i])
	{
		i = label[i];
	}
	return i;
}


static void Union(int i, int j, int *label)  // join i to j
{
	int label_a = Find(i, label);
	int label_b = Find(j, label);
	if (label_a != label_b)
	{
		label[label_a] = label_b;
	}
}


static void speckle_filter_new(Mat &m, int value, int max_size, int max_dis)
{
	int *label = new int[m.rows * m.cols];
	int *area = new int[m.rows * m.cols];
#pragma omp parallel for
    for (int i = 0; i < m.rows * m.cols; ++i)
	{
		label[i] = i;
		area[i] = 0;
	}

#pragma omp parallel for
    for (int i = 0; i < m.rows * m.cols; ++i)
	{
		int row = i / m.cols;
		int col = i % m.cols;
		if (row > 0 && fabs(m.at<float>(row, col) - m.at<float>(row - 1, col)) < max_dis)
		{
			Union(i - m.cols, i, label);
		}
		if (row < m.rows - 1 && fabs(m.at<float>(row, col) - m.at<float>(row + 1, col)) < max_dis)
		{
			Union(i + m.cols, i, label);
		}
		if (col > 0 && fabs(m.at<float>(row, col) - m.at<float>(row, col - 1)) < max_dis)
		{
			Union(i - 1, i, label);
		}
		if (col < m.cols - 1 && fabs(m.at<float>(row, col) - m.at<float>(row, col + 1)) < max_dis)
		{
			Union(i + 1, i, label);
		}
	}

#pragma omp parallel for
    for (int i = 0; i < m.rows * m.cols; ++i)
	{
		label[i] = Find(i, label);
		area[label[i]]++;
	}

#pragma omp parallel for
    for (int i = 0; i < m.rows; ++i)
	{
        for (int j = 0; j < m.cols; ++j)
		{
			if (area[label[i*m.cols + j]] <= max_size)
			{
				m.at<float>(i, j) = value;
			}
		}
	}
}


void Solver::compute_subpixel(const Mat &disp, Mat &filtered_disp)
{

#pragma omp parallel for
    for (int i = 0; i < img_h; ++i)
    {
        for (int j = 0; j < img_w; ++j)
        {
            int d = disp.at<uchar>(i, j);
            if (d > max_disp-1)
            {
                filtered_disp.at<float>(i, j) = invalid_disp;
            }
            else if (!d || d == max_disp - 1)
            {
                filtered_disp.at<float>(i, j) = d;
            }
            else
            {
                int index = i * img_w * max_disp + j * max_disp + d;
                float cost_d = cost[index];
                float cost_d_sub = cost[index - 1];
                float cost_d_plus = cost[index + 1];
                filtered_disp.at<float>(i, j) =
                        std::min(d + (cost_d_sub - cost_d_plus) / (2 * (cost_d_sub + cost_d_plus - 2 * cost_d)), (max_disp-1)*1.f);
            }
        }
    }
}


void Solver::post_filter()
{
	double be = get_cur_ms();

	// median filter
	std::vector<int> v;
    for (int i = MEDIAN_FILTER_H / 2; i < filtered_disp.rows - MEDIAN_FILTER_H / 2; ++i)
	{
        for (int j = MEDIAN_FILTER_W / 2; j < filtered_disp.cols - MEDIAN_FILTER_W / 2; ++j)
		{
            if (filtered_disp.at<float>(i, j) <= max_disp-1)  continue;
			int valid_cnt = 0;
			v.clear();
            for (int m = i - MEDIAN_FILTER_H / 2; m <= i + MEDIAN_FILTER_H / 2; ++m)
			{
                for (int n = j - MEDIAN_FILTER_W / 2; n <= j + MEDIAN_FILTER_W / 2; ++n)
				{
                    if (filtered_disp.at<float>(m, n) <= max_disp - 1)
					{
						v.push_back(filtered_disp.at<float>(m, n));
						valid_cnt++;
					}
				}
			}
			if (valid_cnt > MEDIAN_FILTER_W * MEDIAN_FILTER_H / 2)
			{
                std::sort(v.begin(), v.end());
				filtered_disp.at<float>(i, j) = v[valid_cnt / 2];
			}
		}
	}

	 //speckle_filter
    //speckle_filter(filtered_disp, invalid_disp, SPECKLE_SIZE/scale, SPECKLE_DIS);
    speckle_filter_new(filtered_disp, invalid_disp, SPECKLE_SIZE/scale, SPECKLE_DIS);
	printf("post process takes %lf ms\n", get_cur_ms() - be);

	/*
    for (int i = 0; i < img_h; ++i)
	{
        for (int j = 0; j < img_w; ++j)
		{
			int d = disp.at<uchar>(i, j);
			float d_ = filtered_disp.at<float>(i, j);
			printf("%d -> %f, ", d, d_);
		}
		printf("\n");
	}
	*/
}


void Solver::colormap()
{
	float disp_value = 0;
    for (int i = 0; i < disp.rows; ++i)
	{
        for (int j = 0; j < disp.cols; ++j)
		{
            disp_value = filtered_disp.at<float>(i, j);
//            disp_value = disp.at<uchar>(i, j);
            if (disp_value > max_disp - 1)
			{
				colored_disp.at<Vec3b>(i, j)[0] = 0;
				colored_disp.at<Vec3b>(i, j)[1] = 0;
				colored_disp.at<Vec3b>(i, j)[2] = 0;
			}
			else
			{
                disp_value *= (256 / (max_disp));
				if (disp_value <= 51)
				{
					colored_disp.at<Vec3b>(i, j)[0] = 255;
					colored_disp.at<Vec3b>(i, j)[1] = disp_value * 5;
					colored_disp.at<Vec3b>(i, j)[2] = 0;
				}
				else if (disp_value <= 102)
				{
					disp_value -= 51;
					colored_disp.at<Vec3b>(i, j)[0] = 255 - disp_value * 5;
					colored_disp.at<Vec3b>(i, j)[1] = 255;
					colored_disp.at<Vec3b>(i, j)[2] = 0;
				}
				else if (disp_value <= 153)
				{
					disp_value -= 102;
					colored_disp.at<Vec3b>(i, j)[0] = 0;
					colored_disp.at<Vec3b>(i, j)[1] = 255;
					colored_disp.at<Vec3b>(i, j)[2] = disp_value * 5;
				}
				else if (disp_value <= 204)
				{
					disp_value -= 153;
					colored_disp.at<Vec3b>(i, j)[0] = 0;
					colored_disp.at<Vec3b>(i, j)[1] = 255 - uchar(128.0*disp_value / 51.0 + 0.5);
					colored_disp.at<Vec3b>(i, j)[2] = 255;
				}
				else
				{
					disp_value -= 204;
					colored_disp.at<Vec3b>(i, j)[0] = 0;
					colored_disp.at<Vec3b>(i, j)[1] = 127 - uchar(127.0*disp_value / 51.0 + 0.5);
					colored_disp.at<Vec3b>(i, j)[2] = 255;
				}
			}
		}
	}
}


Solver::~Solver()
{
	delete[] cost;
    delete[] cost_table_l;
    delete[] cost_table_r;
	if (WEIGHTED_COST)
	{
		delete[] weight;
	}
}
