#include "../cpu_inc/Solver.h"


Solver::Solver()
{
	disp.create(IMG_H, IMG_W, CV_8UC1);
	filtered_disp.create(IMG_H, IMG_W, CV_32FC1);
	colored_disp.create(IMG_H, IMG_W, CV_8UC3);

	// dsi
	cost = new float[IMG_H * IMG_W * MAX_DISP];
	cost_table_l = new uint64_t[IMG_H * IMG_W];
	cost_table_r = new uint64_t[IMG_H * IMG_W];

	if (WEIGHTED_COST)
	{
		weight = new float[WIN_H * WIN_W];
#pragma omp parallel for
		for (int i = 0; i < WIN_H; i++)
		{
			for (int j = 0; j < WIN_W; j++)
			{
				weight[i*WIN_W + j] = exp(((i - WIN_H / 2)*(i - WIN_H / 2) + (j - WIN_W / 2)*(j - WIN_W / 2)) / -25.0);
				//std::cout << weight[i*WIN_W + j] << "\t";
			}
			//std::cout << std::endl;
		}
	}
	//std::cin.get();
}


void Solver::show_disp()
{
	// left border invalid
	for (int i = 0; i < disp.rows; i++)
	{
		//uchar *ptr = disp.ptr<uchar>(i);
		float *ptr = filtered_disp.ptr<float>(i);
		for(int j = 0; j < MAX_DISP; j++)
		{
			ptr[j] = INVALID_DISP;
		}
	}
	
	// convert to RGB for better observation
	colormap();

	Mat debug_view, tmp;

	debug_view = debug_view.zeros(IMG_H * 2, IMG_W, CV_8UC3);
	tmp = debug_view(Rect(0, 0, IMG_W, IMG_H));
	cvtColor(img_l, img_l, CV_GRAY2BGR);
	img_l.copyTo(tmp);
	tmp = debug_view(Rect(0, IMG_H - 1, IMG_W, IMG_H));
	colored_disp.copyTo(tmp);

	namedWindow("disp_map", 1);
	imshow("disp_map", debug_view);
	//imwrite("example/test.png", debug_view);

	waitKey();
	destroyWindow("disp_map");
}


void Solver::process(Mat &img_l, Mat &img_r)
{
	std::cout << "Class Solver does nothing!" << std::endl;
	std::cin.get();
}


void Solver::build_dsi()
{
#pragma omp parallel for
	for (int i = 0; i < IMG_H; i++)
	{
		for (int j = 0; j < IMG_W; j++)
		{
			for (int d = 0; d < MAX_DISP; d++)
			{
				int index = i * IMG_W * MAX_DISP + j * MAX_DISP + d;
				//cost[index] = SSD(img_l, img_r, Point(j, i), d, WIN_H, WIN_W, weight);
				cost[index] = CT(img_l, img_r, Point(j, i), d, WIN_H, WIN_W, weight);

				//std::cout << "[" << i << ", " << j << ", " << (int)d << "]:\t" <<  cost[index];
				//std::cin.get();
			}
		}
	}
}


void Solver::build_cost_table()
{
#pragma omp parallel for
	for (int i = 0; i < IMG_H; i++)
	{
		for (int j = 0; j < IMG_W; j++)
		{
			CT_pts(img_l, img_r, j, i, WIN_H, WIN_W, weight, cost_table_l, cost_table_r);
		}
	}
}


void Solver::build_dsi_from_table()
{
#pragma omp parallel for
	for (int i = 0; i < IMG_H; i++)
	{
		for (int j = 0; j < IMG_W; j++)
		{
			for (int d = 0; d < MAX_DISP; d++)
			{
				int index = i * IMG_W * MAX_DISP + j * MAX_DISP + d;
				uint64_t ct_l = cost_table_l[i*IMG_W + j];
				uint64_t ct_r = cost_table_r[i*IMG_W + MAX(j - d, 0)];
				cost[index] = hamming_cost(ct_l, ct_r);
			}
		}
	}
}


float Solver::find_dsi_mean_max()
{
	double max_cost = 0, mean_cost = 0;
	for (int i = 0; i < IMG_H; i++)
	{
		for (int j = 0; j < IMG_W; j++)
		{
			for (int d = 0; d < MAX_DISP; d++)
			{
				int index = i * IMG_W * MAX_DISP + j * MAX_DISP + d;
				mean_cost += cost[index];
				if (cost[index] > max_cost)
				{
					max_cost = cost[index];
				}
			}
		}
	}
	mean_cost /= (IMG_H * IMG_W * MAX_DISP);
	printf("max_cost: %lf, mean_cost: %lf\n", max_cost, mean_cost);
	return mean_cost;
}


float Solver::find_table_mean_max()
{
	double max_cost = 0, mean_cost = 0;
	for (int i = 0; i < IMG_H; i++)
	{
		for (int j = 0; j < IMG_W; j++)
		{
			int index = i * IMG_W + j;
			mean_cost += cost_table_r[index];
			if (cost_table_r[index] > max_cost)
			{
				max_cost = cost_table_r[index];
			}
		}
	}
	mean_cost /= (IMG_H * IMG_W * MAX_DISP);
	printf("max_cost: %lf, mean_cost: %lf\n", max_cost, mean_cost);
	return mean_cost;
}


void Solver::cost_horizontal_filter(int win_size)
{
	for (int i = 0; i < IMG_H; i++)
	{
		for (int d = 0; d < MAX_DISP; d++)
		{
			float sum = 0;
			int index = i * IMG_W * MAX_DISP + d;
			// initialize
			for (int k = 0; k < win_size; k++)
			{
				sum += cost[index];
				index += MAX_DISP;
			}
			// box filter
			for (int j = win_size/2; j < IMG_W - win_size/2; j++)
			{
				cost[i * IMG_W * MAX_DISP + j * MAX_DISP + d] = sum / win_size;
				if (j < IMG_W - win_size / 2 - 1)
				{
					sum += cost[index];
					sum -= cost[index - win_size * MAX_DISP];
					index += MAX_DISP;
				}
			}
		}
	}
}


void Solver::cost_vertical_filter(int win_size)
{
	for (int j = 0; j < IMG_W; j++)
	{
		for (int d = 0; d < MAX_DISP; d++)
		{
			float sum = 0;
			int index = j * MAX_DISP + d;
			// initialize
			for (int k = 0; k < win_size; k++)
			{
				sum += cost[index];
				index += IMG_W * MAX_DISP;
			}
			// box filter
			for (int i = win_size/2; i < IMG_H - win_size/2; i++)
			{
				cost[i * IMG_W * MAX_DISP + j * MAX_DISP + d] = sum / win_size;
				if (i < IMG_H - win_size / 2 - 1)
				{
					sum += cost[index];
					sum -= cost[index - win_size * IMG_W * MAX_DISP];
					index += IMG_W * MAX_DISP;
				}
			}
		}
	}
}


void Solver::fetch_cost(float *p)
{
	memcpy(cost, p, IMG_H * IMG_W * MAX_DISP * sizeof(float));
}


void Solver::fetch_disparity(uchar *d)
{
	int cnt = 0;
	for (int i = 0; i < disp.rows; i++)
	{
		for (int j = 0; j < disp.cols; j++)
		{
			disp.at<uchar>(i, j) = d[cnt++];
		}
	}
}


void Solver::fetch_disparity(float *d)
{
	int cnt = 0;
	for (int i = 0; i < filtered_disp.rows; i++)
	{
		for (int j = 0; j < filtered_disp.cols; j++)
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
	for (int i = 0; i < m.rows * m.cols; i++)
	{
		visited[i] = 0;
		label[i] = 0;
		area[i] = 0;
	}

	int label_cnt = 0;
	for (int i = 0; i < m.rows; i++)
	{
		for (int j = 0; j < m.cols; j++)
		{
			if (visited[i * m.cols + j])  continue;

			label[i*m.cols + j] = ++label_cnt;
			++area[label_cnt];
			ccl_dfs(i, j, m, visited, label, area, label_cnt, max_dis);
		}
	}

	for (int i = 0; i < m.rows; i++)
	{
		for (int j = 0; j < m.cols; j++)
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
	for (int i = 0; i < m.rows * m.cols; i++)
	{
		label[i] = i;
		area[i] = 0;
	}

#pragma omp parallel for
	for (int i = 0; i < m.rows * m.cols; i++)
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
	for (int i = 0; i < m.rows * m.cols; i++)
	{
		label[i] = Find(i, label);
		area[label[i]]++;
	}

#pragma omp parallel for
	for (int i = 0; i < m.rows; i++)
	{
		for (int j = 0; j < m.cols; j++)
		{
			if (area[label[i*m.cols + j]] <= max_size)
			{
				m.at<float>(i, j) = value;
			}
		}
	}
}


void Solver::post_filter()
{
	double be = get_cur_ms();
	// sub-pixel
#pragma omp parallel for
	for (int i = 0; i < IMG_H; i++)
	{
		for (int j = 0; j < IMG_W; j++)
		{
			int d = disp.at<uchar>(i, j);
			if (d > MAX_DISP-1)
			{
				filtered_disp.at<float>(i, j) = INVALID_DISP;
			}
			else if (!d || d == MAX_DISP - 1)
			{
				filtered_disp.at<float>(i, j) = d;
			}
			else
			{
				int index = i * IMG_W * MAX_DISP + j * MAX_DISP + d;
				float cost_d = cost[index];
				float cost_d_sub = cost[index - 1];
				float cost_d_plus = cost[index + 1];
				filtered_disp.at<float>(i, j) = MIN(d + (cost_d_sub - cost_d_plus) / (2 * (cost_d_sub + cost_d_plus - 2 * cost_d)), MAX_DISP-1);
			}
		}
	}

	// median filter
	vector<int> v;
	for (int i = MEDIAN_FILTER_H / 2; i < filtered_disp.rows - MEDIAN_FILTER_H / 2; i++)
	{
		for (int j = MEDIAN_FILTER_W / 2; j < filtered_disp.cols - MEDIAN_FILTER_W / 2; j++)
		{
			if (filtered_disp.at<float>(i, j) <= MAX_DISP-1)  continue;
			int valid_cnt = 0;
			v.clear();
			for (int m = i - MEDIAN_FILTER_H / 2; m <= i + MEDIAN_FILTER_H / 2; m++)
			{
				for (int n = j - MEDIAN_FILTER_W / 2; n <= j + MEDIAN_FILTER_W / 2; n++)
				{
					if (filtered_disp.at<float>(m, n) <= MAX_DISP - 1)
					{
						v.push_back(filtered_disp.at<float>(m, n));
						valid_cnt++;
					}
				}
			}
			if (valid_cnt > MEDIAN_FILTER_W * MEDIAN_FILTER_H / 2)
			{
				sort(v.begin(), v.end());
				filtered_disp.at<float>(i, j) = v[valid_cnt / 2];
			}
		}
	}

	 //speckle_filter
	//speckle_filter(filtered_disp, INVALID_DISP, SPECKLE_SIZE, SPECKLE_DIS);
	speckle_filter_new(filtered_disp, INVALID_DISP, SPECKLE_SIZE, SPECKLE_DIS);
	printf("post process takes %lf ms\n", get_cur_ms() - be);

	/*
	for (int i = 0; i < IMG_H; i++)
	{
		for (int j = 0; j < IMG_W; j++)
		{
			int d = disp.at<uchar>(i, j);
			float d_ = filtered_disp.at<float>(i, j);
			printf("%d -> %f, ", d, d_);
		}
		printf("\n");
	}
	*/
}


void  Solver::colormap()
{
	float disp_value = 0;
	for (int i = 0; i < disp.rows; i++)
	{
		for (int j = 0; j < disp.cols; j++)
		{
			disp_value = filtered_disp.at<float>(i, j);
			//disp_value = disp.at<uchar>(i, j);
			if (disp_value > MAX_DISP - 1)
			{
				colored_disp.at<Vec3b>(i, j)[0] = 0;
				colored_disp.at<Vec3b>(i, j)[1] = 0;
				colored_disp.at<Vec3b>(i, j)[2] = 0;
			}
			else
			{
				disp_value *= (256 / (MAX_DISP));
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
	if (WEIGHTED_COST)
	{
		delete[] weight;
	}
}