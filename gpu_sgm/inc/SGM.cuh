#include "../../inc/global.h"
#include "../../inc/utils.h"
#include "../inc/cost.cuh"
#include "../inc/aggregation.cuh"
#include "../inc/post_filter.cuh"

#include "cuda_inc.cuh"


const int CU_WIN_H = 7;
const int CU_WIN_W = 9;
const int CU_COST_WIN_H = 3;
const int CU_COST_WIN_W = 5;
const float CU_UNIQUE_RATIO = 0.7;
const int CU_MEDIAN_FILTER_H = 5;
const int CU_MEDIAN_FILTER_W = 5;
const int CU_SPECKLE_SIZE = 1000;
const int CU_SPECKLE_DIS = 2;

const bool CU_USE_8_PATH = 1;


class GPU_SGM
{
public:
    explicit GPU_SGM(int h, int w, int s, int d);
	virtual ~GPU_SGM();

	GPU_SGM(const GPU_SGM&) =delete;
    GPU_SGM& operator=(const GPU_SGM&) =delete;
	
	virtual void process(Mat &img_l, Mat &img_r);

	virtual void show_disp(Mat &debug_view); 
	virtual const Mat& get_disp() const { return filtered_disp;}

private:
	void colormap();

	cudaStream_t stream1, stream2, stream3, stream4, stream5, stream6, stream7, stream8;

    int img_h, img_w;
    int scale;
    int max_disp, invalid_disp;

	Mat img_l, img_r;
	uchar *d_img_l, *d_img_r;
	uchar *d_disp;
	float *d_filtered_disp;
	uint64_t *d_cost_table_l, *d_cost_table_r;
	float *d_cost;
	Mat disp, filtered_disp, colored_disp;

	float *d_L1, *d_L2, *d_L3, *d_L4;
	short *d_L5, *d_L6, *d_L7, *d_L8;  // use short due to my poor gpu memory
	float *d_min_L1, *d_min_L2, *d_min_L3, *d_min_L4, *d_min_L5, *d_min_L6, *d_min_L7, *d_min_L8;
	int P1, P2;

	int *d_label, *d_area;
};

typedef std::shared_ptr<GPU_SGM> GSGMPtr;
