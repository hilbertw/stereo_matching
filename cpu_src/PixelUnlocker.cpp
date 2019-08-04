#include "PixelUnlocker.h"

PixelUnlocker::PixelUnlocker()
{}

PixelUnlocker::~PixelUnlocker()
{}

Mat PixelUnlocker::unlock(Mat &img_l, Mat &img_r, Mat &disp)
{
    assert(img_l.rows == img_r.rows && img_l.rows == disp.rows);
    assert(img_l.cols == img_r.cols && img_l.cols == disp.cols);

    Mat new_disp = disp.clone();

    int iter_num = 5;
    while(iter_num-- > 0)
    {
        // Iu + dIx = 0
        for (int i=0; i<disp.rows; ++i)
        {
            for (int j=0; j<disp.cols; ++j)
            {
                if (disp.at<float>(i,j) > MAX_DISP - 1)  continue;  // invalid

                double d_off = 0, weight_sum = 0;
                int valid_sum = 0;
                for (int v=-WIN_H/2; v<=WIN_H/2; ++v)
                {
                    for (int u=-WIN_W/2; u<=WIN_W/2; ++u)
                    {
                        int m = i+v;
                        if (m<0)  continue;
                        if (m>disp.rows-1)  continue;

                        int n = j+u;
                        if (n<0)  continue;
                        if (n>disp.cols-1)  continue;

                        if (disp.at<float>(m,n) > MAX_DISP - 1)  continue;  // invalid
                        if (fabs(disp.at<float>(i,j) - disp.at<float>(m,n)) > 2)  continue;

                        int d_ini = disp.at<float>(m,n);
                        if (n-d_ini < 0)  continue;

                        int n_sub = n-1;
                        if (n_sub<0)  continue;
                        int n_plus = n+1;
                        if (n_plus>disp.cols-1)  continue;

                        valid_sum++;

                        double weight = exp(-(pow(v, 2) + pow(u, 2)) / (2*pow(WIN_W/2, 2)));
                        // weight = 1.0;

                        double Iu = img_l.at<uchar>(m,n) - img_r.at<uchar>(m, n-d_ini);
                        double Ix = img_l.at<uchar>(m,n_plus) - img_l.at<uchar>(m,n_sub) / 2;
                        
                        if (Ix < 0.001)
                            d_off = 0;
                        else
                            d_off += weight * (- Iu / Ix);
                        
                        weight_sum += weight;
                    }
                }
                new_disp.at<float>(i,j) -= d_off / weight_sum;
                // printf("%d, %d: %f\n", i, j, d_off / weight_sum);
            }
        }

        disp = new_disp.clone();
        printf("pixel_unlock itering\n");
    }
    printf("pixel_unlock ok\n");

    return new_disp;
}