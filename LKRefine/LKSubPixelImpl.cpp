#include "LKSubPixelImpl.h"

LKSubPixelImpl::LKSubPixelImpl(int h, int w, int s, int d)
    : LKSubPixel(h, w, s, d),
      img_h(h/s),
      img_w(w/s),
      scale(s),
      max_disp(d),
      invalid_disp(d+1) {}

LKSubPixelImpl::~LKSubPixelImpl() {}

void LKSubPixelImpl::LKRefine(const cv::Mat &img_l,
                              const cv::Mat &img_r,
                              cv::Mat &disp_float)
{
    assert (img_l.rows == img_r.rows);
    assert (img_l.cols == img_r.cols);
    assert (img_l.channels() == img_r.channels());
    assert (img_l.type() == img_r.type());
    assert (img_l.type() == CV_8UC1);

    assert (img_l.rows == disp_float.rows);
    assert (img_l.cols == disp_float.cols);
    assert (disp_float.channels() == 1);
    assert (disp_float.type() == CV_32FC1);

    if (scale > 1)
    {
        this->img_l.create(img_h, img_w, CV_8UC1);
        this->img_r.create(img_h, img_w, CV_8UC1);
        for (int i=0; i<img_h; ++i)
        {
            const uchar *ptr_l = img_l.ptr<uchar>(i*scale);
            const uchar *ptr_r = img_r.ptr<uchar>(i*scale);
            uchar *ptr_l_new = this->img_l.ptr<uchar>(i);
            uchar *ptr_r_new = this->img_r.ptr<uchar>(i);
            for (int j=0; j<img_w; ++j)
            {
                ptr_l_new[j] = ptr_l[j*scale];
                ptr_r_new[j] = ptr_r[j*scale];
            }
        }
    }
    else
    {
        this->img_l = img_l;
        this->img_r = img_r;
    }

    printf("start LK subpixel refine ...\n");
    LKRefineCore(disp_float);
    printf("LK subpixel refine end\n");
}

void LKSubPixelImpl::LKRefineCore(cv::Mat &disp_float) const
{
    // init 
    cv::Mat new_disp = disp_float.clone();

    // Ix_l * doff = I_r(u-dinit-doff) - I_l(u)
    // Ax = b

    success_num = 0;
    process_num = 0;

    // compute gradients on image_left
    const int half_win_size = win_size / 2;
    cv::Mat Ix_l(img_h, img_w, CV_32FC1, cv::Scalar(0));
    for (int i=half_win_size; i<img_h-half_win_size; ++i)
    {
        for (int j=half_win_size; j<img_w-half_win_size; ++j)
        {
            Ix_l.at<float>(i,j) = 
                (img_l.at<uchar>(i,j+1) - img_l.at<uchar>(i,j-1)) * 0.5f;
            
            if (GradValid(Ix_l.at<float>(i,j)))
                ++success_num;

            new_disp.at<float>(i,j) = static_cast<int>(disp_float.at<float>(i,j));
            disp_float.at<float>(i,j) = static_cast<int>(disp_float.at<float>(i,j));
        }
    }

    // iterate on each pixel
    for (int i=half_win_size; i<img_h-half_win_size; ++i)
    {
        for (int j=half_win_size; j<img_w-half_win_size; ++j)
        {
            // printf("refine entry row %d, col %d\n", i, j);

            if (!GradValid(Ix_l.at<float>(i,j)))
            {
                // printf("invalid gradient\n");
                continue;
            }

            if (!DispValid(disp_float.at<float>(i,j)))
            {
                // printf("invalid initial disp\n");
                continue;
            }

            float last_disp = disp_float.at<float>(i,j);

            float last_doff = 0;
            float last_doff_diff = FLT_MAX;

            int iter_cnt = 0;
            while (iter_cnt < iter_num)
            {
                ++iter_cnt;

                // each iteration
                int win_cnt = 0;
                int valid_cnt = 0;
                Eigen::VectorXf win_Ix(win_size*win_size);
                Eigen::VectorXf win_Ires(win_size*win_size);
                Eigen::VectorXf win_weight(win_size*win_size);

                // windows aggregation
                for (int v=-half_win_size; v<=half_win_size; ++v)
                {
                    for (int u=-half_win_size; u<=half_win_size; ++u)
                    {
                        int m = i+v;  // row
                        int n = j+u;  // col

                        if (!GradValid(Ix_l.at<float>(m,n)))
                        {
                            win_weight(win_cnt++) = 0;
                            continue;
                        }

                        if (!DispValid(disp_float.at<float>(m,n)))
                        {
                            win_weight(win_cnt++) = 0;
                            continue;
                        }

                        if (fabs(disp_float.at<float>(i,j) - disp_float.at<float>(m,n)) > 2)
                        {
                            win_weight(win_cnt++) = 0;
                            continue;
                        }

                        float disp_win = disp_float.at<float>(m,n) + last_doff;
                        if (n-disp_win < 0 || n-disp_win > img_w-1)
                        {
                            win_weight(win_cnt++) = 0;
                            continue;
                        }

                        win_weight(win_cnt) = exp(-(v*v + u*u) / (2*half_win_size*half_win_size));
                        // win_weight(win_cnt) = 1;
                        win_Ires(win_cnt) = img_r.at<uchar>(m,n-disp_win) - img_l.at<uchar>(m,n);
                        // win_Ires(win_cnt) = GetInterpolatedValue(img_r, n-disp_win, m) - img_l.at<uchar>(m,n);
                        win_Ix(win_cnt) = Ix_l.at<float>(m,n);

                        ++win_cnt;
                        ++valid_cnt;
                    }
                }

                if (valid_cnt < win_size*win_size*0.1)
                {
                    printf("not enough valid cnt %d\n", valid_cnt);
                    break;
                }

                // weight normalization
                win_weight = win_weight / win_weight.norm();

                const Eigen::VectorXf &J = win_Ix;
                const Eigen::MatrixXf Weight = win_weight.asDiagonal();
                const Eigen::MatrixXf Hessian = J.transpose() * Weight * J;  // 1*1

                if (std::isnan(Hessian(0,0)) || Hessian(0,0)<1e-3)
                {
                    printf("Hessian not good %f\n", Hessian(0,0));
                    break;
                }

                // Gauss-Newton
                Eigen::MatrixXf doff = Hessian.inverse() * J.transpose() * Weight * win_Ires;
                float doff_try = doff(0, 0);

                if (std::isnan(doff_try))
                {
                    // printf("nan d_off\n");
                    break;
                }

                if (fabs(doff_try) > 1)
                {
                    printf("doff %f > 1, row %d, col %d, Hessian %f, grad %f, valid cnt %d\n", 
                            doff_try, i, j, Hessian(0,0), Ix_l.at<float>(i,j), valid_cnt);  // TODO: why?
                    // break;
                }

                if (fabs(doff_try - last_doff) > last_doff_diff)
                {
                    // printf("doff disconverged\n");
                    break;
                }

                if (!DispValid(disp_float.at<float>(i,j) + doff_try))
                {
                    // printf("disp outside\n");
                    break;
                }

                last_disp = disp_float.at<float>(i,j) + doff_try;

                last_doff_diff = fabs(doff_try - last_doff);
                last_doff = doff_try;
                // if (iter_cnt > 1)
                    // printf("iter %d, doff %f, diff %f\n", iter_cnt, last_doff, last_doff_diff);
                // else
                    // printf("iter %d, doff %f\n", iter_cnt, last_doff);
                
                if (last_doff_diff < 1e-6)
                {
                    // printf("early quit\n");
                    break;
                }
            }

            new_disp.at<float>(i,j) = last_disp;
            // printf("refine end row %d, col %d, doff %f\n", 
            //         i, j, new_disp.at<float>(i,j)-disp_float.at<float>(i,j));

            if (fabs(last_doff) > 0.1)
                ++process_num;
        }
    }

    disp_float = new_disp.clone();
    printf("refine num %d / %d\n", process_num, success_num);
}

static inline float PlaneDoff(const Eigen::Vector3f &abc,
                              int u, int v)
{
    return (abc(0)*u + abc(1)*v + abc(2));
}

/*
void LKSubPixelImpl::LKRefineCore(cv::Mat &disp_float) const
{
    // init 
    cv::Mat new_disp = disp_float.clone();

    // Ix_l * (au+bv+c) = I_r(u-dinit-(au+bv+c)) - I_l(u)
    // Ax = b

    // compute gradients on image_left
    int half_win_size = win_size / 2;
    cv::Mat Ix_l(img_h, img_w, CV_32FC1, cv::Scalar(0));
#pragma omp parallel for
    for (int i=img_h/2; i<img_h-half_win_size; ++i)
    {
        for (int j=half_win_size; j<img_w-half_win_size; ++j)
        {
            if (!DispValid(disp_float.at<float>(i,j)))
                continue;
            
            Ix_l.at<float>(i,j) = 
                (img_l.at<uchar>(i,j+1) - img_l.at<uchar>(i,j-1)) * 0.5f;
            
            disp_float.at<float>(i,j) = static_cast<int>(disp_float.at<float>(i,j));
            // new_disp.at<float>(i,j) = static_cast<int>(disp_float.at<float>(i,j));  // disable init guess by fitting
        }
    }

    // iterate on each pixel
    for (int i=img_h/2; i<img_h-half_win_size; ++i)
    {
        for (int j=half_win_size; j<img_w-half_win_size; ++j)
        {
            // printf("refine entry row %d, col %d\n", i, j);

            if (!GradValid(Ix_l.at<float>(i,j)))
            {
                // printf("invalid gradient\n");
                continue;
            }

            if (!DispValid(disp_float.at<float>(i,j)))
            {
                // printf("invalid initial disp\n");
                continue;
            }

            float last_disp = disp_float.at<float>(i,j);

            Eigen::Vector3f last_abc(0, 0, 0);
            float last_abc_diff = FLT_MAX;

            int iter_cnt = 0;
            while (iter_cnt < iter_num)
            {
                ++iter_cnt;

                // each iteration
                int valid_cnt = 0;
                Eigen::VectorXf win_Ix(win_size*win_size);
                Eigen::VectorXf win_uIx(win_size*win_size);
                Eigen::VectorXf win_vIx(win_size*win_size);
                Eigen::VectorXf win_Ires(win_size*win_size);

                // windows aggretation
                for (int v=-half_win_size; v<=half_win_size; ++v)
                {
                    for (int u=-half_win_size; u<=half_win_size; ++u)
                    {
                        int m = i+v;
                        int n = j+u;

                        if (!DispValid(disp_float.at<float>(m,n)))
                            continue;

                        if (fabs(disp_float.at<float>(i,j) - disp_float.at<float>(m,n)) > 2)
                            continue;

                        float disp_win = disp_float.at<float>(m,n) + PlaneDoff(last_abc, u, v);
                        if (n-disp_win < 0 || n-disp_win > img_w-1)  continue;

                        // // TODO
                        // double weight = exp(-(pow(v, 2) + pow(u, 2)) / (2*pow(half_win_size, 2)));
                        // weight = 1.0;

                        win_Ires(valid_cnt) = GetInterpolatedValue(img_r, n-disp_win, m) - img_l.at<uchar>(m,n);
                        win_Ix(valid_cnt) = Ix_l.at<float>(m,n);
                        win_uIx(valid_cnt) = u*Ix_l.at<float>(m,n);
                        win_vIx(valid_cnt) = v*Ix_l.at<float>(m,n);

                        valid_cnt++;
                    }
                }

                if (valid_cnt < win_size*win_size*0.1)
                {
                    // printf("not enough valid cnt (%d) in win size\n", valid_cnt);
                    break;
                }
                else
                {
                    win_Ix.resize(valid_cnt);
                    win_uIx.resize(valid_cnt);
                    win_vIx.resize(valid_cnt);
                    win_Ires.resize(valid_cnt);

                    Eigen::MatrixXf J = Eigen::MatrixXf::Zero(valid_cnt, 3);  // n*3
                    for (int k = 0; k < valid_cnt; k++) {
                        J(k, 0) = win_uIx(k);
                        J(k, 1) = win_vIx(k);
                        J(k, 2) = win_Ix(k);
                    }
                    const Eigen::MatrixXf Hessian = J.transpose() * J;  // 3*3

                    if (Hessian.trace() < 1e-3)
                    {
                        // printf("Hessian not good\n");
                        break;
                    }

                    // Gauss-Newton
                    Eigen::MatrixXf abc = Hessian.inverse() * J.transpose() * win_Ires;  // 3*1

                    if (std::isnan(abc(0)) || std::isnan(abc(1)) || std::isnan(abc(2)))
                    {
                        // printf("nan d_off\n");
                        break;
                    }

                    if (fabs(PlaneDoff(abc, 0, 0)) > 1)
                    {
                        printf("doff > 1\n");  // TODO: why?
                        break;
                    }

                    if (fabs(abc.norm() - last_abc.norm()) > last_abc_diff)
                    {
                        // printf("doff disconverged\n");
                        break;
                    }

                    if (!DispValid(disp_float.at<float>(i,j) + PlaneDoff(abc, 0, 0)))
                    {
                        // printf("disp outside\n");
                        break;
                    }

                    last_disp = disp_float.at<float>(i,j) + PlaneDoff(abc, 0, 0);

                    last_abc_diff = fabs(abc.norm() - last_abc.norm());
                    last_abc = abc;
                    if (iter_cnt > 1)
                        printf("iter %d, doff %f, diff %f\n", 
                                iter_cnt, PlaneDoff(last_abc, 0, 0), last_abc_diff);
                    else
                        printf("iter %d, doff %f\n", iter_cnt, PlaneDoff(last_abc, 0, 0));

                    if (last_abc_diff < 1e-6)
                    {
                        printf("early quit\n");
                        break;
                    }
                }
            }

            new_disp.at<float>(i,j) = last_disp;
            printf("refine end row %d, col %d, doff %f\n", 
                    i, j, new_disp.at<float>(i,j)-disp_float.at<float>(i,j));
        }
    }

    disp_float = new_disp.clone();
}
*/

uchar LKSubPixelImpl::GetInterpolatedValue(const cv::Mat &img,
                                           float u, float v) const
{
    // bilinear interpolation
    /*
    I1           I5      I2
    u0v0---------|----u1v0
    |            |       |
    |-----------u2v2-----|
    u0v1---------|----u1v1
    I3          I6      I4
    */

    int u0 = static_cast<int>(u);
    int u1 = u0 + 1;
    int v0 = static_cast<int>(v);
    // int v1 = v0 + 1;
    int v1 = v0;

    if (u1 > img.cols-1) u1 = img.cols-1;
    if (v1 > img.rows-1) u1 = img.rows-1;

    uchar I1 = img.at<uchar>(u0, v0);
    uchar I2 = img.at<uchar>(u1, v0);
    uchar I3 = img.at<uchar>(u0, v1);
    uchar I4 = img.at<uchar>(u1, v1);

    uchar I5, I6;
    if (u1 != u0) {
        I5 = I2 * (u-u0)/(u1-u0) + I1 * (u1-u)/(u1-u0);
        I6 = I4 * (u-u0)/(u1-u0) + I3 * (u1-u)/(u1-u0);
    }
    else {
        I5 = I1;
        I6 = I3;
    }
    
    uchar I7;
    if (v1 != v0) {
        I7 = I6 * (v-v0)/(v1-v0) + I5 * (v1-v)/(v1-v0);
    }
    else {
        I7 = I5;
    }

    return I7;
}