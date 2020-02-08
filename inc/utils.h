#pragma once

#include "global.h"


double get_cur_ms();

std::string num2str(int i);

std::string num2strbeta(int i);

void stereo_record(int camid, std::string address);

struct CamIntrinsics
{
    float fx;
    float fy;
    float cx;
    float cy;
};

CamIntrinsics read_calib(std::string file_addr);
