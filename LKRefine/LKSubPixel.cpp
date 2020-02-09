#include "LKSubPixel.h"
#include "LKSubPixelImpl.h"

LKSubPixel::LKSubPixel(int h, int w, int s, int d)
{
    assert (h>0 && w>0 && s>0 && d>0);

    assert (s==1 || s==2);

    assert (d==32 || d==64 || d==128);
}

LKSubPixel::~LKSubPixel() {}

std::shared_ptr<LKSubPixel> LKSubPixel::create(int h, int w, int s, int d)
{
    return static_cast<std::shared_ptr<LKSubPixel>>(new LKSubPixelImpl(h, w, s, d));
}