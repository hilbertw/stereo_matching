#include "../cpu_inc/utils.h"


double get_cur_ms()
{
	return getTickCount() * 1000.f / getTickFrequency();
}