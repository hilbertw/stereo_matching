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
