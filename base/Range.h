#ifndef RANGE_H
#define RANGE_H
#include "Setting.h"

struct Range {
	INT r, t;
	float pro;
	// static bool cmp_rt(const Range &a, const Range &b) {
	// 	return (a.r < b.r)||(a.r == b.r && a.t < b.t);
	// }
	static bool cmp_tr(const Range &a, const Range &b) {
		return (a.t < b.t)||(a.t == b.t && a.r < b.r)||(a.t == b.t && a.r == b.r && a.pro < b.pro);
	}
};
#endif