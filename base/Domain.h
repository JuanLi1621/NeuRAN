#ifndef DOMAIN_H
#define DOMAIN_H
#include "Setting.h"

struct Domain {
	INT h, r;
	float pro;
	// static bool cmp_rh(const Domain &a, const Domain &b) {
	// 	return (a.r < b.r)||(a.r == b.r && a.h < b.h);
	// }
	static bool cmp_hr(const Domain &a, const Domain &b) {
		return (a.h < b.h)||(a.h == b.h && a.r < b.r)||(a.h == b.h && a.r == b.r && a.pro < b.pro);
	}
};
#endif