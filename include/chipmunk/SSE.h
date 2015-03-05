#define __SSE__
#ifndef SSE_H
#define SSE_H
#include "chipmunk\chipmunk_private.h"
#include "chipmunk\chipmunk.h"
#ifdef __SSE__
#include <xmmintrin.h>

typedef struct{
	cpFloat x, y;
}vect32_t;

typedef union{
	__m128 m;
	cpFloat a[4];
}float32x4_t;

typedef union{
	cpVect vect[2];
	__m128 mm;
	cpFloat arr[4];
}vect32x2_t;

void cpArbiterApplyImpulse_SSE(cpArbiter *arb);
#endif
#endif
