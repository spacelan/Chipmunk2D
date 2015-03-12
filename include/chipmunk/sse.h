#ifndef SSE_H
#define SSE_H
#ifdef __SSE__
#include "chipmunk/chipmunk_private.h"
#include "chipmunk/chipmunk.h"
#include <xmmintrin.h>
void cpArbiterApplyImpulse_SSE(cpArbiter *arb);
#endif
#endif
