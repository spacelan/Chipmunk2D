#include "chipmunk/sse.h"
#include <xmmintrin.h>
#ifdef __SSE__

#define _MM_LOADL_PI(x,y) (_mm_loadl_pi((x), (__m64 const*)(y)))
#define _MM_LOADH_PI(x,y) (_mm_loadh_pi((x), (__m64 const*)(y)))
#define _MM_STOREL_PI(x,y) (_mm_storel_pi((__m64*)(x), (y)))
#define _MM_STOREH_PI(x,y) (_mm_storeh_pi((__m64*)(x), (y)))
#define _MM_GET_LANE(x,y) (((float const*)&x)[y])
void cpArbiterApplyImpulse_SSE(cpArbiter *arb)
{
	cpBody *a = arb->body_a;
    cpBody *b = arb->body_b;
    cpFloat friction = arb->u;
	__m128 n;// = { arb->n.x, arb->n.y, arb->n.x, arb->n.y };
    n = _MM_LOADL_PI(n, &arb->n);
    n = _MM_LOADH_PI(n, &arb->n);

	__m128 surface_vr;// = { 0, 0, arb->surface_vr.x, arb->surface_vr.y };
	surface_vr = _MM_LOADH_PI(_mm_setzero_ps(), &arb->surface_vr);

	__m128 v_bias;// = {a->v_bias.x, a->v_bias.y, b->v_bias.x, b->v_bias.y};
	v_bias = _MM_LOADL_PI(v_bias, &a->v_bias);
	v_bias = _MM_LOADH_PI(v_bias, &b->v_bias);

	__m128 w_bias = _mm_setr_ps( a->w_bias, a->w_bias,
		                  b->w_bias, b->w_bias );

	__m128 v;// = { a->v.x, a->v.y, b->v.x, b->v.y };
	v = _MM_LOADL_PI(v, &a->v);
	v = _MM_LOADH_PI(v, &b->v);

	__m128 w = _mm_setr_ps( a->w, a->w,
		             b->w, b->w );

	__m128 m_inv = _mm_setr_ps( a->m_inv, a->m_inv,
		                 b->m_inv, b->m_inv );

	__m128 i_inv = _mm_setr_ps( a->i_inv, a->i_inv,
		                 b->i_inv, b->i_inv );

	for (int i = 0; i < arb->count; i++){
		struct cpContact *con = &arb->contacts[i];

		__m128 r_perp = _mm_setr_ps( -con->r1.y, con->r1.x,
			                  -con->r2.y, con->r2.x );

		__m128 rp_mul_wb;
		rp_mul_wb = _mm_mul_ps(r_perp, w_bias);

		__m128 vb;
		vb = _mm_add_ps(v_bias, rp_mul_wb);

		__m128 rp_mul_w;
		rp_mul_w = _mm_mul_ps(r_perp, w);

		__m128 vr;
		vr = _mm_add_ps(v, rp_mul_w);

		//__m128 vb_vr = { vb.vect[1].x - vb.vect[0].x, vb.vect[1].y - vb.vect[0].y,vr.vect[1].x - vr.vect[0].x, vr.vect[1].y - vr.vect[0].y );

		__m128 vb_vr = _mm_setr_ps( _MM_GET_LANE(vb, 2) - _MM_GET_LANE(vb, 0), _MM_GET_LANE(vb, 3) - _MM_GET_LANE(vb, 1),
		                 _MM_GET_LANE(vr, 2) - _MM_GET_LANE(vr, 0), _MM_GET_LANE(vr, 3) - _MM_GET_LANE(vr, 1));

		//add surface_vr to vr
		vb_vr = _mm_add_ps(vb_vr, surface_vr);

		//vb and vr are vects
		__m128 vbvr_mul_n;
		vbvr_mul_n = _mm_mul_ps(vb_vr, n);

		//vbn and vrn are scalars
		__m128 vbn_vrn = _mm_setr_ps(_MM_GET_LANE(vbvr_mul_n, 0) + _MM_GET_LANE(vbvr_mul_n, 1),
		                  _MM_GET_LANE(vbvr_mul_n, 2) + _MM_GET_LANE(vbvr_mul_n, 3),0,0);

        //---------------------------------------------------------------------------------
		__m128 nMass = _mm_set_ps1(con->nMass);

		__m128 bias_bounce = _mm_setr_ps( con->bias, -con->bounce,0,0);

        //vect[0] only. jbn and jn are scalars
        __m128 jbn_jn;
        jbn_jn = _mm_mul_ps(_mm_sub_ps(bias_bounce, vbn_vrn), nMass);

		__m128 jbnOld_jnOld = _mm_setr_ps( con->jBias, con->jnAcc,0,0);

        jbn_jn = _mm_max_ps(_mm_add_ps(jbn_jn, jbnOld_jnOld), _mm_setzero_ps());
        //apply to con
        con->jBias = _MM_GET_LANE(jbn_jn,0);//jbn_jn.arr[0];
        con->jnAcc = _MM_GET_LANE(jbn_jn,1);//jbn_jn.arr[1];

        //vect[0] only
        __m128 jApply;
        jApply = _mm_sub_ps(jbn_jn, jbnOld_jnOld);

        //------------------------------------------------------------------------------------
        //vrt is scalar
        float vrt = -_MM_GET_LANE(vb_vr,2)*_MM_GET_LANE(n,1) + _MM_GET_LANE(vb_vr,3)*_MM_GET_LANE(n,0);
        float jtMax = friction*con->jnAcc;
        float jt = -vrt*con->tMass;
        float jtOld = con->jtAcc;
        con->jtAcc = cpfclamp(jtOld+jt, -jtMax, jtMax);

        //------------------------------------------------------------------------------------
        //j==>{vect[0]:-j,vect[1]:j}
        __m128 j;
        j = _mm_mul_ps(n, _mm_setr_ps(-_MM_GET_LANE(jApply,0), -_MM_GET_LANE(jApply,0),
                                      _MM_GET_LANE(jApply,0), _MM_GET_LANE(jApply,0)));

        v_bias = _mm_add_ps(v_bias, _mm_mul_ps(j, m_inv));

        __m128 rp_mul_j;
        rp_mul_j = _mm_mul_ps(r_perp, j);

		__m128 r_cross_j = _mm_setr_ps( _MM_GET_LANE(rp_mul_j,0) + _MM_GET_LANE(rp_mul_j,1), _MM_GET_LANE(rp_mul_j,0) + _MM_GET_LANE(rp_mul_j,1),
                             _MM_GET_LANE(rp_mul_j,2) + _MM_GET_LANE(rp_mul_j,3), _MM_GET_LANE(rp_mul_j,2) + _MM_GET_LANE(rp_mul_j,3) );

        w_bias = _mm_add_ps(w_bias, _mm_mul_ps(i_inv, r_cross_j));

		__m128 rot_para = _mm_setr_ps( _MM_GET_LANE(jApply,1), con->jtAcc - jtOld,
                            con->jtAcc - jtOld, _MM_GET_LANE(jApply,1) );

        __m128 n_mul_para;
        n_mul_para = _mm_mul_ps(n, rot_para);

		__m128 k = _mm_setr_ps( -(_MM_GET_LANE(n_mul_para,0) - _MM_GET_LANE(n_mul_para,1)), -(_MM_GET_LANE(n_mul_para,2) + _MM_GET_LANE(n_mul_para,3)),
                     _MM_GET_LANE(n_mul_para,0) - _MM_GET_LANE(n_mul_para,1), _MM_GET_LANE(n_mul_para,2) + _MM_GET_LANE(n_mul_para,3) );

        v = _mm_add_ps(v, _mm_mul_ps(k, m_inv));

        __m128 rp_mul_k;
        rp_mul_k = _mm_mul_ps(r_perp, k);

		__m128 r_cross_k = _mm_setr_ps( _MM_GET_LANE(rp_mul_k,0) + _MM_GET_LANE(rp_mul_k,1), _MM_GET_LANE(rp_mul_k,0) + _MM_GET_LANE(rp_mul_k,1),
			                     _MM_GET_LANE(rp_mul_k,2) + _MM_GET_LANE(rp_mul_k,3), _MM_GET_LANE(rp_mul_k,2) + _MM_GET_LANE(rp_mul_k,3) );

        w = _mm_add_ps(w, _mm_mul_ps(i_inv, r_cross_k));
    }
    _MM_STOREL_PI(&a->v_bias, v_bias);//a->v_bias = v_bias.vect[0];
    _MM_STOREH_PI(&b->v_bias, v_bias);//b->v_bias = v_bias.vect[1];
    a->w_bias = _MM_GET_LANE(w_bias,0);
    b->w_bias = _MM_GET_LANE(w_bias,2);
    _MM_STOREL_PI(&a->v, v);//a->v = v.vect[0];
    _MM_STOREH_PI(&b->v, v);//b->v = v.vect[1];
    a->w = _MM_GET_LANE(w,0);
    b->w = _MM_GET_LANE(w,2);
}
#endif
