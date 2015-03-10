#include "chipmunk/sse.h"
#include <xmmintrin.h>
#define __SSE__
#ifdef __SSE__

#define _MM_LOADL_PI(x,y) (_mm_loadl_pi((x), (__m64 const*)(y)))
#define _MM_LOADH_PI(x,y) (_mm_loadh_pi((x), (__m64 const*)(y)))
#define _MM_STOREL_PI(x,y) (_mm_storel_pi((__m64*)(x), (y)))
#define _MM_STOREH_PI(x,y) (_mm_storeh_pi((__m64*)(x), (y)))
#define _MM_GET_LANE(x,y) (_mm_cvtss_f32(_mm_shuffle_ps(x,x,y)))
void cpArbiterApplyImpulse_SSE(cpArbiter *arb)
{
	cpBody *a = arb->body_a;
    cpBody *b = arb->body_b;
    cpFloat friction = arb->u;
	__m128 n;// = { arb->n.x, arb->n.y, arb->n.x, arb->n.y };
    n = _MM_LOADL_PI(n, &arb->n);
	n = _mm_movelh_ps(n, n);

	__m128 surface_vr;// = { 0, 0, arb->surface_vr.x, arb->surface_vr.y };
	surface_vr = _MM_LOADH_PI(_mm_setzero_ps(), &arb->surface_vr);

	__m128 v_bias;// = {a->v_bias.x, a->v_bias.y, b->v_bias.x, b->v_bias.y};
	v_bias = _MM_LOADL_PI(v_bias, &a->v_bias);
	v_bias = _MM_LOADH_PI(v_bias, &b->v_bias);
/*
	__m128 w_bias = _mm_setr_ps( a->w_bias, a->w_bias,
		                  b->w_bias, b->w_bias );
*/
    __m128 w_bias = _mm_movelh_ps(_mm_set_ps1(a->w_bias), _mm_set_ps1(b->w_bias));

	__m128 v;// = { a->v.x, a->v.y, b->v.x, b->v.y };
	v = _MM_LOADL_PI(v, &a->v);
	v = _MM_LOADH_PI(v, &b->v);
/*
	__m128 w = _mm_setr_ps( a->w, a->w,
		             b->w, b->w );
*/
    __m128 w = _mm_movelh_ps(_mm_set_ps1(a->w), _mm_set_ps1(b->w));
/*
	__m128 m_inv = _mm_setr_ps( a->m_inv, a->m_inv,
		                 b->m_inv, b->m_inv );
*/
    __m128 m_inv = _mm_movelh_ps(_mm_set_ps1(a->m_inv), _mm_set_ps1(b->m_inv));
/*
	__m128 i_inv = _mm_setr_ps( a->i_inv, a->i_inv,
		                 b->i_inv, b->i_inv );
*/
    __m128 i_inv = _mm_movelh_ps(_mm_set_ps1(a->i_inv), _mm_set_ps1(b->i_inv));

    static __m128 perp = {-1, 1, -1, 1};

    int i = arb->count;
	struct cpContact *con = arb->contacts;
	while(i--){
		__m128 r = _mm_load_ps(&(con->r1));
		//r = _MM_LOADL_PI(r,&con->r1);
		//r = _MM_LOADH_PI(r,&con->r2);

		__m128 r_perp = _mm_mul_ps(_mm_shuffle_ps(r,r,_MM_SHUFFLE(2, 3, 0, 1)), perp);

		__m128 vb = _mm_add_ps(v_bias, _mm_mul_ps(r_perp, w_bias));

		__m128 vr = _mm_add_ps(v, _mm_mul_ps(r_perp, w));

		__m128 vb_vr_a = _mm_movelh_ps(vb, vr);

		__m128 vb_vr_b = _mm_movehl_ps(vr, vb);

		__m128 vb_vr = _mm_sub_ps(vb_vr_b, vb_vr_a);

		//add surface_vr to vr
		vb_vr = _mm_add_ps(vb_vr, surface_vr);

		//vb and vr are vects
		__m128 vbvr_mul_n = _mm_mul_ps(vb_vr, n);

		__m128 vbn_vrn = _mm_add_ps(vbvr_mul_n, _mm_shuffle_ps(vbvr_mul_n, vbvr_mul_n, _MM_SHUFFLE(2, 3, 0, 1)));

        //---------------------------------------------------------------------------------
		__m128 nMass = _mm_set_ps1(con->nMass);

		//__m128 bias_bounce = _mm_movelh_ps(_mm_set_ps1(con->bias), _mm_set_ps1(-con->bounce));
		__m128 bias_bounce;
		bias_bounce = _MM_LOADL_PI(bias_bounce, &(con->bias));
		bias_bounce = _mm_unpacklo_ps(bias_bounce, bias_bounce);
		bias_bounce = _mm_mul_ps(bias_bounce, _mm_shuffle_ps(perp, perp, _MM_SHUFFLE(0, 2, 1, 3)));

        //vect[0] only. jbn and jn are scalars
        __m128 jbn_jn;
        jbn_jn = _mm_mul_ps(_mm_sub_ps(bias_bounce, vbn_vrn), nMass);

		//__m128 jbnOld_jnOld = _mm_movelh_ps(_mm_set_ps1(con->jBias), _mm_set_ps1(con->jnAcc));
		__m128 jbnOld_jnOld;
		jbnOld_jnOld = _MM_LOADL_PI(jbnOld_jnOld, &(con->jBias));
		jbnOld_jnOld = _mm_unpacklo_ps(jbnOld_jnOld, jbnOld_jnOld);

        jbn_jn = _mm_max_ps(_mm_add_ps(jbn_jn, jbnOld_jnOld), _mm_setzero_ps());

        //vect[0] only
        __m128 jApply;
        jApply = _mm_sub_ps(jbn_jn, jbnOld_jnOld);

        //apply to con
        //con->jBias = _MM_GET_LANE(jbn_jn,0);//jbn_jn.arr[0];
        //con->jnAcc = _MM_GET_LANE(jbn_jn,2);//jbn_jn.arr[1];
        _MM_STOREL_PI(&(con->jBias), _mm_shuffle_ps(jbn_jn, jbn_jn, _MM_SHUFFLE(3, 2, 2, 0)));

        //------------------------------------------------------------------------------------
        //vrt is scalar
		__m128 _vrt = _mm_mul_ps(n, _mm_shuffle_ps(vb_vr, vb_vr, _MM_SHUFFLE(3, 2, 2, 3)));
		_vrt = _mm_sub_ss(_mm_shuffle_ps(_vrt, _vrt, _MM_SHUFFLE(3, 2, 0, 1)), _vrt);
		float vrt = _mm_cvtss_f32(_vrt);
        float jtMax = friction*con->jnAcc;
        float jt = vrt*con->tMass;
        float jtOld = con->jtAcc;
        con->jtAcc = cpfclamp(jtOld+jt, -jtMax, jtMax);

        //------------------------------------------------------------------------------------
        //j==>{vect[0]:-j,vect[1]:j}
        __m128 j = _mm_mul_ps(n, _mm_mul_ps(_mm_movelh_ps(jApply,jApply),_mm_shuffle_ps(perp,perp,_MM_SHUFFLE(3, 1, 2, 0))));

        v_bias = _mm_add_ps(v_bias, _mm_mul_ps(j, m_inv));

        __m128 rp_mul_j = _mm_mul_ps(r_perp, j);

        __m128 r_cross_j = _mm_add_ps(rp_mul_j, _mm_shuffle_ps(rp_mul_j, rp_mul_j, _MM_SHUFFLE(2, 3, 0, 1)));

        w_bias = _mm_add_ps(w_bias, _mm_mul_ps(i_inv, r_cross_j));
        //------------------------------------------------------------------------------------
		__m128 rot_para = _mm_setr_ps( _MM_GET_LANE(jApply,2), -(con->jtAcc - jtOld),
                            con->jtAcc - jtOld, _MM_GET_LANE(jApply,2) );

        __m128 n_mul_para = _mm_mul_ps(n, rot_para);

        __m128 k = _mm_add_ps(n_mul_para, _mm_shuffle_ps(n_mul_para, n_mul_para, _MM_SHUFFLE(2, 3, 0, 1)));

        k = _mm_mul_ps(k, perp);

        k = _mm_shuffle_ps(k, k, _MM_SHUFFLE(3, 1, 2, 0));

        v = _mm_add_ps(v, _mm_mul_ps(k, m_inv));

        __m128 rp_mul_k = _mm_mul_ps(r_perp, k);

        __m128 r_cross_k = _mm_add_ps(rp_mul_k, _mm_shuffle_ps(rp_mul_k, rp_mul_k, _MM_SHUFFLE(2, 3, 0, 1)));

        w = _mm_add_ps(w, _mm_mul_ps(i_inv, r_cross_k));
        con++;
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
