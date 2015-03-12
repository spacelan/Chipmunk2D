#ifdef __SSE__
#include "chipmunk/sse.h"

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

    __m128 n;
    n = _MM_LOADL_PI(n, &arb->n);
    n = _mm_movelh_ps(n, n);

    __m128 surface_vr;
    surface_vr = _MM_LOADH_PI(_mm_setzero_ps(), &arb->surface_vr);

    __m128 v_bias;
    v_bias = _MM_LOADL_PI(v_bias, &a->v_bias);
    v_bias = _MM_LOADH_PI(v_bias, &b->v_bias);

    __m128 v;
    v = _MM_LOADL_PI(v, &a->v);
    v = _MM_LOADH_PI(v, &b->v);

    __m128 a_param = _mm_load_ps(&(a->m_inv));
    __m128 b_param = _mm_load_ps(&(b->m_inv));

    __m128 inv = _mm_unpacklo_ps(a_param, b_param);
    __m128 m_inv = _mm_unpacklo_ps(inv, inv);
    __m128 i_inv = _mm_unpackhi_ps(inv, inv);

    __m128 w_b = _mm_unpackhi_ps(a_param, b_param);
    __m128 w = _mm_unpacklo_ps(w_b, w_b);
    __m128 w_bias = _mm_unpackhi_ps(w_b, w_b);

    static __m128 perp = {-1, 1, -1, 1};

    int32_t i = arb->count;
    struct cpContact *con = arb->contacts;
    while(i--){
        __m128 r = _mm_load_ps((float*)&(con->r1));
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
        //vb_vr dot n
        __m128 vbn_vrn = _mm_add_ps(vbvr_mul_n, _mm_shuffle_ps(vbvr_mul_n, vbvr_mul_n, _MM_SHUFFLE(2, 3, 0, 1)));

        //---------------------------------------------------------------------------------

        __m128 jjbb = _mm_load_ps(&(con->jBias));

        __m128 jbnOld_jnOld;
        jbnOld_jnOld = _mm_unpacklo_ps(jjbb, jjbb);

        __m128 bias_bounce;
        bias_bounce = _mm_unpackhi_ps(jjbb, jjbb);
        bias_bounce = _mm_mul_ps(bias_bounce, _mm_shuffle_ps(perp, perp, _MM_SHUFFLE(0, 2, 1, 3)));

        __m128 nMass = _mm_load_ps1(&(con->nMass));
        //jbn and jn are scalars
        __m128 jbn_jn;
        jbn_jn = _mm_mul_ps(_mm_sub_ps(bias_bounce, vbn_vrn), nMass);
        jbn_jn = _mm_max_ps(_mm_add_ps(jbn_jn, jbnOld_jnOld), _mm_setzero_ps());

        __m128 jApply;
        jApply = _mm_sub_ps(jbn_jn, jbnOld_jnOld);

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
        //j >> - - + +
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
    _MM_STOREL_PI(&a->v_bias, v_bias);
    _MM_STOREH_PI(&b->v_bias, v_bias);

    _MM_STOREL_PI(&a->v, v);
    _MM_STOREH_PI(&b->v, v);

    _MM_STOREL_PI(&(a->w), _mm_unpacklo_ps(w, w_bias));
    _MM_STOREL_PI(&(b->w), _mm_unpackhi_ps(w, w_bias));
}
#endif
