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
    static __m128 perp = {-1, 1, -1, 1};
    static __m128 nnpp = {-1, -1, 1, 1};
    //cpBody *a = arb->body_a;
    //cpBody *b = arb->body_b;
    //__m128 friction = _mm_set_ss(arb->u);


    __m128 n_surface = _mm_load_ps(&(arb->n));

    __m128 n = _mm_movelh_ps(n_surface, n_surface);
    __m128 n_perp = _mm_mul_ps(_mm_shuffle_ps(n,n,_MM_SHUFFLE(2, 3, 0, 1)), perp);

    __m128 surface_vr = _mm_movehl_ps(_mm_setzero_ps(), n_surface);

    __m128 a_v_b = _mm_load_ps(&(arb->body_a->v));
    __m128 b_v_b = _mm_load_ps(&(arb->body_b->v));

    __m128 v = _mm_movelh_ps(a_v_b, b_v_b);
    __m128 v_bias = _mm_movehl_ps(b_v_b, a_v_b);

    __m128 a_param = _mm_load_ps(&(arb->body_a->m_inv));
    __m128 b_param = _mm_load_ps(&(arb->body_b->m_inv));

    __m128 inv = _mm_unpacklo_ps(a_param, b_param);
    __m128 m_inv = _mm_unpacklo_ps(inv, inv);
    __m128 i_inv = _mm_unpackhi_ps(inv, inv);

    __m128 w_b = _mm_unpackhi_ps(a_param, b_param);
    __m128 w = _mm_unpacklo_ps(w_b, w_b);
    __m128 w_bias = _mm_unpackhi_ps(w_b, w_b);

    int32_t i = arb->count;
    struct cpContact *con = arb->contacts;
    while(i--)
    {
        __m128 r = _mm_load_ps((float*)&(con->r1));
        __m128 r_perp = _mm_mul_ps(_mm_shuffle_ps(r,r,_MM_SHUFFLE(2, 3, 0, 1)), perp);

        __m128 vb = _mm_add_ps(v_bias, _mm_mul_ps(r_perp, w_bias));
        __m128 vr = _mm_add_ps(v, _mm_mul_ps(r_perp, w));
        __m128 vr_vb_a = _mm_movelh_ps(vr, vb);
        __m128 vr_vb_b = _mm_movehl_ps(vb, vr);
        __m128 vr_vb = _mm_sub_ps(vr_vb_b, vr_vb_a);
        //add surface_vr to vr
        vr_vb = _mm_add_ps(vr_vb, surface_vr);

        //vb and vr are vects
        __m128 vrvb_mul_n = _mm_mul_ps(vr_vb, n);
        //vr_vb dot n
        __m128 vrn_vbn = _mm_add_ps(vrvb_mul_n, _mm_shuffle_ps(vrvb_mul_n, vrvb_mul_n, _MM_SHUFFLE(2, 3, 0, 1)));

        //---------------------------------------------------------------------------------

        __m128 nnbb = _mm_load_ps(&(con->jnAcc));

        __m128 jnOld_jbnOld;
        jnOld_jbnOld = _mm_unpacklo_ps(nnbb, nnbb);

        __m128 bounce_bias;
        bounce_bias = _mm_unpackhi_ps(nnbb, nnbb);
        bounce_bias = _mm_mul_ps(bounce_bias, nnpp);

        __m128 nMass = _mm_load_ps1(&(con->nMass));
        //jbn and jn are scalars
        __m128 jn_jbn;
        jn_jbn = _mm_mul_ps(_mm_sub_ps(bounce_bias, vrn_vbn), nMass);
        jn_jbn = _mm_add_ps(jn_jbn, jnOld_jbnOld);
        jn_jbn = _mm_max_ps(jn_jbn, _mm_setzero_ps());

        __m128 jApply;
        jApply = _mm_sub_ps(jn_jbn, jnOld_jbnOld);

        _MM_STOREL_PI(&(con->jnAcc), _mm_shuffle_ps(jn_jbn, jn_jbn, _MM_SHUFFLE(3, 2, 2, 0)));

        //------------------------------------------------------------------------------------
        //vrt is scalar
        __m128 vrt = _mm_mul_ps(vr_vb, n_perp);
        vrt = _mm_add_ps(vrt, _mm_shuffle_ps(vrt, vrt, _MM_SHUFFLE(2, 3, 0, 1)));
        __m128 jtMax = _mm_mul_ss(*((__m128*)&(arb->u)), jn_jbn);
        __m128 jtOld = _mm_load_ss(&(con->jtAcc));
        //__m128 jt = _mm_mul_ss(vrt, _mm_shuffle_ps(jtOld, jtOld, 1));
        __m128 jt = _mm_mul_ss(vrt, *((__m128*)&(con->tMass)));
        __m128 jtAcc = _mm_min_ss(_mm_max_ss(_mm_sub_ss(jtOld, jt), _mm_mul_ss(jtMax, nnpp)), jtMax);
        _mm_store_ss(&(con->jtAcc), jtAcc);

        //------------------------------------------------------------------------------------
        //j >> - - + +
        __m128 j = _mm_mul_ps(n, _mm_mul_ps(_mm_movehl_ps(jApply,jApply),nnpp));

        v_bias = _mm_add_ps(v_bias, _mm_mul_ps(j, m_inv));

        __m128 rp_mul_j = _mm_mul_ps(r_perp, j);

        __m128 r_cross_j = _mm_add_ps(rp_mul_j, _mm_shuffle_ps(rp_mul_j, rp_mul_j, _MM_SHUFFLE(2, 3, 0, 1)));

        w_bias = _mm_add_ps(w_bias, _mm_mul_ps(i_inv, r_cross_j));

        //-----------------------------------------------------------------------------------
        __m128 rot_para = _mm_sub_ss(jtAcc, jtOld);
        rot_para = _mm_movelh_ps(rot_para, jApply);
        rot_para = _mm_shuffle_ps(rot_para, rot_para, _MM_SHUFFLE(0, 2, 2, 0));
        rot_para = _mm_mul_ss(rot_para, nnpp);

        __m128 n_mul_para = _mm_mul_ps(n_perp, rot_para);

        __m128 k = _mm_sub_ps(n_mul_para, _mm_shuffle_ps(n_mul_para, n_mul_para, _MM_SHUFFLE(2, 3, 0, 1)));

        k = _mm_shuffle_ps(k, k, _MM_SHUFFLE(3, 1, 2, 0));

        v = _mm_add_ps(v, _mm_mul_ps(k, m_inv));

        __m128 rp_mul_k = _mm_mul_ps(r_perp, k);

        __m128 r_cross_k = _mm_add_ps(rp_mul_k, _mm_shuffle_ps(rp_mul_k, rp_mul_k, _MM_SHUFFLE(2, 3, 0, 1)));

        w = _mm_add_ps(w, _mm_mul_ps(i_inv, r_cross_k));
        con++;
    }

    a_v_b = _mm_movelh_ps(v, v_bias);
    b_v_b = _mm_movehl_ps(v_bias, v);

    _mm_store_ps(&(arb->body_a->v), a_v_b);
    _mm_store_ps(&(arb->body_b->v), b_v_b);

    _MM_STOREL_PI(&(arb->body_a->w), _mm_unpacklo_ps(w, w_bias));
    _MM_STOREL_PI(&(arb->body_b->w), _mm_unpackhi_ps(w, w_bias));
}
#endif
