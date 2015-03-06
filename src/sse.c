#include "chipmunk/sse.h"
#include <xmmintrin.h>
//#ifdef __SSE__

#define _MM_LOADL_PI(x,y) (_mm_loadl_pi((x), (__m64 const*)(y)))
#define _MM_LOADH_PI(x,y) (_mm_loadh_pi((x), (__m64 const*)(y)))
#define _MM_STOREL_PI(x,y) (_mm_storel_pi((__m64*)(x), (y)))
#define _MM_STOREH_PI(x,y) (_mm_storeh_pi((__m64*)(x), (y)))

void cpArbiterApplyImpulse_SSE(cpArbiter *arb)
{
	cpBody *a = arb->body_a;
    cpBody *b = arb->body_b;
    cpFloat friction = arb->u;
	vect32x2_t n;// = { arb->n.x, arb->n.y, arb->n.x, arb->n.y };
    n.mm = _MM_LOADL_PI(n.mm, &arb->n);
    n.mm = _MM_LOADH_PI(n.mm, &arb->n);

	vect32x2_t surface_vr;// = { 0, 0, arb->surface_vr.x, arb->surface_vr.y };
	surface_vr.mm = _MM_LOADH_PI(_mm_setzero_ps(), &arb->surface_vr);

	vect32x2_t v_bias;// = {a->v_bias.x, a->v_bias.y, b->v_bias.x, b->v_bias.y};
	v_bias.mm = _MM_LOADL_PI(v_bias.mm, &a->v_bias);
	v_bias.mm = _MM_LOADH_PI(v_bias.mm, &b->v_bias);

	vect32x2_t w_bias = { a->w_bias, a->w_bias,
		                  b->w_bias, b->w_bias };

	vect32x2_t v;// = { a->v.x, a->v.y, b->v.x, b->v.y };
	v.mm = _MM_LOADL_PI(v.mm, &a->v);
	v.mm = _MM_LOADH_PI(v.mm, &b->v);

	vect32x2_t w = { a->w, a->w,
		             b->w, b->w };

	vect32x2_t m_inv = { a->m_inv, a->m_inv,
		                 b->m_inv, b->m_inv };

	vect32x2_t i_inv = { a->i_inv, a->i_inv,
		                 b->i_inv, b->i_inv };

	for (int i = 0; i < arb->count; i++){
		struct cpContact *con = &arb->contacts[i];

		vect32x2_t r_perp = { -con->r1.y, con->r1.x,
			                  -con->r2.y, con->r2.x };

		vect32x2_t rp_mul_wb;
		rp_mul_wb.mm = _mm_mul_ps(r_perp.mm, w_bias.mm);

		vect32x2_t vb;
		vb.mm = _mm_add_ps(v_bias.mm, rp_mul_wb.mm);

		vect32x2_t rp_mul_w;
		rp_mul_w.mm = _mm_mul_ps(r_perp.mm, w.mm);

		vect32x2_t vr;
		vr.mm = _mm_add_ps(v.mm, rp_mul_w.mm);

		vect32x2_t vb_vr = { vb.vect[1].x - vb.vect[0].x, vb.vect[1].y - vb.vect[0].y,
			vr.vect[1].x - vr.vect[0].x, vr.vect[1].y - vr.vect[0].y };
		//add surface_vr to vr
		vb_vr.mm = _mm_add_ps(vb_vr.mm, surface_vr.mm);

		//vb and vr are vects
		vect32x2_t vbvr_mul_n;
		vbvr_mul_n.mm = _mm_mul_ps(vb_vr.mm, n.mm);

		//vbn and vrn are scalars
		vect32x2_t vbn_vrn = {vbvr_mul_n.vect[0].x + vbvr_mul_n.vect[0].y,
                              vbvr_mul_n.vect[1].x + vbvr_mul_n.vect[1].y};

        //---------------------------------------------------------------------------------
		vect32x2_t nMass = {con->nMass, con->nMass, con->nMass, con->nMass};

		vect32x2_t bias_bounce = { con->bias, -con->bounce};

        //vect[0] only. jbn and jn are scalars
        vect32x2_t jbn_jn;
        jbn_jn.mm = _mm_mul_ps(_mm_sub_ps(bias_bounce.mm, vbn_vrn.mm), nMass.mm);

		vect32x2_t jbnOld_jnOld = { con->jBias, con->jnAcc};

        jbn_jn.mm = _mm_max_ps(_mm_add_ps(jbn_jn.mm, jbnOld_jnOld.mm), _mm_setzero_ps());
        //apply to con
        con->jBias = jbn_jn.arr[0];
        con->jnAcc = jbn_jn.arr[1];

        //vect[0] only
        vect32x2_t jApply;
        jApply.mm = _mm_sub_ps(jbn_jn.mm, jbnOld_jnOld.mm);

        //------------------------------------------------------------------------------------
        //vrt is scalar
        float vrt = -vb_vr.vect[1].x*n.vect[0].y + vb_vr.vect[1].y*n.vect[0].x;
        float jtMax = friction*con->jnAcc;
        float jt = -vrt*con->tMass;
        float jtOld = con->jtAcc;
        con->jtAcc = cpfclamp(jtOld+jt, -jtMax, jtMax);

        //------------------------------------------------------------------------------------
        //j==>{vect[0]:-j,vect[1]:j}
        vect32x2_t j;
        j.mm = _mm_mul_ps(n.mm, _mm_setr_ps(-jApply.arr[0], -jApply.arr[0],
                                            jApply.arr[0], jApply.arr[0]));

        v_bias.mm = _mm_add_ps(v_bias.mm, _mm_mul_ps(j.mm, m_inv.mm));

        vect32x2_t rp_mul_j;
        rp_mul_j.mm = _mm_mul_ps(r_perp.mm, j.mm);

		vect32x2_t r_cross_j = { rp_mul_j.arr[0] + rp_mul_j.arr[1], rp_mul_j.arr[0] + rp_mul_j.arr[1],
			                     rp_mul_j.arr[2] + rp_mul_j.arr[3], rp_mul_j.arr[2] + rp_mul_j.arr[3] };

        w_bias.mm = _mm_add_ps(w_bias.mm, _mm_mul_ps(i_inv.mm, r_cross_j.mm));

		vect32x2_t rot_para = { jApply.arr[1], con->jtAcc - jtOld,
			                    con->jtAcc - jtOld, jApply.arr[1] };

        vect32x2_t n_mul_para;
        n_mul_para.mm = _mm_mul_ps(n.mm, rot_para.mm);

		vect32x2_t k = { -(n_mul_para.arr[0] - n_mul_para.arr[1]), -(n_mul_para.arr[2] + n_mul_para.arr[3]),
			             n_mul_para.arr[0] - n_mul_para.arr[1], n_mul_para.arr[2] + n_mul_para.arr[3] };

        v.mm = _mm_add_ps(v.mm, _mm_mul_ps(k.mm, m_inv.mm));

        vect32x2_t rp_mul_k;
        rp_mul_k.mm = _mm_mul_ps(r_perp.mm, k.mm);

		vect32x2_t r_cross_k = { rp_mul_k.arr[0] + rp_mul_k.arr[1], rp_mul_k.arr[0] + rp_mul_k.arr[1],
			                     rp_mul_k.arr[2] + rp_mul_k.arr[3], rp_mul_k.arr[2] + rp_mul_k.arr[3] };

        w.mm = _mm_add_ps(w.mm, _mm_mul_ps(i_inv.mm, r_cross_k.mm));
    }
    _MM_STOREL_PI(&a->v_bias, v_bias.mm);//a->v_bias = v_bias.vect[0];
    _MM_STOREH_PI(&b->v_bias, v_bias.mm);//b->v_bias = v_bias.vect[1];
    a->w_bias = w_bias.arr[0];
    b->w_bias = w_bias.arr[2];
    _MM_STOREL_PI(&a->v, v.mm);//a->v = v.vect[0];
    _MM_STOREH_PI(&b->v, v.mm);//b->v = v.vect[1];
    a->w = w.arr[0];
    b->w = w.arr[2];
}
//#endif
