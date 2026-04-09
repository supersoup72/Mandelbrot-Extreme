/*
 * mandelbrot_core.c — ARM64 NEON-accelerated Mandelbrot pixel engine.
 *
 * Compiled with: -O3 -march=native -ffast-math -funroll-loops
 *
 * SIMD design (ARM64 NEON — the ARM equivalent of x86 AVX2):
 * ─────────────────────────────────────────────────────────────
 * ARM64 NEON provides 128-bit registers holding 2× float64 (float64x2_t).
 * AVX2 on x86 holds 4× float64 (256-bit); NEON holds 2× (128-bit).
 * The technique is identical — process multiple pixels in lockstep:
 *
 *   for each iteration:
 *     compute z² + c for ALL lanes simultaneously (NEON vmul/vadd/vsub)
 *     check escape condition for ALL lanes (vcgtq_f64 → bitmask)
 *     add 1 to iter count only for lanes that have NOT yet escaped
 *       (masked increment via vandq + vaddq)
 *     continue until ALL lanes escaped or hit max_iter
 *
 * This gives 2× pixel throughput on the iteration loop, and ARM64's
 * out-of-order execution can overlap operations across the two lanes.
 *
 * Key NEON intrinsics used:
 *   vdupq_n_f64(x)       — broadcast scalar to both lanes
 *   vld1q_f64(ptr)       — load 2 doubles from memory
 *   vmulq_f64(a,b)       — a[0]*b[0], a[1]*b[1]
 *   vfmaq_f64(c,a,b)     — c + a*b (FMA, 1 instruction)
 *   vsubq_f64(a,b)       — subtraction
 *   vaddq_f64(a,b)       — addition
 *   vcgtq_f64(a,b)       — compare >  → uint64x2 mask (all-1s or all-0s per lane)
 *   vandq_u64(a,b)       — bitwise AND (used for masked increment)
 *   vaddvq_u64(a)        — horizontal add (check if any lane still active)
 *   vgetq_lane_f64(v,i)  — extract single lane
 *
 * Performance expectation on Cortex-A55 (budget Android):  ~1.6× vs scalar
 * Performance expectation on Cortex-A75/A78 (flagship):    ~1.9× vs scalar
 * Combined with x4-row interleaving: effective 3-4× vs original single-row.
 */

#include "mandelbrot_core.h"
#include <math.h>
#include <string.h>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

/* ═══════════════════════════════════════════════════════════════════════════
 *  SINGLE-PIXEL STANDARD PATH  (Mariani-Silver border pixels)
 * ═══════════════════════════════════════════════════════════════════════════ */

double mb_pixel_std(double px, double py, int max_iter) {
    double py2 = py*py, px025 = px-0.25;
    double q = px025*px025 + py2;
    if (q*(q+px025) <= 0.25*py2 || (px+1.0)*(px+1.0)+py2 <= 0.0625)
        return -1.0;
    double rx=0,ry=0,rx2=0,ry2=0,old_rx=0,old_ry=0;
    int cp=8, iter=0;
    while (iter < max_iter && rx2+ry2 <= 16.0) {
        ry=2.0*rx*ry+py; rx=rx2-ry2+px; rx2=rx*rx; ry2=ry*ry; iter++;
        if(rx2+ry2>16.0) goto esc;
        ry=2.0*rx*ry+py; rx=rx2-ry2+px; rx2=rx*rx; ry2=ry*ry; iter++;
        if(rx2+ry2>16.0) goto esc;
        ry=2.0*rx*ry+py; rx=rx2-ry2+px; rx2=rx*rx; ry2=ry*ry; iter++;
        if(rx2+ry2>16.0) goto esc;
        ry=2.0*rx*ry+py; rx=rx2-ry2+px; rx2=rx*rx; ry2=ry*ry; iter++;
        if(rx2+ry2>16.0) goto esc;
        ry=2.0*rx*ry+py; rx=rx2-ry2+px; rx2=rx*rx; ry2=ry*ry; iter++;
        if(rx2+ry2>16.0) goto esc;
        ry=2.0*rx*ry+py; rx=rx2-ry2+px; rx2=rx*rx; ry2=ry*ry; iter++;
        if(rx2+ry2>16.0) goto esc;
        ry=2.0*rx*ry+py; rx=rx2-ry2+px; rx2=rx*rx; ry2=ry*ry; iter++;
        if(rx2+ry2>16.0) goto esc;
        ry=2.0*rx*ry+py; rx=rx2-ry2+px; rx2=rx*rx; ry2=ry*ry; iter++;
        if(rx==old_rx && ry==old_ry) return -1.0;
        if(iter>=cp){ old_rx=rx; old_ry=ry; cp*=2; }
    }
    if(rx2+ry2 <= 16.0) return -1.0;
esc:;
    double a=rx*rx-ry*ry+px, b=2.0*rx*ry+py;
    double c=a*a-b*b+px,     d=2.0*a*b+py;
    return smooth_color(iter+2, c, d);
}

/* ═══════════════════════════════════════════════════════════════════════════
 *  SINGLE-PIXEL JULIA PATH
 * ═══════════════════════════════════════════════════════════════════════════ */

double mb_pixel_julia(double zr0, double zi0, double jcx, double jcy, int max_iter) {
    double rx=zr0, ry=zi0, rx2=rx*rx, ry2=ry*ry;
    double old_rx=0, old_ry=0;
    int cp=8, iter=0;
    while (iter < max_iter && rx2+ry2 <= 16.0) {
        ry=2.0*rx*ry+jcy; rx=rx2-ry2+jcx; rx2=rx*rx; ry2=ry*ry; iter++;
        if(rx2+ry2>16.0) goto jesc;
        ry=2.0*rx*ry+jcy; rx=rx2-ry2+jcx; rx2=rx*rx; ry2=ry*ry; iter++;
        if(rx2+ry2>16.0) goto jesc;
        ry=2.0*rx*ry+jcy; rx=rx2-ry2+jcx; rx2=rx*rx; ry2=ry*ry; iter++;
        if(rx2+ry2>16.0) goto jesc;
        ry=2.0*rx*ry+jcy; rx=rx2-ry2+jcx; rx2=rx*rx; ry2=ry*ry; iter++;
        if(rx2+ry2>16.0) goto jesc;
        ry=2.0*rx*ry+jcy; rx=rx2-ry2+jcx; rx2=rx*rx; ry2=ry*ry; iter++;
        if(rx2+ry2>16.0) goto jesc;
        ry=2.0*rx*ry+jcy; rx=rx2-ry2+jcx; rx2=rx*rx; ry2=ry*ry; iter++;
        if(rx2+ry2>16.0) goto jesc;
        ry=2.0*rx*ry+jcy; rx=rx2-ry2+jcx; rx2=rx*rx; ry2=ry*ry; iter++;
        if(rx2+ry2>16.0) goto jesc;
        ry=2.0*rx*ry+jcy; rx=rx2-ry2+jcx; rx2=rx*rx; ry2=ry*ry; iter++;
        if(rx==old_rx && ry==old_ry) return -1.0;
        if(iter>=cp){ old_rx=rx; old_ry=ry; cp*=2; }
    }
    if(rx2+ry2 <= 16.0) return -1.0;
jesc:;
    double a=rx*rx-ry*ry+jcx, b=2.0*rx*ry+jcy;
    double c=a*a-b*b+jcx,     d=2.0*a*b+jcy;
    return smooth_color(iter+2, c, d);
}

/* ═══════════════════════════════════════════════════════════════════════════
 *  SCALAR SINGLE-ROW  (fallback, used for Julia rows)
 * ═══════════════════════════════════════════════════════════════════════════ */

void mb_row_std(
    double* __restrict__ out, int w,
    double cx_world, double py,
    double pixel_size, double half_w,
    int max_iter
) {
    double py2 = py*py;
    for (int x = 0; x < w; x++) {
        double px = cx_world + ((double)x - half_w) * pixel_size;
        double px025 = px-0.25, q = px025*px025+py2;
        if (q*(q+px025) <= 0.25*py2 || (px+1.0)*(px+1.0)+py2 <= 0.0625) {
            out[x]=-1.0; continue;
        }
        double rx=0,ry=0,rx2=0,ry2=0,old_rx=0,old_ry=0;
        int cp=8, iter=0;
        while (iter < max_iter && rx2+ry2 <= 16.0) {
            ry=2.0*rx*ry+py; rx=rx2-ry2+px; rx2=rx*rx; ry2=ry*ry; iter++;
            if(rx2+ry2>16.0) break;
            ry=2.0*rx*ry+py; rx=rx2-ry2+px; rx2=rx*rx; ry2=ry*ry; iter++;
            if(rx2+ry2>16.0) break;
            ry=2.0*rx*ry+py; rx=rx2-ry2+px; rx2=rx*rx; ry2=ry*ry; iter++;
            if(rx2+ry2>16.0) break;
            ry=2.0*rx*ry+py; rx=rx2-ry2+px; rx2=rx*rx; ry2=ry*ry; iter++;
            if(rx2+ry2>16.0) break;
            ry=2.0*rx*ry+py; rx=rx2-ry2+px; rx2=rx*rx; ry2=ry*ry; iter++;
            if(rx2+ry2>16.0) break;
            ry=2.0*rx*ry+py; rx=rx2-ry2+px; rx2=rx*rx; ry2=ry*ry; iter++;
            if(rx2+ry2>16.0) break;
            ry=2.0*rx*ry+py; rx=rx2-ry2+px; rx2=rx*rx; ry2=ry*ry; iter++;
            if(rx2+ry2>16.0) break;
            ry=2.0*rx*ry+py; rx=rx2-ry2+px; rx2=rx*rx; ry2=ry*ry; iter++;
            if(rx==old_rx && ry==old_ry){ iter=max_iter; break; }
            if(iter>=cp){ old_rx=rx; old_ry=ry; cp*=2; }
        }
        if(rx2+ry2>16.0){
            double a=rx*rx-ry*ry+px, b=2.0*rx*ry+py;
            double c=a*a-b*b+px,     d=2.0*a*b+py;
            out[x]=smooth_color(iter+2,c,d);
        } else { out[x]=-1.0; }
    }
}

/* ── Scalar Julia row — only compiled when no SIMD path is available ─────── */
#if !defined(__ARM_NEON) && !(defined(__AVX2__) && defined(__FMA__))
void mb_row_julia(
    double* __restrict__ out, int w,
    double cx_world, double py,
    double pixel_size, double half_w,
    double jcx, double jcy,
    int max_iter
) {
    for (int x = 0; x < w; x++) {
        double rx=cx_world+((double)x-half_w)*pixel_size, ry=py;
        double rx2=rx*rx, ry2=ry*ry, old_rx=0, old_ry=0;
        int cp=8, iter=0;
        while (iter < max_iter && rx2+ry2 <= 16.0) {
            ry=2.0*rx*ry+jcy; rx=rx2-ry2+jcx; rx2=rx*rx; ry2=ry*ry; iter++;
            if(rx2+ry2>16.0) break;
            ry=2.0*rx*ry+jcy; rx=rx2-ry2+jcx; rx2=rx*rx; ry2=ry*ry; iter++;
            if(rx2+ry2>16.0) break;
            ry=2.0*rx*ry+jcy; rx=rx2-ry2+jcx; rx2=rx*rx; ry2=ry*ry; iter++;
            if(rx2+ry2>16.0) break;
            ry=2.0*rx*ry+jcy; rx=rx2-ry2+jcx; rx2=rx*rx; ry2=ry*ry; iter++;
            if(rx2+ry2>16.0) break;
            ry=2.0*rx*ry+jcy; rx=rx2-ry2+jcx; rx2=rx*rx; ry2=ry*ry; iter++;
            if(rx2+ry2>16.0) break;
            ry=2.0*rx*ry+jcy; rx=rx2-ry2+jcx; rx2=rx*rx; ry2=ry*ry; iter++;
            if(rx2+ry2>16.0) break;
            ry=2.0*rx*ry+jcy; rx=rx2-ry2+jcx; rx2=rx*rx; ry2=ry*ry; iter++;
            if(rx2+ry2>16.0) break;
            ry=2.0*rx*ry+jcy; rx=rx2-ry2+jcx; rx2=rx*rx; ry2=ry*ry; iter++;
            if(rx==old_rx && ry==old_ry){ iter=max_iter; break; }
            if(iter>=cp){ old_rx=rx; old_ry=ry; cp*=2; }
        }
        if(rx2+ry2>16.0){
            double a=rx*rx-ry*ry+jcx, b=2.0*rx*ry+jcy;
            double c=a*a-b*b+jcx,     d=2.0*a*b+jcy;
            out[x]=smooth_color(iter+2,c,d);
        } else { out[x]=-1.0; }
    }
}
#endif /* !NEON && !AVX2 */

/* ═══════════════════════════════════════════════════════════════════════════
 *  NEON 2-WIDE DOUBLE ROW  (primary hot path for standard Mandelbrot)
 *
 *  Processes pixels in pairs.  Both pixels iterate simultaneously in NEON
 *  float64x2 registers.  Escaped pixels are masked out — their z values
 *  freeze and their iteration counter stops incrementing.  The loop exits
 *  only when both lanes have escaped or max_iter is reached.
 *
 *  Masked iteration counter trick (same as AVX2 Mandelbrot on x86):
 *    active = lanes where mag2 <= 16  (uint64x2, all-1s = active)
 *    iter_vec += (active & 1)         (adds 1 only to active lanes)
 *
 *  After the NEON loop, tail pixel (if w is odd) uses scalar path.
 *  Non-NEON builds fall back to scalar mb_row_std automatically.
 * ═══════════════════════════════════════════════════════════════════════════ */

#ifdef __ARM_NEON

void mb_row_std_neon(
    double* __restrict__ out, int w,
    double cx_world, double py,
    double pixel_size, double half_w,
    int max_iter
) {
    const float64x2_t v_escape  = vdupq_n_f64(16.0);
    const float64x2_t v_two     = vdupq_n_f64(2.0);
    const float64x2_t v_py      = vdupq_n_f64(py);
    const float64x2_t v_neg1    = vdupq_n_f64(-1.0);
    /* For bulb/cardioid check */
    const float64x2_t v_025     = vdupq_n_f64(0.25);
    const float64x2_t v_0625    = vdupq_n_f64(0.0625);
    const float64x2_t v_1       = vdupq_n_f64(1.0);
    const float64x2_t v_py2     = vdupq_n_f64(py * py);

    /* Process pairs of pixels */
    int x = 0;
    for (; x <= w - 2; x += 2) {
        /* Pixel x-coordinates for the pair */
        double px0 = cx_world + ((double)(x)   - half_w) * pixel_size;
        double px1 = cx_world + ((double)(x+1) - half_w) * pixel_size;
        float64x2_t v_px = {px0, px1};

        /* ── Bulb / cardioid rejection (scalar check, rare branch) ── */
        /* If both pixels are inside bulb, write -1 and skip. */
        /* We check individually to avoid complicating the NEON path. */
        int in_bulb0, in_bulb1;
        {
            double px025 = px0-0.25, q = px025*px025+py*py;
            in_bulb0 = (q*(q+px025) <= 0.25*py*py) || ((px0+1.0)*(px0+1.0)+py*py <= 0.0625);
        }
        {
            double px025 = px1-0.25, q = px025*px025+py*py;
            in_bulb1 = (q*(q+px025) <= 0.25*py*py) || ((px1+1.0)*(px1+1.0)+py*py <= 0.0625);
        }
        if (in_bulb0 && in_bulb1) { out[x]=-1.0; out[x+1]=-1.0; continue; }

        /* ── NEON Mandelbrot iteration ── */
        float64x2_t vr  = vdupq_n_f64(0.0);  /* z.real */
        float64x2_t vi  = vdupq_n_f64(0.0);  /* z.imag */
        float64x2_t vr2 = vdupq_n_f64(0.0);  /* z.real² */
        float64x2_t vi2 = vdupq_n_f64(0.0);  /* z.imag² */

        /* iter_vec: iteration count per lane (stored as float64 for NEON add) */
        float64x2_t iter_vec = vdupq_n_f64(0.0);

        /* active_mask: uint64x2, all-1s for lanes still iterating */
        uint64x2_t active = vceqq_f64(vdupq_n_f64(0.0), vdupq_n_f64(0.0)); /* all 1s */

        /* Mask out lanes that were already in the bulb */
        if (in_bulb0) active = vsetq_lane_u64(0, active, 0);
        if (in_bulb1) active = vsetq_lane_u64(0, active, 1);

        int iter = 0;
        for (; iter < max_iter; iter++) {
            /* z_new.real = r² - i² + px */
            float64x2_t new_r = vaddq_f64(vsubq_f64(vr2, vi2), v_px);
            /* z_new.imag = 2*r*i + py  (FMA: py + 2*r*i) */
            float64x2_t new_i = vfmaq_f64(v_py, v_two, vmulq_f64(vr, vi));

            vr  = new_r;
            vi  = new_i;
            vr2 = vmulq_f64(vr, vr);
            vi2 = vmulq_f64(vi, vi);

            float64x2_t mag2 = vaddq_f64(vr2, vi2);

            /* active lanes: mag2 <= 16 */
            /* vcleq_f64 returns all-1s where a <= b */
            uint64x2_t still_in = vcleq_f64(mag2, v_escape);

            /* Only count iterations for still-active lanes */
            active = vandq_u64(active, still_in);

            /* iter_vec += 1.0 for active lanes (reinterpret mask as float: 1.0 has bits 0x3FF0...0) */
            /* Simpler: use a float64x2 of 1.0 ANDed with active mask */
            float64x2_t one_masked = vreinterpretq_f64_u64(
                vandq_u64(vreinterpretq_u64_f64(v_1), active));
            iter_vec = vaddq_f64(iter_vec, one_masked);

            /* Exit when no lane is active */
            if (vgetq_lane_u64(active, 0) == 0 && vgetq_lane_u64(active, 1) == 0)
                break;
        }

        /* Extract results per lane */
        for (int lane = 0; lane < 2; lane++) {
            double px_l = (lane == 0) ? px0 : px1;
            if (lane == 0 && in_bulb0) { out[x]   = -1.0; continue; }
            if (lane == 1 && in_bulb1) { out[x+1] = -1.0; continue; }

            double it_f   = (lane == 0) ? vgetq_lane_f64(iter_vec, 0)
                                        : vgetq_lane_f64(iter_vec, 1);
            int    it     = (int)it_f;
            double mag2_l = (lane == 0)
                ? vgetq_lane_f64(vaddq_f64(vr2,vi2), 0)
                : vgetq_lane_f64(vaddq_f64(vr2,vi2), 1);

            if (mag2_l <= 16.0) {
                /* Did not escape */
                out[x + lane] = -1.0;
            } else {
                double rl = (lane==0) ? vgetq_lane_f64(vr,0) : vgetq_lane_f64(vr,1);
                double il = (lane==0) ? vgetq_lane_f64(vi,0) : vgetq_lane_f64(vi,1);
                /* Two extra iterations for smooth colouring */
                double a = rl*rl - il*il + px_l, b = 2.0*rl*il + py;
                double c = a*a  - b*b   + px_l, d = 2.0*a*b  + py;
                out[x + lane] = smooth_color(it + 2, c, d);
            }
        }
    }

    /* ── Scalar tail for odd width ── */
    for (; x < w; x++) {
        double px = cx_world + ((double)x - half_w) * pixel_size;
        double px025 = px-0.25, q = px025*px025+py*py;
        if (q*(q+px025) <= 0.25*py*py || (px+1.0)*(px+1.0)+py*py <= 0.0625) {
            out[x]=-1.0; continue;
        }
        double rx=0,ry=0,rx2=0,ry2=0,old_rx=0,old_ry=0;
        int cp=8, iter=0;
        while (iter<max_iter && rx2+ry2<=16.0) {
            ry=2.0*rx*ry+py; rx=rx2-ry2+px; rx2=rx*rx; ry2=ry*ry; iter++;
            if(rx2+ry2>16.0) break;
            ry=2.0*rx*ry+py; rx=rx2-ry2+px; rx2=rx*rx; ry2=ry*ry; iter++;
            if(rx2+ry2>16.0) break;
            ry=2.0*rx*ry+py; rx=rx2-ry2+px; rx2=rx*rx; ry2=ry*ry; iter++;
            if(rx2+ry2>16.0) break;
            ry=2.0*rx*ry+py; rx=rx2-ry2+px; rx2=rx*rx; ry2=ry*ry; iter++;
            if(rx2+ry2>16.0) break;
            ry=2.0*rx*ry+py; rx=rx2-ry2+px; rx2=rx*rx; ry2=ry*ry; iter++;
            if(rx2+ry2>16.0) break;
            ry=2.0*rx*ry+py; rx=rx2-ry2+px; rx2=rx*rx; ry2=ry*ry; iter++;
            if(rx2+ry2>16.0) break;
            ry=2.0*rx*ry+py; rx=rx2-ry2+px; rx2=rx*rx; ry2=ry*ry; iter++;
            if(rx2+ry2>16.0) break;
            ry=2.0*rx*ry+py; rx=rx2-ry2+px; rx2=rx*rx; ry2=ry*ry; iter++;
            if(rx==old_rx&&ry==old_ry){iter=max_iter;break;}
            if(iter>=cp){old_rx=rx;old_ry=ry;cp*=2;}
        }
        if(rx2+ry2>16.0){
            double a=rx*rx-ry*ry+px,b=2.0*rx*ry+py;
            double c=a*a-b*b+px,d=2.0*a*b+py;
            out[x]=smooth_color(iter+2,c,d);
        } else { out[x]=-1.0; }
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 *  NEON 2-WIDE JULIA ROW
 *
 *  Julia: z → z² + c, z starts at pixel coord, c=(jcx,jcy) is constant.
 *  No cardioid/bulb rejection. Otherwise same masked-counter technique.
 * ═══════════════════════════════════════════════════════════════════════════ */
void mb_row_julia(
    double* __restrict__ out, int w,
    double cx_world, double py,
    double pixel_size, double half_w,
    double jcx, double jcy,
    int max_iter
) {
    const float64x2_t v_escape  = vdupq_n_f64(16.0);
    const float64x2_t v_two     = vdupq_n_f64(2.0);
    const float64x2_t v_jcx     = vdupq_n_f64(jcx);
    const float64x2_t v_jcy     = vdupq_n_f64(jcy);
    /* All-ones mask: 0xFFFFFFFFFFFFFFFF per lane */
    const uint64x2_t  v_one_u   = vdupq_n_u64(1);

    int x = 0;
    for (; x <= w - 2; x += 2) {
        double zr0 = cx_world + ((double)(x+0) - half_w) * pixel_size;
        double zr1 = cx_world + ((double)(x+1) - half_w) * pixel_size;
        double zr_init[2] = {zr0, zr1};

        float64x2_t vr  = vld1q_f64(zr_init);
        float64x2_t vi  = vdupq_n_f64(py);
        float64x2_t vr2 = vmulq_f64(vr, vr);
        float64x2_t vi2 = vmulq_f64(vi, vi);
        uint64x2_t  iter_vec = vdupq_n_u64(0);
        uint64x2_t  active   = vdupq_n_u64(~(uint64_t)0); /* all lanes active */

        for (int iter = 0; iter < max_iter; iter++) {
            /* new_r = r² - i² + jcx */
            float64x2_t new_r = vaddq_f64(vsubq_f64(vr2, vi2), v_jcx);
            /* new_i = 2*r*i + jcy  (FMA) */
            float64x2_t new_i = vfmaq_f64(v_jcy, v_two, vmulq_f64(vr, vi));
            vr  = new_r;
            vi  = new_i;
            vr2 = vmulq_f64(vr, vr);
            vi2 = vmulq_f64(vi, vi);

            float64x2_t mag2    = vaddq_f64(vr2, vi2);
            /* still_in: mag2 <= 16 → all-1s per lane, else 0 */
            uint64x2_t  still_in = vcleq_f64(mag2, v_escape);
            active    = vandq_u64(active, still_in);
            iter_vec  = vaddq_u64(iter_vec, vandq_u64(v_one_u, active));

            if (vaddvq_u64(active) == 0) break;
        }

        uint64_t iters[2]; vst1q_u64(iters, iter_vec);
        double   rs[2];    vst1q_f64(rs,    vr);
        double   is_[2];   vst1q_f64(is_,   vi);
        float64x2_t mag2_f = vaddq_f64(vr2, vi2);
        double   mag2s[2]; vst1q_f64(mag2s, mag2_f);

        for (int i = 0; i < 2; i++) {
            if (mag2s[i] <= 16.0) {
                out[x+i] = -1.0;
            } else {
                int it = (int)iters[i];
                double a = rs[i]*rs[i] - is_[i]*is_[i] + jcx;
                double b = 2.0*rs[i]*is_[i] + jcy;
                double c = a*a - b*b + jcx;
                double d = 2.0*a*b  + jcy;
                out[x+i] = smooth_color(it + 2, c, d);
            }
        }
    }
    /* Scalar tail (odd width) */
    for (; x < w; x++) {
        double rx = cx_world + ((double)x - half_w) * pixel_size, ry = py;
        double rx2=rx*rx, ry2=ry*ry, old_rx=0, old_ry=0;
        int cp=8, iter=0;
        while (iter < max_iter && rx2+ry2 <= 16.0) {
            ry=2.0*rx*ry+jcy; rx=rx2-ry2+jcx; rx2=rx*rx; ry2=ry*ry; iter++;
            if(rx2+ry2>16.0) break;
            ry=2.0*rx*ry+jcy; rx=rx2-ry2+jcx; rx2=rx*rx; ry2=ry*ry; iter++;
            if(rx2+ry2>16.0) break;
            ry=2.0*rx*ry+jcy; rx=rx2-ry2+jcx; rx2=rx*rx; ry2=ry*ry; iter++;
            if(rx2+ry2>16.0) break;
            ry=2.0*rx*ry+jcy; rx=rx2-ry2+jcx; rx2=rx*rx; ry2=ry*ry; iter++;
            if(rx==old_rx && ry==old_ry){ iter=max_iter; break; }
            if(iter>=cp){ old_rx=rx; old_ry=ry; cp*=2; }
        }
        if(rx2+ry2>16.0){
            double a=rx*rx-ry*ry+jcx, b=2.0*rx*ry+jcy;
            double c=a*a-b*b+jcx, d=2.0*a*b+jcy;
            out[x]=smooth_color(iter+2,c,d);
        } else { out[x]=-1.0; }
    }
}

#elif defined(__AVX2__) && defined(__FMA__)

/* ═══════════════════════════════════════════════════════════════════════════
 *  AVX2 4-WIDE DOUBLE ROW  (x86-64 fast path)
 *
 *  AVX2 __m256d holds 4× float64 in a 256-bit register — twice the width
 *  of ARM NEON.  The algorithm is identical to the NEON path:
 *    • Compute z² + c for all 4 lanes simultaneously
 *    • Mask escaped lanes out of the iteration counter
 *    • Exit when all 4 lanes have escaped or hit max_iter
 *
 *  Key AVX2 intrinsics:
 *    _mm256_set1_pd(x)         — broadcast scalar to all 4 lanes
 *    _mm256_set_pd(d,c,b,a)    — load 4 distinct values (note: reverse order)
 *    _mm256_mul_pd(a,b)        — 4-wide multiply
 *    _mm256_fmadd_pd(a,b,c)    — a*b+c  (FMA, requires __FMA__)
 *    _mm256_sub_pd(a,b)        — subtraction
 *    _mm256_add_pd(a,b)        — addition
 *    _mm256_cmp_pd(a,b,_CMP_LE_OQ) — compare <=, returns all-1s per lane
 *    _mm256_and_pd(a,b)        — bitwise AND (mask application)
 *    _mm256_movemask_pd(a)     — 4-bit mask from sign bits (exit test)
 *    _mm256_storeu_pd(ptr,v)   — store 4 doubles unaligned
 *
 *  Expected speedup vs scalar on modern x86: ~3.0-3.8×
 *  Combined with x4-row interleaving: effective ~12-15× vs original.
 * ═══════════════════════════════════════════════════════════════════════════ */

#include <immintrin.h>

void mb_row_std_neon(
    double* __restrict__ out, int w,
    double cx_world, double py,
    double pixel_size, double half_w,
    int max_iter
) {
    const __m256d v_escape = _mm256_set1_pd(16.0);
    const __m256d v_two    = _mm256_set1_pd(2.0);
    const __m256d v_one    = _mm256_set1_pd(1.0);
    const __m256d v_py     = _mm256_set1_pd(py);
    const __m256d v_neg1   = _mm256_set1_pd(-1.0);
    double py2 = py * py;

    int x = 0;
    /* ── Process 4 pixels at a time ── */
    for (; x <= w - 4; x += 4) {
        double px0 = cx_world + ((double)(x+0) - half_w) * pixel_size;
        double px1 = cx_world + ((double)(x+1) - half_w) * pixel_size;
        double px2 = cx_world + ((double)(x+2) - half_w) * pixel_size;
        double px3 = cx_world + ((double)(x+3) - half_w) * pixel_size;

        /* Bulb/cardioid scalar pre-check — skip entire group if all inside */
        int b0,b1,b2,b3;
        { double p=px0-0.25,q=p*p+py2; b0=(q*(q+p)<=0.25*py2)||((px0+1)*(px0+1)+py2<=0.0625); }
        { double p=px1-0.25,q=p*p+py2; b1=(q*(q+p)<=0.25*py2)||((px1+1)*(px1+1)+py2<=0.0625); }
        { double p=px2-0.25,q=p*p+py2; b2=(q*(q+p)<=0.25*py2)||((px2+1)*(px2+1)+py2<=0.0625); }
        { double p=px3-0.25,q=p*p+py2; b3=(q*(q+p)<=0.25*py2)||((px3+1)*(px3+1)+py2<=0.0625); }

        if (b0 && b1 && b2 && b3) {
            out[x]=out[x+1]=out[x+2]=out[x+3]=-1.0;
            continue;
        }

        /* Load cx per pixel — _mm256_set_pd fills lanes 3,2,1,0 (reversed) */
        __m256d v_cx = _mm256_set_pd(px3, px2, px1, px0);

        /* z.r, z.i, z.r², z.i² — all start at 0 */
        __m256d vr  = _mm256_setzero_pd();
        __m256d vi  = _mm256_setzero_pd();
        __m256d vr2 = _mm256_setzero_pd();
        __m256d vi2 = _mm256_setzero_pd();

        /* Iteration counter per lane (as double for masked add) */
        __m256d iter_vec = _mm256_setzero_pd();

        /* Active mask: all lanes active initially.
         * Lanes in the bulb start inactive. */
        __m256d active = _mm256_cmp_pd(_mm256_setzero_pd(),
                                       _mm256_setzero_pd(), _CMP_EQ_OQ); /* all 1s */
        if (b0) active = _mm256_blend_pd(active, _mm256_setzero_pd(), 0x1);
        if (b1) active = _mm256_blend_pd(active, _mm256_setzero_pd(), 0x2);
        if (b2) active = _mm256_blend_pd(active, _mm256_setzero_pd(), 0x4);
        if (b3) active = _mm256_blend_pd(active, _mm256_setzero_pd(), 0x8);

        for (int iter = 0; iter < max_iter; iter++) {
            /* z_new.r = r² - i² + cx  */
            __m256d new_r = _mm256_add_pd(_mm256_sub_pd(vr2, vi2), v_cx);
            /* z_new.i = 2*r*i + py  (FMA: py + r*i*2) */
            __m256d new_i = _mm256_fmadd_pd(v_two, _mm256_mul_pd(vr, vi), v_py);

            vr  = new_r;
            vi  = new_i;
            vr2 = _mm256_mul_pd(vr, vr);
            vi2 = _mm256_mul_pd(vi, vi);

            __m256d mag2 = _mm256_add_pd(vr2, vi2);

            /* Lanes still inside: mag2 <= 16 */
            __m256d still_in = _mm256_cmp_pd(mag2, v_escape, _CMP_LE_OQ);

            /* Narrow active: only lanes that were active AND still inside */
            active = _mm256_and_pd(active, still_in);

            /* iter_vec += 1.0 masked to active lanes */
            iter_vec = _mm256_add_pd(iter_vec, _mm256_and_pd(v_one, active));

            /* Exit early if all lanes escaped (_mm256_movemask_pd == 0) */
            if (_mm256_movemask_pd(active) == 0) break;
        }

        /* Extract and write results */
        double iters[4], rs[4], is[4], mag2s[4];
        _mm256_storeu_pd(iters, iter_vec);
        _mm256_storeu_pd(rs,    vr);
        _mm256_storeu_pd(is,    vi);
        __m256d mag2_final = _mm256_add_pd(vr2, vi2);
        _mm256_storeu_pd(mag2s, mag2_final);

        double pxs[4] = {px0, px1, px2, px3};
        int    bulbs[4] = {b0, b1, b2, b3};
        for (int i = 0; i < 4; i++) {
            if (bulbs[i] || mag2s[i] <= 16.0) {
                out[x+i] = -1.0;
            } else {
                int it = (int)iters[i];
                double a = rs[i]*rs[i] - is[i]*is[i] + pxs[i];
                double b = 2.0*rs[i]*is[i] + py;
                double c = a*a - b*b + pxs[i];
                double d = 2.0*a*b + py;
                out[x+i] = smooth_color(it + 2, c, d);
            }
        }
    }

    /* ── Scalar tail for remainder (w % 4 != 0) ── */
    for (; x < w; x++) {
        double px = cx_world + ((double)x - half_w) * pixel_size;
        double px025 = px-0.25, q = px025*px025+py2;
        if (q*(q+px025) <= 0.25*py2 || (px+1.0)*(px+1.0)+py2 <= 0.0625) {
            out[x]=-1.0; continue;
        }
        double rx=0,ry=0,rx2=0,ry2=0,old_rx=0,old_ry=0;
        int cp=8, iter=0;
        while (iter<max_iter && rx2+ry2<=16.0) {
            ry=2.0*rx*ry+py; rx=rx2-ry2+px; rx2=rx*rx; ry2=ry*ry; iter++;
            if(rx2+ry2>16.0) break;
            ry=2.0*rx*ry+py; rx=rx2-ry2+px; rx2=rx*rx; ry2=ry*ry; iter++;
            if(rx2+ry2>16.0) break;
            ry=2.0*rx*ry+py; rx=rx2-ry2+px; rx2=rx*rx; ry2=ry*ry; iter++;
            if(rx2+ry2>16.0) break;
            ry=2.0*rx*ry+py; rx=rx2-ry2+px; rx2=rx*rx; ry2=ry*ry; iter++;
            if(rx2+ry2>16.0) break;
            ry=2.0*rx*ry+py; rx=rx2-ry2+px; rx2=rx*rx; ry2=ry*ry; iter++;
            if(rx2+ry2>16.0) break;
            ry=2.0*rx*ry+py; rx=rx2-ry2+px; rx2=rx*rx; ry2=ry*ry; iter++;
            if(rx2+ry2>16.0) break;
            ry=2.0*rx*ry+py; rx=rx2-ry2+px; rx2=rx*rx; ry2=ry*ry; iter++;
            if(rx2+ry2>16.0) break;
            ry=2.0*rx*ry+py; rx=rx2-ry2+px; rx2=rx*rx; ry2=ry*ry; iter++;
            if(rx==old_rx&&ry==old_ry){iter=max_iter;break;}
            if(iter>=cp){old_rx=rx;old_ry=ry;cp*=2;}
        }
        if(rx2+ry2>16.0){
            double a=rx*rx-ry*ry+px, b=2.0*rx*ry+py;
            double c=a*a-b*b+px, d=2.0*a*b+py;
            out[x]=smooth_color(iter+2,c,d);
        } else { out[x]=-1.0; }
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 *  AVX2 4-WIDE JULIA ROW
 *
 *  Julia iteration: z → z² + c  where c=(jcx,jcy) is fixed and z starts
 *  at the pixel coordinate.  No cardioid/bulb rejection (doesn't apply).
 *  Otherwise identical structure to mb_row_std_neon AVX2 above.
 * ═══════════════════════════════════════════════════════════════════════════ */
void mb_row_julia(
    double* __restrict__ out, int w,
    double cx_world, double py,
    double pixel_size, double half_w,
    double jcx, double jcy,
    int max_iter
) {
    const __m256d v_escape = _mm256_set1_pd(16.0);
    const __m256d v_two    = _mm256_set1_pd(2.0);
    const __m256d v_one    = _mm256_set1_pd(1.0);
    const __m256d v_jcx    = _mm256_set1_pd(jcx);
    const __m256d v_jcy    = _mm256_set1_pd(jcy);
    const __m256d v_active_all = _mm256_cmp_pd(_mm256_setzero_pd(),
                                               _mm256_setzero_pd(), _CMP_EQ_OQ);
    int x = 0;
    for (; x <= w - 4; x += 4) {
        /* Initial z = pixel coordinate */
        double zr0 = cx_world + ((double)(x+0) - half_w) * pixel_size;
        double zr1 = cx_world + ((double)(x+1) - half_w) * pixel_size;
        double zr2 = cx_world + ((double)(x+2) - half_w) * pixel_size;
        double zr3 = cx_world + ((double)(x+3) - half_w) * pixel_size;

        __m256d vr  = _mm256_set_pd(zr3, zr2, zr1, zr0);
        __m256d vi  = _mm256_set1_pd(py);
        __m256d vr2 = _mm256_mul_pd(vr, vr);
        __m256d vi2 = _mm256_mul_pd(vi, vi);
        __m256d iter_vec = _mm256_setzero_pd();
        __m256d active   = v_active_all;

        for (int iter = 0; iter < max_iter; iter++) {
            /* new_r = r² - i² + jcx */
            __m256d new_r = _mm256_add_pd(_mm256_sub_pd(vr2, vi2), v_jcx);
            /* new_i = 2*r*i + jcy */
            __m256d new_i = _mm256_fmadd_pd(v_two, _mm256_mul_pd(vr, vi), v_jcy);
            vr  = new_r;
            vi  = new_i;
            vr2 = _mm256_mul_pd(vr, vr);
            vi2 = _mm256_mul_pd(vi, vi);

            __m256d mag2     = _mm256_add_pd(vr2, vi2);
            __m256d still_in = _mm256_cmp_pd(mag2, v_escape, _CMP_LE_OQ);
            active           = _mm256_and_pd(active, still_in);
            iter_vec         = _mm256_add_pd(iter_vec, _mm256_and_pd(v_one, active));

            if (_mm256_movemask_pd(active) == 0) break;
        }

        double iters[4], rs[4], is_[4];
        _mm256_storeu_pd(iters, iter_vec);
        _mm256_storeu_pd(rs,    vr);
        _mm256_storeu_pd(is_,   vi);
        __m256d mag2_final = _mm256_add_pd(vr2, vi2);
        double  mag2s[4];
        _mm256_storeu_pd(mag2s, mag2_final);

        for (int i = 0; i < 4; i++) {
            if (mag2s[i] <= 16.0) {
                out[x+i] = -1.0;
            } else {
                int it = (int)iters[i];
                double a = rs[i]*rs[i] - is_[i]*is_[i] + jcx;
                double b = 2.0*rs[i]*is_[i] + jcy;
                double c = a*a - b*b + jcx;
                double d = 2.0*a*b  + jcy;
                out[x+i] = smooth_color(it + 2, c, d);
            }
        }
    }
    /* Scalar tail */
    for (; x < w; x++) {
        double rx = cx_world + ((double)x - half_w) * pixel_size, ry = py;
        double rx2=rx*rx, ry2=ry*ry, old_rx=0, old_ry=0;
        int cp=8, iter=0;
        while (iter < max_iter && rx2+ry2 <= 16.0) {
            ry=2.0*rx*ry+jcy; rx=rx2-ry2+jcx; rx2=rx*rx; ry2=ry*ry; iter++;
            if(rx2+ry2>16.0) break;
            ry=2.0*rx*ry+jcy; rx=rx2-ry2+jcx; rx2=rx*rx; ry2=ry*ry; iter++;
            if(rx2+ry2>16.0) break;
            ry=2.0*rx*ry+jcy; rx=rx2-ry2+jcx; rx2=rx*rx; ry2=ry*ry; iter++;
            if(rx2+ry2>16.0) break;
            ry=2.0*rx*ry+jcy; rx=rx2-ry2+jcx; rx2=rx*rx; ry2=ry*ry; iter++;
            if(rx2+ry2>16.0) break;
            ry=2.0*rx*ry+jcy; rx=rx2-ry2+jcx; rx2=rx*rx; ry2=ry*ry; iter++;
            if(rx2+ry2>16.0) break;
            ry=2.0*rx*ry+jcy; rx=rx2-ry2+jcx; rx2=rx*rx; ry2=ry*ry; iter++;
            if(rx2+ry2>16.0) break;
            ry=2.0*rx*ry+jcy; rx=rx2-ry2+jcx; rx2=rx*rx; ry2=ry*ry; iter++;
            if(rx2+ry2>16.0) break;
            ry=2.0*rx*ry+jcy; rx=rx2-ry2+jcx; rx2=rx*rx; ry2=ry*ry; iter++;
            if(rx==old_rx && ry==old_ry){ iter=max_iter; break; }
            if(iter>=cp){ old_rx=rx; old_ry=ry; cp*=2; }
        }
        if(rx2+ry2>16.0){
            double a=rx*rx-ry*ry+jcx, b=2.0*rx*ry+jcy;
            double c=a*a-b*b+jcx, d=2.0*a*b+jcy;
            out[x]=smooth_color(iter+2,c,d);
        } else { out[x]=-1.0; }
    }
}

#else
/* Scalar fallback for non-NEON, non-AVX2 builds */
void mb_row_std_neon(
    double* __restrict__ out, int w,
    double cx_world, double py,
    double pixel_size, double half_w,
    int max_iter
) {
    mb_row_std(out, w, cx_world, py, pixel_size, half_w, max_iter);
}
#endif /* __ARM_NEON / __AVX2__ */

/* ═══════════════════════════════════════════════════════════════════════════
 *  4-ROW INTERLEAVED BATCH  (calls NEON row function for each row)
 *
 *  Scheduling 4 rows per goroutine dispatch reduces Go scheduler overhead.
 *  Each row uses the NEON 2-wide path internally, so we get 4×2 = 8 pixels
 *  of effective parallelism per goroutine wakeup vs 1 in the original code.
 * ═══════════════════════════════════════════════════════════════════════════ */

void mb_row_std_x4(
    double* __restrict__ out, int w, int n_rows,
    double cx_world, double py_base,
    double pixel_size, double half_w,
    int max_iter
) {
    for (int r = 0; r < n_rows; r++) {
        double py = py_base + (double)r * pixel_size;
        mb_row_std_neon(out + r*w, w, cx_world, py, pixel_size, half_w, max_iter);
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 *  PERTURBATION THEORY INNER LOOP
 * ═══════════════════════════════════════════════════════════════════════════ */

double mb_perturb_pixel(
    const double* __restrict__ ref_x,
    const double* __restrict__ ref_y,
    const double* __restrict__ ref2x,
    const double* __restrict__ ref2y,
    int ref_len,
    double dcx, double dcy,
    double dx0, double dy0,
    int sa_iter,
    double px, double py,
    int max_iter
) {
    double dx=dx0, dy=dy0, rx=0.0, ry=0.0;
    int iter=sa_iter, escaped=0;

    const double *rx_ptr=ref_x+iter, *ry_ptr=ref_y+iter;
    const double *r2x_ptr=ref2x+iter, *r2y_ptr=ref2y+iter;

    int remaining = ref_len - iter;
    if (remaining > max_iter - iter) remaining = max_iter - iter;

    for (int i = 0; i < remaining; i++) {
        rx = rx_ptr[i] + dx;
        ry = ry_ptr[i] + dy;
        if (rx*rx + ry*ry > 16.0) { escaped=1; iter+=i+1; break; }
        double dx2=dx*dx, dy2=dy*dy, dxdy2=2.0*dx*dy;
        double ndx = r2x_ptr[i]*dx - r2y_ptr[i]*dy + dx2 - dy2 + dcx;
        double ndy = r2x_ptr[i]*dy + r2y_ptr[i]*dx + dxdy2 + dcy;
        dx=ndx; dy=ndy;
        if (i == remaining-1) iter+=remaining;
    }

    if (!escaped) {
        if (dx*dx+dy*dy > (rx*rx+ry*ry)*1e6) return -2.0;
        return -1.0;
    }
    double rx2=rx*rx-ry*ry+px, ry2=2.0*rx*ry+py;
    double rx3=rx2*rx2-ry2*ry2+px, ry3=2.0*rx2*ry2+py;
    return smooth_color(iter+2, rx3, ry3);
}
