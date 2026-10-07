/* qwen35_paroquant -- Qwen3.5 / 3.6 / 3.8 dense, a NATIVE ParoQuant trunk (path A).
 *
 * IT IS arch/qwen35_mxfp6/qwen35_mxfp6.cpp WITH THE ROTATION KEPT. A ParoQuant checkpoint ships its
 * trunk linears rotated: the stored codes reconstruct to `w = R(v) . D`, so a GEMM that reads the
 * codes `v` must be fed `a = (x / t_scale) . M_dec^T`, not the plain normed activation. The sibling
 * plugin qwen35_mxfp6 serves the PATH-B container, where rad-convert undoes the rotation at read
 * time and requantises to plain `fp6_e2m3*e8m0[1x32]` -- simpler, but it loses the ParoQuant
 * quality (a second 3-mantissa-bit rounding after the rotation). This plugin serves the rotation
 * itself, so a weight's error is the source's alone.
 *
 * WHAT DIFFERS (all gated on QWEN35_PAROQUANT, so no other MX arch sees it):
 *
 *   - the weight is a 5-plane `givens` encoding -- the packed e2m3 codes and the e8m0[1x32] plane a
 *     plain MXFP6 GEMM reads, plus the checkpoint's `pairs`/`angle`/`1-channel_scales` tables as
 *     trailing planes. r4d_layout.cpp's mxw_enc_layout lays the codes and scale out exactly as for a
 *     plain MXFP6 weight; mx_dtype_str admits the transform only under this macro.
 *   - gate_proj and up_proj are TWO GEMMs, not one stacked weight: their `theta`/`channel_scales`
 *     differ (the layer's `pairs` are shared), so one activation cannot feed both. Each reads the
 *     output of its own `paroquant_quant_fp8_row` producer (rad_mx.h's ParoProducer) over the
 *     block's normed bf16; down_proj reads a producer over the bf16 silu product.
 *   - the lm_head, the MTP head and the DFlash2 drafter stay BLOCK FP8, exactly as in the MX plugin.
 *
 * A container is produced with
 *
 *     rad-convert <Qwen3.8-27B paroquant_mxfp6 checkpoint> --quant-id paroquant_mxfp6 \
 *         -o q38-27b-paroquant-mxfp6.rad
 *
 * and reports `qwen35` + `paroquant_mxfp6`, the pair only this .so claims.
 */
#define QWEN35_NS           qwen35_paroquant
#define QWEN35_ARCH_ID      "qwen35"
#define QWEN35_QUANT        "paroquant_mxfp6"
#define QWEN35_MX_CODE_NAME "E2M3"
#define QWEN35_PAROQUANT    1

#include "../qwen35_mxfp4/qwen35_mxfp4.cpp"
