/* qwen35_mxfp8 -- Qwen3.5 / 3.6 / 3.8 dense, OCP-MX FP8 weights and fp8 activations.
 *
 * IT IS arch/qwen35_mxfp4/qwen35_mxfp4.cpp AT A DIFFERENT CODEBOOK, and that is the whole of this
 * file: four macros and an include. The MX trunk in rad_mx.h reads its codebook from the
 * container's own encoding -- `mx_dtype_str()` picks mxfp4a8 / mxfp8a8 / mxfp6a8 off
 * `plane[0].dtype`, and the per-row reference (the fp4 fold) is present exactly when the codebook
 * is e2m1 -- so the graph, the schedule, the folds, the heads and the drafter are letter-for-letter
 * the plugin next door. What differs is the selection key and the guard text, and those are what
 * the macros below carry: a container converted with
 *
 *     rad-convert <Qwen3.8-27B bf16 checkpoint> --quant-id mxfp8 \
 *         --recipe data/recipes/q38-27b-mxfp8.recipe -o q38-27b-mxfp8.rad
 *
 * reports `qwen35` + `mxfp8` (spec §2.4), and only this .so claims that pair.
 *
 * THE FORMAT IT SERVES is the OCP-MX grid with e4m3 codes: `fp8_e4m3*e8m0[1x32]`, the codes a
 * plain u8 plane and the exponents one byte per 32 elements along K. One stored weight is TWO
 * operands of `gemm_nt_q` at dtype mxfp8a8 -- codes and exponent plane, no reference byte: e4m3
 * codes self-scale, the kernel applies the e8m0 per group, so there is nothing for a per-row
 * reference to fold against and the gated decode fold stays mxfp4's. That difference lives in
 * rad_mx.h (the `folds` predicate) and in libr4d (A2's shape/layout rows); it does not appear
 * here.
 *
 * WHY A SECOND TRANSLATION UNIT AT ALL: one .so cannot claim two quantisation ids, and an
 * operator looking at what is installed should see one plugin per (architecture, quantisation).
 * The precedent is qwen35moe_fp8 -- three macros and an include.
 */
#define QWEN35_NS           qwen35_mxfp8
#define QWEN35_ARCH_ID      "qwen35"
#define QWEN35_QUANT        "mxfp8"
#define QWEN35_MX_CODE_NAME "E4M3"

#include "../qwen35_mxfp4/qwen35_mxfp4.cpp"
