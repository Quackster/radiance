/* qwen35_mxfp6 -- Qwen3.5 / 3.6 / 3.8 dense, OCP-MX FP6 weights and fp8 activations.
 *
 * IT IS arch/qwen35_mxfp4/qwen35_mxfp4.cpp AT A DIFFERENT CODEBOOK, and that is the whole of this
 * file: four macros and an include -- the qwen35moe pattern. The MX trunk in rad_mx.h reads its
 * codebook from the container's own encoding (`mx_dtype_str()` picks mxfp6a8 off `plane[0].dtype`,
 * and the per-row reference, the fp4 fold, is present exactly when the codebook is e2m1), so the
 * graph, the schedule, the folds, the heads and the drafter are the plugin next door. The macros
 * below carry the selection key and the guard text: a container converted with
 *
 *     rad-convert <Qwen3.8-27B bf16 checkpoint> --quant-id mxfp6 \
 *         --recipe data/recipes/q38-27b-mxfp6.recipe -o q38-27b-mxfp6.rad
 *
 * reports `qwen35` + `mxfp6` (spec §2.4), and only this .so claims that pair.
 *
 * THE FORMAT IT SERVES is the OCP-MX grid with e2m3 codes: `fp6_e2m3*e8m0[1x32]`. The container's
 * canonical code plane is PACKED -- four 6-bit codes per three bytes LSB-first along the row, so
 * [N, 3K/4] bytes -- and libr4d's layout hook maps it into fragment order with the same slot
 * structure as mxfp4's, then the kernel maps each e2m3 code to an exact e4m3 byte (every e2m3
 * value, subnormals included, is exactly an e4m3 value) before the WMMA. One stored weight is TWO
 * operands of `gemm_nt_q` at dtype mxfp6a8 -- codes and exponent plane, no reference byte: the
 * codes self-scale and the e8m0 rides in the epilogue, so the gated decode fold stays mxfp4's.
 * All of that lives in libr4d and rad_mx.h; none of it appears here.
 */
#define QWEN35_NS           qwen35_mxfp6
#define QWEN35_ARCH_ID      "qwen35"
#define QWEN35_QUANT        "mxfp6"
#define QWEN35_MX_CODE_NAME "E2M3"

#include "../qwen35_mxfp4/qwen35_mxfp4.cpp"
