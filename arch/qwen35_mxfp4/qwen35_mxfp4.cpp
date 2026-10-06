/* qwen35_mxfp4 -- Qwen3.5 / 3.6 / 3.8 dense, OCP-MXFP4 WEIGHTS and block-scaled FP8 activations.
 *
 * IT IS arch/qwen35_fp8/qwen35_fp8.cpp UNDER A SECOND QUANTISATION DESCRIPTOR, and that is the
 * whole of this file: three macros and an include, as arch/qwen35moe_fp8 is. The graph does not
 * change with the weight format. Every projection is a LinearFP8 (arch/common/rad_fp8.h), which
 * reads its weight's encoding and declares the GEMM that serves it -- block fp8 at fp8a8, OCP-MXFP4
 * at mxfp4a8 -- against the same E4M3 activation pair, so every fused producer in front of a
 * linear (the norm, the gate, the delta net's and the attention's epilogues) feeds an MXFP4 weight
 * unchanged. What a second translation unit buys is SELECTION (spec §2.4): a checkpoint whose
 * quantisation descriptor is `mxfp4` picks this plugin and never the fp8 one.
 *
 * ============================== WHAT IT SERVES ==============================
 *
 * A compressed-tensors release with
 *
 *     "quantization_config": { "quant_method": "compressed-tensors",
 *                              "format": "mxfp4-pack-quantized", ... group_size 32 ... }
 *
 * -- llm-compressor's GPTQ/RTN MXFP4, e.g. a GPTQ-MXFP4 Qwen3.8-27B. core/format/checkpoint.cpp
 * reads each `X.weight_packed` / `X.weight_scale` pair as the two planes of
 * `fp4_e2m1*e8m0[1x32]`, so rad-convert copies them byte for byte. What the release leaves bf16
 * (lm_head, the MTP head, the vision tower) is the recipe's to decide, and
 * data/recipes/q38-27b-mxfp4.recipe makes the head and the MTP block fp8 and widens the delta
 * net's a/b projection back to the bf16 this graph reads it at (exact: every e2m1 times a power
 * of two is a bf16).
 */
#define QWEN35_NS      qwen35_mxfp4
#define QWEN35_QUANT   "mxfp4"
#define QWEN35_MXFP4   1

#include "../qwen35_fp8/qwen35_fp8.cpp"
