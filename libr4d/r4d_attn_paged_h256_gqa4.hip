// R4D attention at FOUR queries per KV head, head_dim 256 -- Qwen3.5-0.8B and its siblings that
// keep the 27B's head width with fewer heads (8 query heads over 2 KV heads).
//
// The geometry and the dispatch only, as r4d_attn_paged_h256_gqa12.hip is: r4d_attn_paged_gqa8.hip
// is the parameterised body and its kernels are templates on `gqa`. What the ratio changes is the
// two bounds, and both are in the ROW table rather than here:
//
//   the decode verify width   64 / gqa = 16, against 8 at gqa 8 and 10 at gqa 6. A workgroup holds
//                             64 (query, head) rows and carries q_len * gqa of them.
//   a prefill block's width   384 / gqa = 96 query positions, against 48 at gqa 8.
//
// NO head_dim 128 LEG. gqa 4 at 128 is the DFlash2 drafter's windowed geometry, which
// r4d_attn_paged_h128_gqa4.hip serves; R4D_ATTN_H128 leaves those entry points out of this unit.
// The page is the body's default 16.
#define R4D_ATTN_GQA  4
#define R4D_ATTN_H128 0

#include "r4d_attn_paged_gqa8.hip"
