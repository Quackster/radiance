// LFM2's 64-wide heads use the existing query-tiled paged attention kernel.
// The same launch serves prefill and short queries; a split-KV decode specialization
// can be added independently without changing the architecture or KV layout.
#include "r4d.h"
#include "r4d_attn_args.h"
#include "r4d_common.h"
#include <cmath>
#include <algorithm>
using std::max;
#include "r4d_attn_prefill_h256_gqa6.hip"

extern "C" int r4d_rad_attn_h64_gqa4_bf16(const RadArgs* a, RadStream s) {
    R4DArgs r;
    const int rc = fill_paged(a, &r, 64);
    if (rc != RAD_OK) return rc;
    if (r.block_size != 16 || r.kv_heads <= 0 || r.q_heads != 4 * r.kv_heads ||
        r.window != 0 || r.causal != 1) return RAD_E_SHAPE;
    if (a->t[AP_Q].dtype != RAD_BF16 || a->t[AP_KV].dtype != RAD_BF16 ||
        a->t[AP_OUT].dtype != RAD_BF16) return RAD_E_DTYPE;
    // Disable the 256-wide K fetch, as the existing 128-wide specialization does.
    constexpr int waves = 8, tile = 32, options = 3430923, query_tile = waves * 16 / 4;
    hipLaunchKernelGGL((r4d_attn_prefill_kernel<waves, tile, 64, 4, 16, 1, options>),
                      dim3((r.q_len + query_tile - 1) / query_tile, r.kv_heads, r.num_seqs),
                      dim3(waves * 32), 0, (hipStream_t)s, r);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}
