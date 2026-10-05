/* relayout_parity_test.cpp -- a kernel's relayout of libquant's planes is the stored form the
 * kernel reads, pinned byte for byte.
 *
 * A container holds a quantiser's canonical planes and the kernel only arranges them at load
 * (spec.md §4.3). For every stored format, libquant's planes passed through the kernel's relayout
 * are held to a recorded FNV-1a hash of the stored bytes -- the hashes are what the containers in
 * service were quantised to, so a recipe reconverts one without moving a single served bit, and a
 * relayout or quantiser edit that changes a format fails here rather than as a quality drift no
 * test downstream is sharp enough to see. Where the kernel can be read back, the inverse has to
 * give the planes back exactly.
 *
 * libr4d needs a HIP toolchain to build and its layouts do not, so their host sources are compiled
 * into this test (tests/CMakeLists.txt). The n-gram table's hook is libavx's, from the plugin.
 */
#include "quant_fixture.h"
#include "r4d_plugin.h"

#include <dlfcn.h>
#include <unistd.h>

using namespace qfix;

namespace {

/* One kernel family's hooks. */
struct Hooks {
    RadLayoutFn     layout = nullptr;
    RadRelayoutFn   relayout = nullptr;
    RadUnrelayoutFn unrelayout = nullptr;
};

uint64_t fnv(const std::vector<uint8_t>& b, uint64_t h = 0xcbf29ce484222325ull) {
    for (uint8_t c : b) h = (h ^ c) * 0x100000001b3ull;
    return h;
}

/* Two hashes into one, order-sensitive -- a XOR would cancel two equal planes to zero. */
uint64_t chain(uint64_t a, uint64_t b) {
    for (int i = 0; i < 8; ++i) a = (a ^ ((b >> (8 * i)) & 0xFF)) * 0x100000001b3ull;
    return a;
}

/* The planes `roles` names, as the loader will hand them to a relayout: dense [rows, cols] in each
 * plane's own elements. `bufs` replaces the data when given -- the inverse's destinations. */
struct Selection {
    std::vector<int> sel;
    std::vector<RadTensor> t;
};
Selection select(const Quantised& q, int64_t R, int64_t C, const std::vector<const char*>& roles,
                 std::vector<std::vector<uint8_t>>* bufs = nullptr) {
    Selection s;
    for (const char* role : roles) {
        const int i = rad_enc_find(&q.enc, role);
        s.sel.push_back(i);
        RadTensor t{};
        if (i < 0) { s.t.push_back(t); continue; }
        const RadEncPlane& p = q.enc.plane[i];
        int64_t pr = 0, pc = 0;
        rad_enc_plane_dims(&p, R, C, &pr, &pc);
        if (bufs) bufs->emplace_back(q.planes[(size_t)i].size(), 0);
        t.data = bufs ? bufs->back().data() : const_cast<uint8_t*>(q.planes[(size_t)i].data());
        t.dtype = p.dtype;
        t.rank = 2;
        t.shape[0] = pr;
        t.shape[1] = pc;
        t.stride[0] = pc;
        t.stride[1] = 1;
        s.t.push_back(t);
    }
    return s;
}

/* One operand of one weight: what to compare. */
struct Op {
    int operand;
    std::vector<const char*> roles;
};

/* THE CHECK. Quantise `w` with libquant, store each operand through the kernel's hooks, invert.
 * Returns the hash of everything stored, in operand order. */
uint64_t stored(const char* label, const Hooks& nf, Opts kp, const char* quant,
                Opts qo, const char* wname, int64_t R, int64_t C, const std::vector<float>& w,
                const std::vector<Op>& ops, Quantised* keep = nullptr) {
    uint64_t h = 0xcbf29ce484222325ull;
    const Quantised q = quantise(quantizer(quant), qo, wname, R, C, w);
    if (q.status != RAD_OK) {
        CHECK_EQ(q.status, RAD_OK);
        return 0;
    }
    for (const Op& op : ops) {
        Selection s = select(q, R, C, op.roles);
        RadLayout L{};
        const int lrc = nf.layout(kp.data(), kp.n(), op.operand, &q.enc, s.sel.data(), s.t.data(),
                                  (int)s.t.size(), &L);
        std::vector<uint8_t> got;
        if (lrc == RAD_E_UNSUPPORTED) {
            /* stored as it is: the one plane, untouched */
            CHECK_EQ(op.roles.size(), (size_t)1);
            got = q.planes[(size_t)s.sel[0]];
        } else {
            CHECK_EQ(lrc, RAD_OK);
            CHECK(nf.relayout != nullptr);
            if (lrc != RAD_OK || !nf.relayout) {
                std::fprintf(stderr, "    %s operand %d: layout %d\n", label, op.operand, lrc);
                return 0;
            }
            got.assign((size_t)L.bytes, 0xA5);
            CHECK_EQ(nf.relayout(kp.data(), kp.n(), op.operand, &q.enc, s.sel.data(), s.t.data(),
                                 (int)s.t.size(), got.data(), L.bytes), RAD_OK);
        }
        h = fnv(got, h);

        /* ...and back, where the kernel can be read back at all */
        if (lrc == RAD_OK && nf.unrelayout) {
            std::vector<std::vector<uint8_t>> bufs;
            Selection back = select(q, R, C, op.roles, &bufs);
            CHECK_EQ(nf.unrelayout(kp.data(), kp.n(), op.operand, &q.enc, back.sel.data(),
                                   got.data(), (int64_t)got.size(), back.t.data(),
                                   (int)back.t.size()), RAD_OK);
            for (size_t i = 0; i < bufs.size(); ++i)
                CHECK(bufs[i] == q.planes[(size_t)s.sel[i]]);
        }
    }
    if (keep) *keep = q;
    return h;
}

/* A golden: the stored bytes' hash as the containers in service hold them. */
void golden(const char* label, uint64_t got, uint64_t want) {
    if (got != want)
        std::fprintf(stderr, "    %s: stored hash 0x%016llxull, recorded 0x%016llxull\n", label,
                     (unsigned long long)got, (unsigned long long)want);
    CHECK_EQ(got, want);
}

/* The edges every scale rule has a case for: a zero block, an all-positive one and an all-negative
 * one -- an asymmetric grid widens each to take in zero -- at blocks (0, 0), (1, 1) and (2, 0) of a
 * [br x bc] grid, clipped to the weight. */
void edges(std::vector<float>& w, int64_t R, int64_t C, int64_t br, int64_t bc) {
    auto block = [&](int64_t bi, int64_t bj, int sign) {
        const int64_t r0 = bi * br, c0 = (bj * bc < C) ? bj * bc : 0;
        for (int64_t r = r0; r < r0 + br && r < R; ++r)
            for (int64_t c = c0; c < c0 + bc && c < C; ++c) {
                float& v = w[(size_t)(r * C + c)];
                v = sign == 0 ? 0.0f : (float)sign * std::fabs(v);
            }
    };
    block(0, 0, 0);
    block(1, 1, 1);
    block(2, 0, -1);
}

std::vector<float> weight(int64_t R, int64_t C, uint64_t seed, int64_t br, int64_t bc) {
    std::vector<float> w = draw(R * C, seed, 20.0f);
    edges(w, R, C, br, bc);
    return w;
}

const Hooks nFp8Frag   { r4d_rad_layout_fp8_block, r4d_rad_relayout_fp8_block,
                       r4d_rad_unrelayout_fp8_block };
const Hooks nFp8Rows   { r4d_rad_layout_fp8_rowmajor, nullptr, nullptr };
const Hooks nFp8Lm     { r4d_rad_layout_fp8_logits, r4d_rad_relayout_fp8_logits,
                       r4d_rad_unrelayout_fp8_logits };
const Hooks nHc        { r4d_rad_layout_hc_e4m3, r4d_rad_relayout_hc_e4m3,
                       r4d_rad_unrelayout_hc_e4m3 };
const Hooks nMtp       { r4d_rad_layout_mtp_e4m3, r4d_rad_relayout_mtp_e4m3,
                       r4d_rad_unrelayout_mtp_e4m3 };
const Hooks nW4        { r4d_rad_layout_w4, r4d_rad_relayout_w4, r4d_rad_unrelayout_w4 };
const Hooks nW8a16     { r4d_rad_layout_w8a16, r4d_rad_relayout_w8a16, nullptr };
const Hooks nW8a8      { r4d_rad_layout_w8a8, r4d_rad_relayout_w8a8, nullptr };
const Hooks nW2        { r4d_rad_layout_w2a8, r4d_rad_relayout_w2a8, r4d_rad_unrelayout_w2a8 };
const Hooks nMx        { r4d_rad_layout_mxfp4, r4d_rad_relayout_mxfp4, nullptr };
const Hooks nMoeW4     { r4d_rad_layout_moe_w4, r4d_rad_relayout_moe_w4,
                       r4d_rad_unrelayout_moe_w4 };
const Hooks nMoeW4Mod4 { r4d_rad_layout_moe_w4_mod4, r4d_rad_relayout_moe_w4_mod4,
                       r4d_rad_unrelayout_moe_w4_mod4 };

/* gemm_nt_q's operands and the other ops' weight slots, as the kernels number them */
enum { B = 2, BSCALE = 3, BREF = 6, LM_W = 1, HC_DOWN = 2, HC_UP = 3, MTP_FCH = 4, MTP_FCE = 5 };

Opts fp8_grid() {
    Opts o;
    o.s("codes", "fp8_e4m3").s("block", "128x128").s("scale", "bf16");
    return o;
}

}  /* namespace */

TEST(block_fp8_in_fragment_order_and_row_major) {
    NEED_LIBQUANT();
    {
        const int64_t N = 272, K = 384;                  /* a short last row block */
        const std::vector<float> w = weight(N, K, 11, 128, 128);
        Opts kp;
        kp.i("N", N).i("K", K);
        golden("fp8 frag", stored("fp8 frag", nFp8Frag, kp, "rtn", fp8_grid(),
                                  "blk.0.attn_q.weight", N, K, w,
                                  { { B, { "codes" } }, { BSCALE, { "scale" } } }),
               0xd60d2b6299d47a63ull);
    }
    {
        const int64_t N = 400, K = 320;                  /* short blocks on both axes */
        const std::vector<float> w = weight(N, K, 12, 128, 128);
        Opts kp;
        kp.i("N", N).i("K", K);
        golden("fp8 rows", stored("fp8 rows", nFp8Rows, kp, "rtn", fp8_grid(),
                                  "blk.0.attn_o.weight", N, K, w,
                                  { { B, { "codes" } }, { BSCALE, { "scale" } } }),
               0xe90f8e25169cc5e5ull);
    }
}

TEST(the_fp8_lm_head_carries_its_block_scales_in_each_row) {
    NEED_LIBQUANT();
    const int64_t V = 300, E = 384;
    const std::vector<float> w = weight(V, E, 13, 128, 128);
    Opts kp;
    kp.i("n_vocab", V).i("n_embd", E);
    golden("fp8 lm", stored("fp8 lm", nFp8Lm, kp, "rtn", fp8_grid(), "output.weight", V, E,
                            w, { { LM_W, { "codes", "scale" } } }),
           0x4a710b4ca7fa4db0ull);
}

TEST(the_fp8_hyper_connection_and_mtp_rows) {
    NEED_LIBQUANT();
    const int64_t n = 128, hc = 4, lr = 256;
    Opts kp;
    kp.i("n", n).i("hc", hc).i("lowrank", lr);
    {
        /* mix_down: [lowrank, hc*n], 128 columns a scale */
        const std::vector<float> w = weight(lr, hc * n, 14, 1, 128);
        Opts g;
        g.s("codes", "fp8_e4m3").i("group", 128).s("scale", "f32");
        golden("hc down", stored("hc down", nHc, kp, "rtn", g, "blk.0.hc_down.weight", lr,
                                 hc * n, w, { { HC_DOWN, { "codes", "scale" } } }),
               0x62a46db4dbf831aeull);
    }
    {
        /* mix_up: [hc*n, lowrank], a quarter of the row a scale */
        const std::vector<float> w = weight(hc * n, lr, 15, 1, lr / 4);
        Opts g;
        g.s("codes", "fp8_e4m3").i("group", lr / 4).s("scale", "f32");
        golden("hc up", stored("hc up", nHc, kp, "rtn", g, "blk.0.hc_up.weight", hc * n, lr,
                               w, { { HC_UP, { "codes", "scale" } } }),
               0x764b1ae9ee02e27full);
    }
    {
        const int64_t m = 256;
        Opts mp;
        mp.i("n", m);
        const std::vector<float> wh = weight(m, m, 16, 1, 128);
        const std::vector<float> we = weight(m, m, 116, 1, 128);
        Opts g;
        g.s("codes", "fp8_e4m3").i("group", 128).s("scale", "f32");
        golden("mtp", chain(stored("mtp fc_hidden", nMtp, mp, "rtn", g,
                                   "mtp.fc_hidden.weight", m, m, wh,
                                   { { MTP_FCH, { "codes", "scale" } } }),
                            stored("mtp fc_embedding", nMtp, mp, "rtn", g,
                                   "mtp.fc_embd.weight", m, m, we,
                                   { { MTP_FCE, { "codes", "scale" } } })),
               0x5fb1a57420d4471dull);
    }
}

TEST(dense_int4_symmetric_and_asymmetric) {
    NEED_LIBQUANT();
    const int64_t N = 64, K = 256;
    const std::vector<float> w = weight(N, K, 17, 1, 128);
    {
        Opts kp, g;
        kp.i("N", N).i("K", K).s("dtype", "w4a8");
        g.s("codes", "i4").i("group", 128).s("scale", "f16");
        golden("w4", stored("w4", nW4, kp, "rtn", g, "blk.0.ffn_up.weight", N, K, w,
                            { { B, { "codes" } }, { BSCALE, { "scale" } } }),
               0x289633b39d08e864ull);
    }
    for (const char* z : { "u4", "u8" }) {
        Opts kp, g;
        kp.i("N", N).i("K", K).s("dtype", "w4a8_asym");
        g.s("codes", "u4").s("zero", z).i("group", 128).s("scale", "f16");
        golden("w4 asym", stored("w4 asym", nW4, kp, "rtn", g, "blk.0.ffn_up.weight", N, K, w,
                                 { { B, { "codes" } }, { BSCALE, { "scale", "zero" } } }),
               0x5854c2996f14a80eull);
    }
}

TEST(dense_int8_for_the_f16_and_the_int8_kernel) {
    NEED_LIBQUANT();
    const int64_t N = 48, K = 256;
    const std::vector<float> w = weight(N, K, 18, 1, 128);
    Opts kp, g;
    kp.i("N", N).i("K", K);
    g.s("codes", "i8").i("group", 128).s("scale", "f16").s("clamp", "sym");
    golden("w8a16", stored("w8a16", nW8a16, kp, "rtn", g, "blk.0.ffn_up.weight", N, K, w,
                           { { B, { "codes" } }, { BSCALE, { "scale" } } }),
           0x61e5176f24b6d72bull);
    golden("w8a8", stored("w8a8", nW8a8, kp, "rtn", g, "blk.0.ffn_up.weight", N, K, w,
                          { { B, { "codes" } }, { BSCALE, { "scale" } } }),
           0x7c2b502335fe84bbull);
}

TEST(two_bit_asymmetric) {
    NEED_LIBQUANT();
    const int64_t N = 32, K = 256;
    const std::vector<float> w = weight(N, K, 19, 1, 128);
    Opts kp, g;
    kp.i("N", N).i("K", K);
    g.s("codes", "u2").s("zero", "u8").i("group", 128).s("scale", "f16");
    golden("w2a8", stored("w2a8", nW2, kp, "rtn", g, "output.weight", N, K, w,
                          { { B, { "codes" } }, { BSCALE, { "scale", "zero" } } }),
           0x3bfb973c838751c2ull);
}

TEST(mxfp4_codes_exponents_and_the_row_reference) {
    NEED_LIBQUANT();
    const int64_t N = 32, K = 256;
    const std::vector<float> w = weight(N, K, 20, 1, 32);
    Opts kp, g;
    kp.i("N", N).i("K", K);
    g.s("codes", "fp4_e2m1").i("group", 32).s("scale", "e8m0");
    golden("mxfp4", stored("mxfp4", nMx, kp, "rtn", g, "blk.0.ffn_up.weight", N, K, w,
                           { { B, { "codes" } }, { BSCALE, { "scale" } }, { BREF, { "scale" } } }),
           0xc35d26b146836d77ull);
}

TEST(moe_int4_experts_plain_and_rotated_both_classes) {
    NEED_LIBQUANT();
    const int64_t N = 64, K = 256, N1 = 48, K1 = 128;
    const std::vector<float> w = weight(N, K, 21, 1, 128);
    const std::vector<float> w1 = weight(N1, K1, 22, 1, 128);
    for (const char* dt : { "w4a8", "w4a8h" }) {
        const bool rot = std::strcmp(dt, "w4a8h") == 0;
        Opts kp, g;
        kp.i("N", N).i("K", K).i("N_odd", N1).i("K_odd", K1).s("dtype", dt);
        g.s("codes", "i4").i("group", 128).s("scale", "bf16");
        if (rot) g.s("transform", "fwht128");
        const uint64_t h = chain(
            stored(dt, nMoeW4, kp, "rtn", g, "blk.0.ffn_down_exps.0.weight", N, K, w,
                   { { B, { "codes" } }, { BSCALE, { "scale" } } }),
            stored(dt, nMoeW4, kp, "rtn", g, "blk.0.ffn_down_exps.1.weight", N1, K1, w1,
                   { { 6, { "codes" } }, { 7, { "scale" } } }));
        golden(dt, h, rot ? 0xcc329df22f1f6c40ull : 0x15a0093ea83a5a90ull);
    }
}

TEST(moe_codebook_experts_both_classes) {
    NEED_LIBQUANT();
    /* w4nla8h: the rotated plane whose codes index libquant's w4nl table. The codes view takes the
     * table beside the codes, the relayout checks it against the kernel's and stores the codes
     * exactly as w4a8h stores its own, and the inverse gives back both planes. */
    const int64_t N = 64, K = 256, N1 = 48, K1 = 128;
    const std::vector<float> w = weight(N, K, 21, 1, 128);
    const std::vector<float> w1 = weight(N1, K1, 22, 1, 128);
    Opts kp, g;
    kp.i("N", N).i("K", K).i("N_odd", N1).i("K_odd", K1).s("dtype", "w4nla8h");
    g.s("table", "w4nl").i("group", 128).s("scale", "bf16").s("transform", "fwht128")
     .s("rule", "search");
    const uint64_t h = chain(
        stored("w4nla8h", nMoeW4, kp, "rtn", g, "blk.0.ffn_down_exps.0.weight", N, K, w,
               { { B, { "codes", "table" } }, { BSCALE, { "scale" } } }),
        stored("w4nla8h", nMoeW4, kp, "rtn", g, "blk.0.ffn_down_exps.1.weight", N1, K1, w1,
               { { 6, { "codes", "table" } }, { 7, { "scale" } } }));
    golden("w4nla8h", h, 0xbb9b202196aef720ull);
}

TEST(moe_codebook_experts_at_e4m3_scales_a_64) {
    NEED_LIBQUANT();
    /* w4nl64a8h: the codebook's codes, stored as w4nla8h stores them, and an E4M3 scale a
     * (row, 64 of K) under one fixed f32 -- the scale view selects both, the relayout checks the
     * second is the kernel's constant and stores the E4M3 plane as it is, and the inverse gives
     * back both. A second level of any other value is refused when its bytes are relaid. */
    const int64_t N = 64, K = 256, N1 = 48, K1 = 128;
    const std::vector<float> w = weight(N, K, 21, 1, 128);
    const std::vector<float> w1 = weight(N1, K1, 22, 1, 128);
    Opts kp, g;
    kp.i("N", N).i("K", K).i("N_odd", N1).i("K_odd", K1).s("dtype", "w4nl64a8h").i("group", 128);
    g.s("table", "w4nl").i("group", 64).s("scale", "fp8_e4m3").s("scale2", "f32")
     .s("block2", "*x*").f("scale2_value", 0x1p-13).s("transform", "fwht128").s("rule", "search");
    const uint64_t h = chain(
        stored("w4nl64a8h", nMoeW4, kp, "rtn", g, "blk.0.ffn_down_exps.0.weight", N, K, w,
               { { B, { "codes", "table" } }, { BSCALE, { "scale", "scale.1" } } }),
        stored("w4nl64a8h", nMoeW4, kp, "rtn", g, "blk.0.ffn_down_exps.1.weight", N1, K1, w1,
               { { 6, { "codes", "table" } }, { 7, { "scale", "scale.1" } } }));
    golden("w4nl64a8h", h, 0x528e273bdee3be1dull);

    Opts g2;
    g2.s("table", "w4nl").i("group", 64).s("scale", "fp8_e4m3").s("scale2", "f32")
      .s("block2", "*x*").f("scale2_value", 0x1p-12).s("transform", "fwht128").s("rule", "search");
    const Quantised q2 = quantise(quantizer("rtn"), g2, "blk.0.ffn_down_exps.0.weight", N, K, w);
    REQUIRE_EQ(q2.status, RAD_OK);
    Selection s = select(q2, N, K, { "scale", "scale.1" });
    RadLayout L{};
    REQUIRE_EQ(r4d_rad_layout_moe_w4(kp.data(), kp.n(), BSCALE, &q2.enc, s.sel.data(), s.t.data(),
                                     (int)s.t.size(), &L), RAD_OK);
    std::vector<uint8_t> dst((size_t)L.bytes);
    CHECK_EQ(r4d_rad_relayout_moe_w4(kp.data(), kp.n(), BSCALE, &q2.enc, s.sel.data(), s.t.data(),
                                     (int)s.t.size(), dst.data(), L.bytes), RAD_E_DTYPE);
    /* and the bf16-scaled codebook's encoding is not this dtype's */
    Opts g3;
    g3.s("table", "w4nl").i("group", 128).s("scale", "bf16").s("transform", "fwht128");
    const Quantised q3 = quantise(quantizer("rtn"), g3, "blk.0.ffn_down_exps.0.weight", N, K, w);
    REQUIRE_EQ(q3.status, RAD_OK);
    Selection s3 = select(q3, N, K, { "codes", "table" });
    CHECK_EQ(r4d_rad_layout_moe_w4(kp.data(), kp.n(), B, &q3.enc, s3.sel.data(), s3.t.data(),
                                   (int)s3.t.size(), &L), RAD_E_DTYPE);
}

TEST(moe_codebook_experts_at_e4m3_scales_a_32) {
    NEED_LIBQUANT();
    /* w4nl32a8h: the 64-wide form's planes at a scale a 32 of K, four E4M3 bytes a 128. */
    const int64_t N = 64, K = 256, N1 = 48, K1 = 128;
    const std::vector<float> w = weight(N, K, 21, 1, 128);
    const std::vector<float> w1 = weight(N1, K1, 22, 1, 128);
    Opts kp, g;
    kp.i("N", N).i("K", K).i("N_odd", N1).i("K_odd", K1).s("dtype", "w4nl32a8h").i("group", 128);
    g.s("table", "w4nl").i("group", 32).s("scale", "fp8_e4m3").s("scale2", "f32")
     .s("block2", "*x*").f("scale2_value", 0x1p-13).s("transform", "fwht128").s("rule", "search");
    const uint64_t h = chain(
        stored("w4nl32a8h", nMoeW4, kp, "rtn", g, "blk.0.ffn_down_exps.0.weight", N, K, w,
               { { B, { "codes", "table" } }, { BSCALE, { "scale", "scale.1" } } }),
        stored("w4nl32a8h", nMoeW4, kp, "rtn", g, "blk.0.ffn_down_exps.1.weight", N1, K1, w1,
               { { 6, { "codes", "table" } }, { 7, { "scale", "scale.1" } } }));
    golden("w4nl32a8h", h, 0xf5ce1cb0bac69211ull);
    /* and the 64-wide encoding is not this dtype's */
    Opts g64;
    g64.s("table", "w4nl").i("group", 64).s("scale", "fp8_e4m3").s("scale2", "f32")
       .s("block2", "*x*").f("scale2_value", 0x1p-13).s("transform", "fwht128");
    const Quantised q = quantise(quantizer("rtn"), g64, "blk.0.ffn_down_exps.0.weight", N, K, w);
    REQUIRE_EQ(q.status, RAD_OK);
    Selection s = select(q, N, K, { "codes", "table" });
    RadLayout L{};
    CHECK_EQ(r4d_rad_layout_moe_w4(kp.data(), kp.n(), B, &q.enc, s.sel.data(), s.t.data(),
                                   (int)s.t.size(), &L), RAD_E_DTYPE);
}

TEST(moe_codebook_experts_at_five_bits) {
    NEED_LIBQUANT();
    /* w5nl64a8h: u5 codes into libquant's 32-entry w5nl table at w4nl64a8h's scales. The codes
     * view relays each row into its nibbles and its sign bits, the scale view is the 64-wide
     * form's, and the inverse gives back the packed codes and the table. A table that is not the
     * kernel's is refused when its bytes are relaid, and the four-bit encoding is not this
     * dtype's. */
    const int64_t N = 64, K = 256, N1 = 48, K1 = 128;
    const std::vector<float> w = weight(N, K, 21, 1, 128);
    const std::vector<float> w1 = weight(N1, K1, 22, 1, 128);
    Opts kp, g;
    kp.i("N", N).i("K", K).i("N_odd", N1).i("K_odd", K1).s("dtype", "w5nl64a8h").i("group", 128);
    g.s("table", "w5nl").i("group", 64).s("scale", "fp8_e4m3").s("scale2", "f32")
     .s("block2", "*x*").f("scale2_value", 0x1p-13).s("transform", "fwht128").s("rule", "search");
    const uint64_t h = chain(
        stored("w5nl64a8h", nMoeW4, kp, "rtn", g, "blk.0.ffn_down_exps.0.weight", N, K, w,
               { { B, { "codes", "table" } }, { BSCALE, { "scale", "scale.1" } } }),
        stored("w5nl64a8h", nMoeW4, kp, "rtn", g, "blk.0.ffn_down_exps.1.weight", N1, K1, w1,
               { { 6, { "codes", "table" } }, { 7, { "scale", "scale.1" } } }));
    golden("w5nl64a8h", h, 0xd773da64690cb3f6ull);

    /* the same 32 levels with one moved: the kernel's table is the only one it decodes */
    Opts g2;
    g2.s("table", "0.8125;2.5;4;5.5;7.5;9;11;12;14;16;18;20;22;24;28;30;-0.8125;-2.5;-4;-5.5;"
                  "-7.5;-9;-11;-12;-14;-16;-18;-20;-22;-24;-28;-30")
      .i("group", 64).s("scale", "fp8_e4m3").s("scale2", "f32").s("block2", "*x*")
      .f("scale2_value", 0x1p-13).s("transform", "fwht128");
    const Quantised q2 = quantise(quantizer("rtn"), g2, "blk.0.ffn_down_exps.0.weight", N, K, w);
    REQUIRE_EQ(q2.status, RAD_OK);
    Selection s2 = select(q2, N, K, { "codes", "table" });
    RadLayout L{};
    REQUIRE_EQ(r4d_rad_layout_moe_w4(kp.data(), kp.n(), B, &q2.enc, s2.sel.data(), s2.t.data(),
                                     (int)s2.t.size(), &L), RAD_OK);
    CHECK_EQ(L.bytes, N * K / 8 * 5);
    std::vector<uint8_t> dst((size_t)L.bytes);
    CHECK_EQ(r4d_rad_relayout_moe_w4(kp.data(), kp.n(), B, &q2.enc, s2.sel.data(), s2.t.data(),
                                     (int)s2.t.size(), dst.data(), L.bytes), RAD_E_DTYPE);
    /* and the four-bit codebook's encoding is not this dtype's */
    Opts g4;
    g4.s("table", "w4nl").i("group", 64).s("scale", "fp8_e4m3").s("scale2", "f32")
      .s("block2", "*x*").f("scale2_value", 0x1p-13).s("transform", "fwht128");
    const Quantised q4 = quantise(quantizer("rtn"), g4, "blk.0.ffn_down_exps.0.weight", N, K, w);
    REQUIRE_EQ(q4.status, RAD_OK);
    Selection s4 = select(q4, N, K, { "codes", "table" });
    CHECK_EQ(r4d_rad_layout_moe_w4(kp.data(), kp.n(), B, &q4.enc, s4.sel.data(), s4.t.data(),
                                   (int)s4.t.size(), &L), RAD_E_DTYPE);
}

namespace {

/* WHAT THE OP ORACLE READS FOR A RE-LAID WEIGHT, got the way Ctx::oracle_unrelayout gets it: the
 * stored bytes back through the kernel's own inverse into the selected planes, and those through
 * the core's enc_selection_logical into the values a dense reference multiplies -- f32 here, held
 * to the same values read straight off the quantiser's planes: a code as itself or as its table
 * entry, a scale times the levels above it. A selection the core cannot read on its own is a
 * weight the oracle has to decline, and the E4M3 scale views were one. A view stored as it is has
 * nothing to invert and is handed over unchanged. */
void oracle_reads(const char* label, const Hooks& nf, Opts kp, int operand, const Quantised& q,
                  int64_t R, int64_t C, const std::vector<const char*>& roles) {
    Selection s = select(q, R, C, roles);
    RadLayout L{};
    const int lrc = nf.layout(kp.data(), kp.n(), operand, &q.enc, s.sel.data(), s.t.data(),
                              (int)s.t.size(), &L);
    if (lrc == RAD_E_UNSUPPORTED) return;
    REQUIRE_EQ(lrc, RAD_OK);
    std::vector<uint8_t> laid((size_t)L.bytes);
    REQUIRE_EQ(nf.relayout(kp.data(), kp.n(), operand, &q.enc, s.sel.data(), s.t.data(),
                           (int)s.t.size(), laid.data(), L.bytes), RAD_OK);
    std::vector<std::vector<uint8_t>> bufs;
    Selection back = select(q, R, C, roles, &bufs);
    REQUIRE_EQ(nf.unrelayout(kp.data(), kp.n(), operand, &q.enc, back.sel.data(), laid.data(),
                             L.bytes, back.t.data(), (int)back.t.size()), RAD_OK);
    std::vector<int64_t> rows, cols;
    std::vector<const void*> pd;
    for (size_t k = 0; k < back.t.size(); ++k) {
        rows.push_back(back.t[k].shape[0]);
        cols.push_back(back.t[k].shape[1]);
        pd.push_back(bufs[k].data());
    }
    const int64_t n = rows[0] * cols[0];
    std::vector<float> got((size_t)n);
    std::string why;
    const int rc = rad::enc_selection_logical(q.enc, back.sel.data(), (int)back.sel.size(),
                                              rows.data(), cols.data(), pd.data(), RAD_F32, n,
                                              got.data(), &why);
    if (rc != RAD_OK)
        std::fprintf(stderr, "    %s operand %d: %s\n", label, operand, why.c_str());
    REQUIRE_EQ(rc, RAD_OK);
    const RadEncPlane& p0 = q.enc.plane[s.sel[0]];
    const uint8_t* src = q.planes[(size_t)s.sel[0]].data();
    const int64_t rb = rad_enc_row_bytes(p0.dtype, cols[0]);
    const float* table = nullptr;
    float level = 1.0f;
    for (size_t k = 1; k < roles.size(); ++k) {
        const std::vector<uint8_t>& pk = q.planes[(size_t)s.sel[k]];
        if (!std::strcmp(roles[k], "table")) table = (const float*)pk.data();
        else level *= rad_load_f32(pk.data(), q.enc.plane[s.sel[k]].dtype, 0);
    }
    int64_t bad = 0;
    for (int64_t i = 0; i < rows[0]; ++i)
        for (int64_t j = 0; j < cols[0]; ++j) {
            float v = rad_load_f32(src + i * rb, p0.dtype, j);
            if (table) v = table[(int64_t)v];
            if (got[(size_t)(i * cols[0] + j)] != v * level) ++bad;
        }
    if (bad) std::fprintf(stderr, "    %s operand %d: %lld of %lld read wrong\n", label, operand,
                          (long long)bad, (long long)n);
    CHECK_EQ(bad, (int64_t)0);
}

/* A four-bit expert format as the quantiser is asked for it, and its two views. */
struct MoeFmt {
    const char* dtype;
    Opts quant;
    std::vector<const char*> codes, scales;
};
std::vector<MoeFmt> moe_formats() {
    auto nl = [](const char* table, long long group) {
        Opts g;
        g.s("table", table).i("group", group).s("scale", "fp8_e4m3").s("scale2", "f32")
         .s("block2", "*x*").f("scale2_value", 0x1p-13).s("transform", "fwht128")
         .s("rule", "search");
        return g;
    };
    Opts i4, nla;
    i4.s("codes", "i4").i("group", 128).s("scale", "bf16").s("transform", "fwht128");
    nla.s("table", "w4nl").i("group", 128).s("scale", "bf16").s("transform", "fwht128")
       .s("rule", "search");
    return {
        { "w4a8h", i4, { "codes" }, { "scale" } },
        { "w4nla8h", nla, { "codes", "table" }, { "scale" } },
        { "w4nl64a8h", nl("w4nl", 64), { "codes", "table" }, { "scale", "scale.1" } },
        { "w4nl32a8h", nl("w4nl", 32), { "codes", "table" }, { "scale", "scale.1" } },
        { "w5nl64a8h", nl("w5nl", 64), { "codes", "table" }, { "scale", "scale.1" } },
    };
}

}  /* namespace */

/* THE FOUR CLASSES ARE THE TWO-CLASS HOOKS ONE CLASS FURTHER ALONG. moe_gemm_q_mod4's class q is
 * operands 2/3 at N/K for q = 0 and 2q+4/2q+5 at N_q/K_q after it, and each is stored, and read
 * back, exactly as moe_gemm_q's hooks store an expert of that geometry -- codes and scales, in
 * every four-bit format, at four class geometries. Then what the op oracle reads through each
 * class's inverse, codes and E4M3 scale views alike, and through the two-class op's own. */
TEST(moe_four_classes_store_and_read_back_as_two_classes_do) {
    NEED_LIBQUANT();
    const int64_t Nq[4] = { 64, 128, 64, 48 }, Kq[4] = { 256, 256, 128, 128 };
    for (MoeFmt& f : moe_formats()) {
        Opts kp4;
        kp4.i("N", Nq[0]).i("K", Kq[0]).i("N_1", Nq[1]).i("K_1", Kq[1]).i("N_2", Nq[2])
           .i("K_2", Kq[2]).i("N_3", Nq[3]).i("K_3", Kq[3]).s("dtype", f.dtype).i("group", 128);
        for (int q = 0; q < 4; ++q) {
            const std::vector<float> w = weight(Nq[q], Kq[q], 30 + q, 1, 128);
            const int cop = q == 0 ? B : 2 * q + 4, sop = cop + 1;
            Opts kp2;
            kp2.i("N", Nq[q]).i("K", Kq[q]).s("dtype", f.dtype).i("group", 128);
            const std::string label = std::string(f.dtype) + " class " + std::to_string(q);
            Quantised kept;
            const uint64_t h4 = stored(label.c_str(), nMoeW4Mod4, kp4, "rtn", f.quant,
                                       "blk.0.ffn_down_exps.0.weight", Nq[q], Kq[q], w,
                                       { { cop, f.codes }, { sop, f.scales } }, &kept);
            const uint64_t h2 = stored(label.c_str(), nMoeW4, kp2, "rtn", f.quant,
                                       "blk.0.ffn_down_exps.0.weight", Nq[q], Kq[q], w,
                                       { { B, f.codes }, { BSCALE, f.scales } });
            if (h4 != h2)
                std::fprintf(stderr, "    %s: four-class hash 0x%016llx, two-class 0x%016llx\n",
                             label.c_str(), (unsigned long long)h4, (unsigned long long)h2);
            CHECK_EQ(h4, h2);
            REQUIRE_EQ(kept.status, RAD_OK);
            oracle_reads(label.c_str(), nMoeW4Mod4, kp4, cop, kept, Nq[q], Kq[q], f.codes);
            oracle_reads(label.c_str(), nMoeW4Mod4, kp4, sop, kept, Nq[q], Kq[q], f.scales);
            oracle_reads(label.c_str(), nMoeW4, kp2, B, kept, Nq[q], Kq[q], f.codes);
            oracle_reads(label.c_str(), nMoeW4, kp2, BSCALE, kept, Nq[q], Kq[q], f.scales);
            /* A class reads its own geometry and nobody else's: class 1 is 128 wide, so an
             * expert of any other class's width is refused there. */
            if (q != 1) {
                Selection s = select(kept, Nq[q], Kq[q], f.codes);
                RadLayout L{};
                CHECK_EQ(nMoeW4Mod4.layout(kp4.data(), kp4.n(), 6, &kept.enc, s.sel.data(),
                                           s.t.data(), (int)s.t.size(), &L), RAD_E_SHAPE);
            }
        }
    }
}

TEST(moe_int4_gptq_against_a_calibration_hessian) {
    NEED_LIBQUANT();
    /* Correlated activations, so the Hessian is far from diagonal and the error feedback moves
     * codes rounding alone would not -- checked below, or a GPTQ that silently fell back to
     * rounding would pass as parity. */
    const int64_t N = 64, K = 256, T = 1024;
    const std::vector<float> w = weight(N, K, 23, 1, 128);
    const std::vector<float> z = draw(T * K, 24);
    const std::vector<float> mix = draw(K * K, 25);
    std::vector<float> x((size_t)(T * K), 0.0f);
    for (int64_t t = 0; t < T; ++t)
        for (int64_t i = 0; i < K; ++i) {
            double s = z[(size_t)(t * K + i)] * 4.0;
            for (int64_t j = 0; j < 8; ++j)
                s += mix[(size_t)(i * K + j)] * z[(size_t)(t * K + j)] * 40.0;
            x[(size_t)(t * K + i)] = (float)s;
        }
    std::vector<float> h((size_t)(K * K), 0.0f);
    for (int64_t t = 0; t < T; ++t)
        for (int64_t i = 0; i < K; ++i)
            for (int64_t j = 0; j < K; ++j)
                h[(size_t)(i * K + j)] += x[(size_t)(t * K + i)] * x[(size_t)(t * K + j)];
    char dir[] = "/tmp/rad_relayout_parity_XXXXXX";
    REQUIRE(mkdtemp(dir) != nullptr);
    const std::string path = std::string(dir) + "/gram.r0.blk.5.ffn_down_exps.bin";
    /* The Gram of `x` as it stands -- the model's own domain -- so the rotated recipe below reads it
     * rotated into its own, as it would a bf16 model's calibration. */
    REQUIRE(write_gram(path, K, T, h, 0));

    const char* name = "blk.5.ffn_down_exps.9.weight";
    for (const char* dt : { "w4a8", "w4a8h" }) {
        const bool rot = std::strcmp(dt, "w4a8h") == 0;
        Opts kp, g, r;
        kp.i("N", N).i("K", K).s("dtype", dt).s("calib", dir).s("weight", name);
        g.s("codes", "i4").i("group", 128).s("scale", "bf16").s("calib", dir);
        r.s("codes", "i4").i("group", 128).s("scale", "bf16");
        if (rot) { g.s("transform", "fwht128"); r.s("transform", "fwht128"); }
        Quantised qg, qr;
        const uint64_t hg = stored(dt, nMoeW4, kp, "gptq", g, name, N, K, w,
                                   { { B, { "codes" } }, { BSCALE, { "scale" } } }, &qg);
        qr = quantise(quantizer("rtn"), r, name, N, K, w);
        REQUIRE_EQ(qr.status, RAD_OK);
        CHECK(qg.planes[0] != qr.planes[0]);           /* the Hessian moved codes */
        CHECK(qg.planes[1] == qr.planes[1]);           /* and not the scales: static groups */
        golden(rot ? "gptq w4a8h" : "gptq w4a8", hg,
               rot ? 0x761cbdd0aff4adedull : 0x4a4db9378ca02303ull);
    }
    std::remove(path.c_str());
    rmdir(dir);
}

TEST(the_ngram_table_is_the_canonical_plane) {
    NEED_LIBQUANT();
    void* so = dlopen((home() + "/kernels/libavx.so").c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!so) {
        std::fprintf(stderr, "  SKIP no libavx under %s\n", home().c_str());
        return;
    }
    Hooks nf;
    nf.layout = (RadLayoutFn)dlsym(so, "avx_layout_ngram");
    REQUIRE(nf.layout != nullptr);
    const int64_t V = 100, E = 64;
    const double wscale = 1.99317932128906e-4;
    const std::vector<float> w = draw(V * E, 26, 20.0f);
    Opts kp, g;
    kp.i("n_vocab", V).i("n_embd", E);
    g.s("codes", "fp8_e4m3").s("block", "*x*").s("scale", "bf16").s("rule", "fixed")
     .f("scale_value", wscale);
    golden("ngram", stored("ngram", nf, kp, "rtn", g, "ple.ngram.weight", V, E, w,
                           { { 1, { "codes" } }, { 2, { "scale" } } }),
           0xa5e81d9b9ff05c2aull);
    dlclose(so);
}

TEST(a_kernel_refuses_an_encoding_it_does_not_read) {
    NEED_LIBQUANT();
    const int64_t N = 64, K = 256;
    const std::vector<float> w = draw(N * K, 27);
    Opts i4, i4r, u4, f8;
    i4.s("codes", "i4").i("group", 128).s("scale", "bf16");
    i4r.s("codes", "i4").i("group", 128).s("scale", "bf16").s("transform", "fwht128");
    u4.s("codes", "u4").s("zero", "u4").i("group", 128).s("scale", "f16");
    f8 = fp8_grid();
    const Quantised q_i4 = quantise(quantizer("rtn"), i4, "x", N, K, w);
    const Quantised q_i4r = quantise(quantizer("rtn"), i4r, "x", N, K, w);
    const Quantised q_u4 = quantise(quantizer("rtn"), u4, "x", N, K, w);
    const Quantised q_f8 = quantise(quantizer("rtn"), f8, "x", N, K, w);
    Opts nl, nf4;
    nl.s("table", "w4nl").i("group", 128).s("scale", "bf16").s("transform", "fwht128");
    nf4.s("table", "nf4").i("group", 128).s("scale", "bf16").s("transform", "fwht128");
    const Quantised q_nl = quantise(quantizer("rtn"), nl, "x", N, K, w);
    const Quantised q_nf4 = quantise(quantizer("rtn"), nf4, "x", N, K, w);
    auto lay = [&](RadLayoutFn f, Opts kp, int operand, const Quantised& q,
                   std::vector<const char*> roles) {
        Selection s = select(q, N, K, roles);
        RadLayout L{};
        return f(kp.data(), kp.n(), operand, &q.enc, s.sel.data(), s.t.data(), (int)s.t.size(), &L);
    };
    Opts dense, plain, rot, sym, asym;
    dense.i("N", N).i("K", K);
    plain.i("N", N).i("K", K).s("dtype", "w4a8");
    rot.i("N", N).i("K", K).s("dtype", "w4a8h");
    sym.i("N", N).i("K", K).s("dtype", "w4a8");
    asym.i("N", N).i("K", K).s("dtype", "w4a8_asym");
    /* an fp8 GEMM handed int4 */
    CHECK_EQ(lay(r4d_rad_layout_fp8_block, dense, B, q_i4, { "codes" }), RAD_E_DTYPE);
    /* a rotated weight to the experts that do not rotate their activation, and the reverse */
    CHECK_EQ(lay(r4d_rad_layout_moe_w4, plain, B, q_i4r, { "codes" }), RAD_E_DTYPE);
    CHECK_EQ(lay(r4d_rad_layout_moe_w4, rot, B, q_i4, { "codes" }), RAD_E_DTYPE);
    CHECK_EQ(lay(r4d_rad_layout_moe_w4, rot, B, q_i4r, { "codes" }), RAD_OK);
    /* the symmetric dense kernel handed an asymmetric grid, and the reverse; and a bf16 scale */
    CHECK_EQ(lay(r4d_rad_layout_w4, sym, B, q_u4, { "codes" }), RAD_E_DTYPE);
    CHECK_EQ(lay(r4d_rad_layout_w4, asym, B, q_i4, { "codes" }), RAD_E_DTYPE);
    CHECK_EQ(lay(r4d_rad_layout_w4, sym, B, q_i4, { "codes" }), RAD_E_DTYPE);
    /* a block-fp8 weight to the int8 kernel */
    CHECK_EQ(lay(r4d_rad_layout_w8a8, dense, B, q_f8, { "codes" }), RAD_E_DTYPE);
    /* the right encoding with the wrong planes selected is a shape error, not a pass */
    CHECK_EQ(lay(r4d_rad_layout_fp8_block, dense, B, q_f8, { "scale" }), RAD_E_SHAPE);
    /* a codebook to the two's complement experts, and -8..7 to the codebook's; the codebook's codes
     * without their table */
    Opts nlk;
    nlk.i("N", N).i("K", K).s("dtype", "w4nla8h");
    CHECK_EQ(lay(r4d_rad_layout_moe_w4, rot, B, q_nl, { "codes" }), RAD_E_DTYPE);
    CHECK_EQ(lay(r4d_rad_layout_moe_w4, nlk, B, q_i4r, { "codes" }), RAD_E_DTYPE);
    CHECK_EQ(lay(r4d_rad_layout_moe_w4, nlk, B, q_nl, { "codes" }), RAD_E_SHAPE);
    CHECK_EQ(lay(r4d_rad_layout_moe_w4, nlk, B, q_nl, { "codes", "table" }), RAD_OK);
    /* ANOTHER TABLE OF THE SAME SHAPE declares -- an encoding cannot say what its table holds --
     * and is refused when its bytes are relaid, before the kernel could decode them as w4nl */
    CHECK_EQ(lay(r4d_rad_layout_moe_w4, nlk, B, q_nf4, { "codes", "table" }), RAD_OK);
    {
        Selection s = select(q_nf4, N, K, { "codes", "table" });
        RadLayout L{};
        REQUIRE_EQ(r4d_rad_layout_moe_w4(nlk.data(), nlk.n(), B, &q_nf4.enc, s.sel.data(),
                                         s.t.data(), (int)s.t.size(), &L), RAD_OK);
        std::vector<uint8_t> dst((size_t)L.bytes);
        CHECK_EQ(r4d_rad_relayout_moe_w4(nlk.data(), nlk.n(), B, &q_nf4.enc, s.sel.data(),
                                         s.t.data(), (int)s.t.size(), dst.data(), L.bytes),
                 RAD_E_DTYPE);
    }
}

RAD_TEST_MAIN()
