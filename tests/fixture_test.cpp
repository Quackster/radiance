/* fixture_test.cpp -- the recorded reference itself: the draw, the reduction and the file.
 *
 * EVERY CORRECTNESS NUMBER THIS PROJECT REPORTS ABOUT A KERNEL IS RELATIVE TO THIS CODE.
 * rad-kbench compares a kernel against an answer libref recorded at another time, possibly on
 * another machine, and the only reason that comparison means anything is that both sides drew
 * byte-identical inputs from a stored seed and reduced their outputs the same way. A drift in
 * either half does not report as a harness fault: it reports as a page of kernel failures, or --
 * much worse -- as agreement, because two sides that both produce zeros agree perfectly.
 *
 * So the cases here are about the properties the comparison RESTS on rather than about the
 * arithmetic:
 *
 *   - a weight's bytes depend on the operand's identity and not on the case, which is what lets
 *     an M sweep draw one weight once and is what makes two libraries comparable at all;
 *   - the blocked draw and the whole draw are one stream, because a large operand takes the first
 *     path and a small one the second and nothing else would notice they had parted;
 *   - the reduction is the same however many threads ran, or a recorded norm is machine-specific;
 *   - an input lands inside the domain its op is ever handed, or the comparison measures the
 *     condition number of the test;
 *   - every dtype a recorded case uses is one the harness can actually draw AND read back;
 *   - a weight's planes are drawn inside what each plane can hold, the reference reads what they
 *     decode to, and a kernel is handed what its own layout hooks make of them.
 */
#include "rad_test.h"
#include "kfixture.h"
#include "opshapes.h"
#include "rad_plugin.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <set>
#include <string>
#include <vector>
#include <unistd.h>

using namespace rad;

namespace {

std::string scratch_path(const char* leaf) {
    const char* dir = std::getenv("TMPDIR");
    return std::string(dir && *dir ? dir : "/tmp") + "/radiance_fixture_test_" + leaf;
}

/* An operand of `n` f32 elements laid out as `shape`, with the draw law the case is about. */
KfOperand opd_f32(std::vector<int64_t> shape, int role = RAD_OPD_IN, int fill = KF_FILL_NORMAL) {
    KfOperand d;
    d.role = role;
    d.dtype = RAD_F32;
    d.shape = std::move(shape);
    d.fill = fill;
    return d;
}

/* A weight operand of a block-fp8 [rows, cols] weight taking `roles`: one plane at its own
 * extents, several at the codes' dtype and the logical extents -- as a declaration records it. */
KfOperand fp8_weight(int64_t rows, int64_t cols, std::vector<const char*> roles) {
    KfOperand d;
    d.role = RAD_OPD_WEIGHT;
    d.has_enc = true;
    rad_enc_clear(&d.enc);
    rad_enc_copy_str(d.enc.scheme, "affine");
    rad_enc_add_plane(&d.enc, "codes", RAD_F8E4M3, 1, 1);
    rad_enc_add_plane(&d.enc, "scale", RAD_BF16, 128, 128);
    for (const char* r : roles) {
        const int i = rad_enc_find(&d.enc, r);
        int64_t pr = 0, pc = 0;
        rad_enc_plane_dims(&d.enc.plane[i], rows, cols, &pr, &pc);
        d.sel.push_back(i);
        d.sel_rows.push_back(pr);
        d.sel_cols.push_back(pc);
    }
    if (roles.size() == 1) {
        d.dtype = d.enc.plane[d.sel[0]].dtype;
        d.shape = { d.sel_rows[0], d.sel_cols[0] };
    } else {
        d.dtype = RAD_F32;
        d.shape = { rows, cols };
    }
    return d;
}

int64_t numel(const KfOperand& d) {
    int64_t n = 1;
    for (int64_t x : d.shape) n *= x;
    return n;
}

std::vector<float> draw_f32(const std::vector<KfOperand>& opd, size_t k, int64_t M, uint64_t seed) {
    std::vector<uint8_t> bytes;
    kf_draw(opd, k, M, kf_vocab_rows(opd), seed, &bytes);
    std::vector<float> out;
    rad_widen(bytes.data(), opd[k].dtype, numel(opd[k]), &out);
    return out;
}

}  /* namespace */

/* ================================================================== the draw's block contract */

/* RAD_FILL_BLOCK is documented as part of the contract rather than an implementation detail, and
 * this is what makes that true. kf_draw streams any operand past one block through
 * rad_fill_normal_block so the float intermediate never gets as big as the tensor; everything
 * smaller goes through rad_fill_normal in one pass. If those two ever stopped producing the same
 * stream, a weight would be drawn differently depending only on its size, silently. */
TEST(a_blocked_draw_and_a_whole_one_are_the_same_stream) {
    const int64_t n = RAD_FILL_BLOCK * 3 + 517;   /* three full blocks and a partial one */
    std::vector<float> whole((size_t)n, 0.0f);
    rad_fill_normal(whole.data(), n, 0xC0FFEEull, 0.02f);

    std::vector<float> piecewise((size_t)n, 0.0f);
    for (int64_t lo = 0; lo < n; lo += RAD_FILL_BLOCK) {
        const int64_t len = (n - lo) < RAD_FILL_BLOCK ? (n - lo) : RAD_FILL_BLOCK;
        rad_fill_normal_block(piecewise.data() + lo, len, 0xC0FFEEull, lo / RAD_FILL_BLOCK, 0.02f);
    }
    CHECK_EQ(std::memcmp(whole.data(), piecewise.data(), (size_t)n * sizeof(float)), 0);
}

/* The threaded path is taken whenever the operand spans more than one block, and the work is
 * handed out round-robin over however many cores the machine has. A draw that moved with the core
 * count would make every recorded reference machine-specific. */
TEST(a_draw_does_not_change_between_runs) {
    const int64_t n = RAD_FILL_BLOCK * 5;
    std::vector<float> a((size_t)n, 0.0f), b((size_t)n, 0.0f);
    rad_fill_normal(a.data(), n, 12345ull, 1.0f);
    rad_fill_normal(b.data(), n, 12345ull, 1.0f);
    CHECK_EQ(std::memcmp(a.data(), b.data(), (size_t)n * sizeof(float)), 0);
    /* And that the fill is actually a unit-variance normal rather than, say, zeros -- a draw that
     * silently produced nothing would satisfy every equality above. */
    double ss = 0;
    for (int64_t i = 0; i < n; ++i) ss += (double)a[(size_t)i] * (double)a[(size_t)i];
    CHECK_NEAR(std::sqrt(ss / (double)n), 1.0, 0.02);
}

TEST(a_case_seed_separates_the_op_the_band_the_domain_and_the_operand) {
    const uint64_t base = 7;
    const uint64_t s = rad_case_seed(base, "rmsnorm", 0, 0, 0);
    CHECK(s != rad_case_seed(base, "softmax", 0, 0, 0));
    CHECK(s != rad_case_seed(base, "rmsnorm", 1, 0, 0));
    CHECK(s != rad_case_seed(base, "rmsnorm", 0, 1, 0));
    CHECK(s != rad_case_seed(base, "rmsnorm", 0, 0, 1));
    CHECK_EQ(s, rad_case_seed(base, "rmsnorm", 0, 0, 0));
    CHECK(rad_case_seed(base, "rmsnorm", 0, 0, 0) != 0);
}

/* ================================================================== narrow and widen */

/* THE PAIR HAS TO BE COMPLETE AT BOTH ENDS. rad_narrow writes the operand a kernel reads and
 * rad_widen reads the operand a kernel wrote, so a dtype handled by one and not the other is a
 * case that draws zeros or compares zeros -- and two zeroed sides agree, which reports as a
 * pass. Held to the integer and float types the core's vocabulary names; the packed weight
 * formats (W4, W2, MXFP4) are deliberately absent from both, since only the kernel's layout hook
 * knows what their bytes mean.
 *
 * This is completeness and not identity, because the two halves are not inverses for every type:
 * a drawn i8 is a QUANTISED unit normal and carries a scale, so writing 5.0 and reading 127 back
 * is the pair working. Identity is asserted separately below, for the types where a value is its
 * own meaning. */
TEST(every_plain_dtype_the_harness_writes_it_can_read_back) {
    struct Row { uint32_t dt; const char* name; };
    const Row rows[] = {
        { RAD_F32, "f32" }, { RAD_BF16, "bf16" }, { RAD_F16, "f16" },
        { RAD_F8E4M3, "f8e4m3" }, { RAD_F8E5M2, "f8e5m2" },
        { RAD_I8, "i8" }, { RAD_U8, "u8" }, { RAD_I16, "i16" },
        { RAD_I32, "i32" }, { RAD_I64, "i64" }, { RAD_U32, "u32" }, { RAD_BOOL, "bool" },
    };
    const float src[] = { 1.0f, 2.0f, 3.0f, 4.0f };
    const int64_t n = (int64_t)(sizeof src / sizeof src[0]);

    for (const Row& r : rows) {
        std::vector<uint8_t> packed((size_t)rad_dtype_bytes(r.dt, n), 0);
        rad_narrow(src, r.dt, n, packed.data());
        bool wrote = false;
        for (uint8_t b : packed) if (b) { wrote = true; break; }

        std::vector<float> back;
        rad_widen(packed.data(), r.dt, n, &back);
        CHECK_EQ((int64_t)back.size(), n);
        bool read = false;
        for (float x : back) if (x != 0.0f) { read = true; break; }

        if (!wrote || !read)
            std::fprintf(stderr, "    %s: narrow wrote %s, widen read %s\n", r.name,
                         wrote ? "bytes" : "NOTHING", read ? "values" : "NOTHING");
        CHECK(wrote);
        CHECK(read);
    }
}

/* AN INTEGER OPERAND'S VALUE IS ITS MEANING, so these round trip exactly. They carry token ids,
 * slot numbers, sequence lengths and table sizes -- a scale or a rounding rule on any of them is
 * not a loss of precision, it is a different index. (i8 and u8 are absent on purpose: an i8
 * operand in this vocabulary is a quantised activation, not an index.) */
TEST(an_integer_operand_round_trips_exactly) {
    const uint32_t dts[] = { RAD_I16, RAD_I32, RAD_I64, RAD_U32 };
    const float src[] = { 0.0f, 1.0f, 17.0f, 4095.0f, 65535.0f, 7.0f, 250.0f, 3.0f };
    const int64_t n = (int64_t)(sizeof src / sizeof src[0]);
    for (uint32_t dt : dts) {
        std::vector<uint8_t> packed((size_t)rad_dtype_bytes(dt, n), 0);
        rad_narrow(src, dt, n, packed.data());
        std::vector<float> back;
        rad_widen(packed.data(), dt, n, &back);
        if ((int64_t)back.size() != n) { CHECK_EQ((int64_t)back.size(), n); continue; }
        for (int64_t i = 0; i < n; ++i) {
            /* i16 tops out below the largest probe and is expected to; everything else is exact. */
            if (dt == RAD_I16 && src[i] > 32767.0f) continue;
            if (back[(size_t)i] != src[i])
                std::fprintf(stderr, "    %s: %g came back %g\n", rad_dtype_name(dt), src[i],
                             back[(size_t)i]);
            CHECK_EQ(back[(size_t)i], src[i]);
        }
    }
    /* bool carries one bit of each value and is expected to. */
    const float flags[] = { 0.0f, 1.0f, 0.0f, 9.0f };
    std::vector<uint8_t> packed(4, 0xEE);
    rad_narrow(flags, RAD_BOOL, 4, packed.data());
    std::vector<float> back;
    rad_widen(packed.data(), RAD_BOOL, 4, &back);
    CHECK_EQ((int64_t)back.size(), (int64_t)4);
    for (int i = 0; i < 4 && i < (int)back.size(); ++i)
        CHECK_EQ(back[(size_t)i], flags[i] != 0.0f ? 1.0f : 0.0f);
}

/* The float formats round trip to their own precision, which is what makes a recorded bf16
 * reference comparable against a bf16 output at all. */
TEST(a_float_dtype_round_trips_to_its_own_precision) {
    struct Row { uint32_t dt; double tol; };
    const Row rows[] = { { RAD_F32, 0.0 }, { RAD_BF16, 0.05 }, { RAD_F16, 0.005 },
                         { RAD_F8E4M3, 0.0 }, { RAD_F8E5M2, 1.0 } };
    const float src[] = { 0.0f, 1.0f, 2.0f, 5.0f, 9.0f, -3.0f, 7.0f, -4.0f };
    const int64_t n = (int64_t)(sizeof src / sizeof src[0]);
    for (const Row& r : rows) {
        std::vector<uint8_t> packed((size_t)rad_dtype_bytes(r.dt, n), 0);
        rad_narrow(src, r.dt, n, packed.data());
        std::vector<float> back;
        rad_widen(packed.data(), r.dt, n, &back);
        if ((int64_t)back.size() != n) { CHECK_EQ((int64_t)back.size(), n); continue; }
        for (int64_t i = 0; i < n; ++i)
            CHECK_NEAR(back[(size_t)i], src[i], r.tol);
    }
}

TEST(an_operand_seed_is_distinct_per_operand_and_stable) {
    const uint64_t c = 0xABCDEF01ull;
    std::set<uint64_t> seen;
    for (int k = 0; k < 32; ++k) {
        const uint64_t s = kf_operand_seed(c, k);
        CHECK(seen.insert(s).second);
        CHECK_EQ(s, kf_operand_seed(c, k));
    }
    CHECK(kf_operand_seed(c, 0) != kf_operand_seed(c + 1, 0));
}

/* A WEIGHT'S SEED IS ITS IDENTITY, NOT ITS CASE, and that is load-bearing twice over: it is what
 * lets an M sweep of ten query lengths draw one 178-million-element weight once, and it is the
 * key the caller caches the drawn bytes under. Keyed on the case instead, the cache would never
 * hit and the same weight would be a different weight at every point of the sweep. */
TEST(a_weight_is_drawn_from_its_own_identity_and_not_the_case) {
    std::vector<KfOperand> opd = { opd_f32({ 64, 32 }, RAD_OPD_WEIGHT) };

    std::vector<uint8_t> a, b;
    kf_draw(opd, 0, /*M=*/1, kf_vocab_rows(opd), kf_operand_seed(111, 0), &a);
    kf_draw(opd, 0, /*M=*/512, kf_vocab_rows(opd), kf_operand_seed(999, 0), &b);
    CHECK_EQ(a.size(), b.size());
    CHECK_EQ(std::memcmp(a.data(), b.data(), a.size()), 0);

    /* And the identity is position, dtype and extents -- a change in any of them is a different
     * weight, or two unrelated operands would share a cache entry. */
    const uint64_t s = kf_weight_seed(opd[0], 0);
    CHECK_EQ(s, kf_weight_seed(opd[0], 0));
    CHECK(s != kf_weight_seed(opd[0], 1));
    KfOperand other = opd[0];
    other.dtype = RAD_BF16;
    CHECK(s != kf_weight_seed(other, 0));
    other = opd[0];
    other.shape = { 32, 64 };
    CHECK(s != kf_weight_seed(other, 0));
}

/* A weight is drawn at a checkpoint's scale rather than at unit variance. Drawn unit, an fp8
 * quantiser's amax sits three orders of magnitude off where a real one does, so the case would
 * measure the quantiser on data no model produces. */
TEST(a_weight_is_drawn_at_the_scale_a_checkpoint_holds) {
    std::vector<KfOperand> opd = { opd_f32({ 256, 256 }, RAD_OPD_WEIGHT) };
    const std::vector<float> w = draw_f32(opd, 0, 1, 0);
    double ss = 0;
    for (float x : w) ss += (double)x * (double)x;
    CHECK_NEAR(std::sqrt(ss / (double)w.size()), 0.02, 0.002);
}

/* ================================================================== sampling positions */

TEST(a_small_operand_is_sampled_exhaustively) {
    std::vector<int64_t> pos;
    kf_sample_positions(42, 300, &pos);
    CHECK_EQ((int64_t)pos.size(), 300);
    for (int64_t i = 0; i < 300; ++i) CHECK_EQ(pos[(size_t)i], i);
    /* Below the sample count the seed cannot matter, because nothing is drawn. */
    std::vector<int64_t> other;
    kf_sample_positions(43, 300, &other);
    CHECK(pos == other);
}

TEST(a_large_operand_is_sampled_at_positions_both_sides_can_derive) {
    const int64_t n = 1 << 20;
    std::vector<int64_t> a, b;
    kf_sample_positions(0x5EEDull, n, &a);
    kf_sample_positions(0x5EEDull, n, &b);
    CHECK_EQ((int64_t)a.size(), (int64_t)KF_SAMPLES);
    CHECK(a == b);
    for (int64_t p : a) CHECK(p >= 0 && p < n);
    std::vector<int64_t> c;
    kf_sample_positions(0x5EEEull, n, &c);
    CHECK(a != c);
}

TEST(an_empty_operand_is_sampled_at_no_position) {
    std::vector<int64_t> pos = { 9 };
    kf_sample_positions(1, 0, &pos);
    CHECK(pos.empty());
    kf_sample_positions(1, -4, &pos);
    CHECK(pos.empty());
}

/* ================================================================== the reduction */

/* The reduction is threaded over blocks and the partials are added back in BLOCK order, so the
 * sum does not move with the core count. Checked against a serial sum taken the same way rather
 * than against a naive left-to-right one, which is a different number in floating point and
 * would make this a test of associativity. */
TEST(the_reduction_adds_its_blocks_back_in_order) {
    const int64_t n = KF_REDUCE_BLOCK * 3 + 1000;   /* past the threaded threshold */
    std::vector<float> v((size_t)n, 0.0f);
    rad_fill_normal(v.data(), n, 0xBEEFull, 1.0f);

    double got_ss = 0;
    int64_t got_nf = 0;
    std::vector<float> sample;
    kf_reduce(v.data(), n, 7, &got_ss, &got_nf, &sample);

    double want = 0;
    for (int64_t lo = 0; lo < n; lo += KF_REDUCE_BLOCK) {
        const int64_t hi = std::min<int64_t>(lo + KF_REDUCE_BLOCK, n);
        double part = 0;
        for (int64_t i = lo; i < hi; ++i) part += (double)v[(size_t)i] * (double)v[(size_t)i];
        want += part;
    }
    CHECK_EQ(got_ss, want);
    CHECK_EQ(got_nf, (int64_t)0);

    double again_ss = 0;
    int64_t again_nf = 0;
    std::vector<float> again;
    kf_reduce(v.data(), n, 7, &again_ss, &again_nf, &again);
    CHECK_EQ(again_ss, got_ss);
    CHECK(again == sample);
}

/* A NaN on one side and a number on the other is a failure and is checked before any tolerance,
 * so the count has to be exact -- and a non-finite element must not enter the norm, or one NaN
 * makes the whole sum NaN and every comparison against it meaningless. */
TEST(a_non_finite_element_is_counted_and_kept_out_of_the_norm) {
    std::vector<float> v(10, 2.0f);
    v[3] = std::nanf("");
    v[7] = INFINITY;
    double ss = 0;
    int64_t nf = 0;
    std::vector<float> sample;
    kf_reduce(v.data(), (int64_t)v.size(), 1, &ss, &nf, &sample);
    CHECK_EQ(nf, (int64_t)2);
    CHECK_EQ(ss, 8.0 * 4.0);          /* the eight finite 2.0s */
    CHECK_EQ((int64_t)sample.size(), (int64_t)v.size());
    CHECK(std::isnan(sample[3]));
    CHECK(std::isinf(sample[7]));
}

TEST(the_reduction_samples_the_positions_a_reader_will_ask_for) {
    const int64_t n = (int64_t)KF_SAMPLES * 4;
    std::vector<float> v((size_t)n, 0.0f);
    for (int64_t i = 0; i < n; ++i) v[(size_t)i] = (float)i;
    double ss = 0;
    int64_t nf = 0;
    std::vector<float> sample;
    kf_reduce(v.data(), n, 0x1234ull, &ss, &nf, &sample);

    std::vector<int64_t> pos;
    kf_sample_positions(0x1234ull, n, &pos);
    CHECK_EQ(sample.size(), pos.size());
    for (size_t i = 0; i < pos.size() && i < sample.size(); ++i)
        CHECK_EQ(sample[i], (float)pos[i]);
}

/* ================================================================== the input-domain laws
 *
 * Each of these puts an operand inside the domain its op is only ever handed. Drawn outside it,
 * the two implementations disagree by however ill-conditioned the input is and the report blames
 * the kernel for the condition number of the test.
 */

/* Cumulative sequence lengths are the batch's shape, not a draw: every gated-delta-net kernel
 * opens by reading cu[n] before any null test, so a wrong one faults rather than answering. */
TEST(a_cumulative_length_operand_is_the_batch_shape_not_a_draw) {
    KfOperand d = opd_f32({ 5 });
    d.dtype = RAD_I32;
    d.is_index = true;
    d.idx_cu = true;
    const std::vector<float> v = draw_f32({ d }, 0, /*M=*/100, 0xAAAA);
    CHECK_EQ((int64_t)v.size(), (int64_t)5);
    CHECK_EQ(v[0], 0.0f);
    CHECK_EQ(v[4], 100.0f);
    for (size_t i = 1; i < v.size(); ++i) CHECK(v[i] >= v[i - 1]);
}

/* A length is a length. A random context shorter than the query makes causal attention read a
 * prefix that does not exist, and both implementations then agree about the garbage. */
TEST(a_length_operand_is_the_constant_it_was_recorded_with) {
    KfOperand d = opd_f32({ 6 });
    d.dtype = RAD_I32;
    d.is_index = true;
    d.idx_const = 4096;
    const std::vector<float> v = draw_f32({ d }, 0, 8, 0xBBBB);
    for (float x : v) CHECK_EQ(x, 4096.0f);
}

/* A scatter's destinations must be distinct: two tokens on one slot is a race the two
 * implementations resolve differently, so the comparison would measure the scheduler. */
TEST(a_scatter_destination_is_drawn_without_repeats) {
    KfOperand d = opd_f32({ 8 });
    d.dtype = RAD_I32;
    d.is_index = true;
    d.idx_unique = true;
    d.idx_max = 16;
    const std::vector<float> v = draw_f32({ d }, 0, 8, 0xCCCC);
    std::set<int> seen;
    for (float x : v) {
        CHECK(x >= 0.0f && x < 16.0f);
        CHECK(seen.insert((int)x).second);
    }

    /* Asked for more destinations than the range holds, the surplus is marked rather than
     * repeated: a duplicate would be the very race the law exists to keep out. */
    d.idx_max = 4;
    const std::vector<float> w = draw_f32({ d }, 0, 8, 0xCCCC);
    std::set<int> first;
    for (size_t i = 0; i < 4; ++i) CHECK(first.insert((int)w[i]).second);
    for (size_t i = 4; i < w.size(); ++i) CHECK_EQ(w[i], -1.0f);
}

/* An index indexes the operand before it, and its leading extent is the only legal range. */
TEST(an_index_operand_stays_inside_the_operand_it_indexes) {
    std::vector<KfOperand> opd = { opd_f32({ 7, 3 }, RAD_OPD_WEIGHT), opd_f32({ 32 }) };
    opd[1].dtype = RAD_I32;
    opd[1].is_index = true;
    const std::vector<float> v = draw_f32(opd, 1, 32, 0xDDDD);
    bool any_past_zero = false;
    for (float x : v) {
        CHECK(x >= 0.0f && x < 7.0f);
        if (x > 0.0f) any_past_zero = true;
    }
    CHECK(any_past_zero);          /* drawn, not zeroed */

    /* Where the range is not any operand's leading extent -- a KV slot indexes
     * n_blocks * block_size -- the recipe names it and the named range wins. */
    opd[1].idx_max = 200;
    const std::vector<float> w = draw_f32(opd, 1, 32, 0xDDDD);
    bool past_seven = false;
    for (float x : w) {
        CHECK(x >= 0.0f && x < 200.0f);
        if (x >= 7.0f) past_seven = true;
    }
    CHECK(past_seven);
}

TEST(a_sigmoid_gate_is_drawn_inside_zero_to_one) {
    const std::vector<float> v = draw_f32({ opd_f32({ 64 }, RAD_OPD_IN, KF_FILL_SIGMOID) },
                                          0, 64, 0xEEEE);
    bool spread = false;
    for (float x : v) {
        CHECK(x > 0.0f && x < 1.0f);
        if (x > 0.6f || x < 0.4f) spread = true;
    }
    CHECK(spread);
}

/* The chunked delta rule's intra-chunk matrix is I + tril(diag(beta) K K^T, -1) -- unit lower
 * triangular by construction, with a bounded inverse. Drawn with a normal on the diagonal it is
 * not: an element near zero makes the inverse enormous and gdn_kkt_solve then returns values of
 * order 1e20 with a rel_l2 that grows with the chunk count. */
TEST(an_intra_chunk_matrix_is_drawn_unit_lower_triangular) {
    KfOperand d = opd_f32({ 8, 2, 4 }, RAD_OPD_IN, KF_FILL_TRI_LOWER);
    d.fill_chunk = 4;
    const std::vector<float> v = draw_f32({ d }, 0, 8, 0xF00D);
    const int64_t heads = 2, ch = 4, rows = 8;
    for (int64_t t = 0; t < rows; ++t)
        for (int64_t h = 0; h < heads; ++h)
            for (int64_t j = 0; j < ch; ++j) {
                const float x = v[(size_t)((t * heads + h) * ch + j)];
                const int64_t diag = t % ch;
                if (j > diag)       CHECK_EQ(x, 0.0f);
                else if (j == diag) CHECK_EQ(x, 1.0f);
                else                CHECK(std::fabs(x) < 4.0f);
            }
}

/* The gated-delta-net key arrives from the model's own qk_norm with unit rows. Drawn as plain
 * normals over a 128-wide head it has norm ~11, the solve's off-diagonals are then two orders of
 * magnitude too large and its inverse is astronomical. */
TEST(a_qk_normed_key_is_drawn_with_unit_rows) {
    const int64_t r = 6, w = 128;
    const std::vector<float> v = draw_f32({ opd_f32({ r, w }, RAD_OPD_IN, KF_FILL_UNIT_ROW) },
                                          0, r, 0x1010);
    for (int64_t i = 0; i < r; ++i) {
        double ss = 0;
        for (int64_t j = 0; j < w; ++j) {
            const double x = (double)v[(size_t)(i * w + j)];
            ss += x * x;
        }
        CHECK_NEAR(std::sqrt(ss), 1.0, 1e-5);
    }
}

/* The decay plane is an intra-chunk cumulative sum of log-decays that RESETS at every chunk
 * boundary, which is what chunk_local_cumsum produces and what the scan assumes. Fed plain
 * normals the reference overflows to 2e38 and the comparison measures the overflow. */
TEST(a_gate_cumsum_runs_down_a_chunk_and_resets_at_its_boundary) {
    const int64_t rows = 8, heads = 2;
    KfOperand d = opd_f32({ rows, heads }, RAD_OPD_IN, KF_FILL_GATE_CUMSUM);
    KfOperand plain = d;
    plain.fill_chunk = 0;                    /* one chunk over the whole plane: no reset */
    d.fill_chunk = 4;

    const std::vector<float> chunked = draw_f32({ d }, 0, rows, 0x2020);
    const std::vector<float> whole   = draw_f32({ plain }, 0, rows, 0x2020);

    for (int64_t h = 0; h < heads; ++h) {
        /* Strictly decreasing within a chunk: each step subtracts log1p(exp(x)), always positive,
         * and the accumulator starts at zero so every value is negative. */
        for (int64_t t = 0; t < rows; ++t) {
            const float x = chunked[(size_t)(t * heads + h)];
            CHECK(x < 0.0f);
            if (t % 4 != 0) CHECK(x < chunked[(size_t)((t - 1) * heads + h)]);
        }
        /* The first chunk is the same either way, because there is nothing to reset yet. */
        for (int64_t t = 0; t < 4; ++t)
            CHECK_EQ(chunked[(size_t)(t * heads + h)], whole[(size_t)(t * heads + h)]);
        /* And at the boundary the accumulator restarts: the chunked value is exactly the step the
         * unchunked one took, rather than that step added to everything before it. */
        const double step = (double)whole[(size_t)(4 * heads + h)] -
                            (double)whole[(size_t)(3 * heads + h)];
        CHECK_NEAR(chunked[(size_t)(4 * heads + h)], step, 1e-5);
    }
}

TEST(an_output_operand_is_sized_and_left_alone) {
    const std::vector<float> v = draw_f32({ opd_f32({ 16, 4 }, RAD_OPD_OUT) }, 0, 16, 0x3030);
    CHECK_EQ((int64_t)v.size(), (int64_t)64);
    for (float x : v) CHECK_EQ(x, 0.0f);
}

TEST(an_absent_operand_draws_nothing) {
    KfOperand d = opd_f32({ 16 });
    d.absent = true;
    std::vector<uint8_t> bytes;
    kf_draw({ d }, 0, 16, 16, 1, &bytes);
    for (uint8_t b : bytes) CHECK_EQ((int)b, 0);
}

/* The fallback range for an index operand with nothing to index. Derived from the case on both
 * sides rather than read off the model on one of them, or the recorder and the reader would draw
 * different token ids from the same seed. */
TEST(the_fallback_index_range_is_the_widest_leading_extent) {
    CHECK_EQ(kf_vocab_rows({ opd_f32({ 4, 9 }), opd_f32({ 128, 2 }), opd_f32({ 7 }) }),
             (int64_t)128);
    CHECK_EQ(kf_vocab_rows({}), (int64_t)0);
}

/* ================================================================== the file */

TEST(a_fixture_survives_a_save_and_a_load) {
    Kfixture fx;
    fx.source = "/models/thing.rad";
    fx.oracle = "libref@0.1.0";
    fx.recorded = "2026-01-02";

    KfCase c;
    c.op = "rmsnorm";
    c.seed = 0xFEEDFACEDEADBEEFull;
    c.m = 517;
    c.band = "M in (64, 512]";
    c.speed_only = true;
    c.params.push_back(KfParam{ "M", RAD_P_INT, 517, 0, "" });
    c.params.push_back(KfParam{ "eps", RAD_P_F64, 0, 1e-6, "" });
    c.params.push_back(KfParam{ "dtype", RAD_P_STR, 0, 0, "bf16" });

    KfOperand in = opd_f32({ 517, 4096 }, RAD_OPD_IN, KF_FILL_GATE_CUMSUM);
    in.fill_chunk = 64;
    in.is_index = true;
    in.idx_unique = true;
    in.idx_cu = true;
    in.idx_max = 99;
    in.idx_const = 12;
    c.opd.push_back(in);

    KfOperand out = opd_f32({ 517, 4096 }, RAD_OPD_OUT);
    out.dtype = RAD_BF16;
    out.have_ref = true;
    out.ref_n = 517 * 4096;
    out.ref_sumsq = 1234.5;
    out.ref_nonfinite = 3;
    out.sample.assign(KF_SAMPLES, 0.0f);
    for (int i = 0; i < KF_SAMPLES; ++i) out.sample[(size_t)i] = (float)i * 0.5f;
    c.opd.push_back(out);

    KfOperand gone = opd_f32({ 1 });
    gone.absent = true;
    c.opd.push_back(gone);

    c.opd.push_back(fp8_weight(256, 384, { "codes", "scale" }));
    fx.cases.push_back(c);

    const std::string path = scratch_path("roundtrip.rkb");
    CHECK_OK(fx.save(path));

    Kfixture rd;
    CHECK_OK(rd.load(path));
    CHECK_EQ(rd.version, fx.version);
    CHECK_EQ(rd.source, fx.source);
    CHECK_EQ(rd.oracle, fx.oracle);
    CHECK_EQ(rd.recorded, fx.recorded);
    CHECK_EQ(rd.cases.size(), (size_t)1);
    if (rd.cases.size() != 1) { std::remove(path.c_str()); return; }

    const KfCase& g = rd.cases[0];
    CHECK_EQ(g.op, c.op);
    CHECK_EQ(g.seed, c.seed);
    CHECK_EQ(g.m, c.m);
    CHECK_EQ(g.band, c.band);
    CHECK_EQ(g.speed_only, c.speed_only);
    CHECK_EQ(g.params.size(), c.params.size());
    for (size_t i = 0; i < g.params.size() && i < c.params.size(); ++i) {
        CHECK_EQ(g.params[i].key, c.params[i].key);
        CHECK_EQ(g.params[i].kind, c.params[i].kind);
        CHECK_EQ(g.params[i].i, c.params[i].i);
        CHECK_EQ(g.params[i].d, c.params[i].d);
        CHECK_EQ(g.params[i].s, c.params[i].s);
    }
    CHECK_EQ(g.opd.size(), c.opd.size());
    for (size_t i = 0; i < g.opd.size() && i < c.opd.size(); ++i) {
        const KfOperand& a = g.opd[i];
        const KfOperand& b = c.opd[i];
        CHECK_EQ(a.role, b.role);
        CHECK_EQ(a.dtype, b.dtype);
        CHECK(a.shape == b.shape);
        CHECK_EQ(a.absent, b.absent);
        CHECK_EQ(a.fill, b.fill);
        CHECK_EQ(a.is_index, b.is_index);
        CHECK_EQ(a.idx_unique, b.idx_unique);
        CHECK_EQ(a.idx_cu, b.idx_cu);
        CHECK_EQ(a.idx_max, b.idx_max);
        CHECK_EQ(a.idx_const, b.idx_const);
        CHECK_EQ(a.fill_chunk, b.fill_chunk);
        CHECK_EQ(a.have_ref, b.have_ref);
        CHECK_EQ(a.ref_n, b.ref_n);
        CHECK_EQ(a.ref_sumsq, b.ref_sumsq);
        CHECK_EQ(a.ref_nonfinite, b.ref_nonfinite);
        CHECK(a.sample == b.sample);
        CHECK_EQ(a.has_enc, b.has_enc);
        CHECK(rad_enc_equal(&a.enc, &b.enc));
        CHECK(a.sel == b.sel);
        CHECK(a.sel_rows == b.sel_rows);
        CHECK(a.sel_cols == b.sel_cols);
    }
    std::remove(path.c_str());
}

TEST(a_fixture_that_is_not_there_is_not_found) {
    Kfixture fx;
    CHECK_EQ(fx.load(scratch_path("no_such_file.rkb")), RAD_E_NOTFOUND);
}

/* A reader that does not refuse a file it does not understand reads a struct that has moved, and
 * the failure mode is a report full of confident wrong numbers rather than an error. */
TEST(a_fixture_the_reader_does_not_understand_is_refused) {
    Kfixture fx;
    fx.source = "x";
    fx.cases.push_back(KfCase{});
    const std::string path = scratch_path("bad.rkb");
    CHECK_OK(fx.save(path));

    auto poke = [&](long off, const void* bytes, size_t n) {
        FILE* f = std::fopen(path.c_str(), "r+b");
        if (!f) { CHECK(false); return; }
        std::fseek(f, off, SEEK_SET);
        std::fwrite(bytes, 1, n, f);
        std::fclose(f);
    };

    const char wrong_magic[4] = { 'R', 'K', 'B', '9' };
    poke(0, wrong_magic, 4);
    Kfixture rd;
    CHECK_EQ(rd.load(path), RAD_E_FORMAT);

    const char good_magic[4] = { 'R', 'K', 'B', '1' };
    poke(0, good_magic, 4);
    const uint32_t wrong_version = 99;
    poke(4, &wrong_version, 4);
    CHECK_EQ(rd.load(path), RAD_E_FORMAT);

    /* The sample count is derived, not stored per value, so a file recorded at another count
     * would have every sample compared against the wrong element. */
    const uint32_t version = 4;
    poke(4, &version, 4);
    const uint32_t wrong_samples = 512;
    poke(8, &wrong_samples, 4);
    CHECK_EQ(rd.load(path), RAD_E_FORMAT);
    std::remove(path.c_str());
}


/* A FILE WHOSE WEIGHTS DO NOT CARRY THEIR ENCODING IS REFUSED, not read. Its weight operands are
 * tensors a kernel's layout hooks cannot take, and replaying it would hand every relayout a plain
 * plane in place of the encoding the reference was computed through. */
TEST(a_fixture_without_encodings_is_refused) {
    Kfixture fx;
    fx.source = "x";
    fx.cases.push_back(KfCase{});
    const std::string path = scratch_path("v3.rkb");
    CHECK_OK(fx.save(path));
    FILE* f = std::fopen(path.c_str(), "r+b");
    REQUIRE(f != nullptr);
    const uint32_t v3 = 3;
    std::fseek(f, 4, SEEK_SET);
    std::fwrite(&v3, 1, 4, f);
    std::fclose(f);
    Kfixture rd;
    CHECK_EQ(rd.load(path), RAD_E_FORMAT);
    std::remove(path.c_str());
}

TEST(a_truncated_fixture_is_refused) {
    Kfixture fx;
    fx.source = "x";
    KfCase c;
    c.op = "rmsnorm";
    c.opd.push_back(opd_f32({ 4 }));
    fx.cases.push_back(c);
    const std::string path = scratch_path("short.rkb");
    CHECK_OK(fx.save(path));

    FILE* f = std::fopen(path.c_str(), "rb");
    CHECK(f != nullptr);
    if (!f) return;
    std::fseek(f, 0, SEEK_END);
    const long n = std::ftell(f);
    std::fclose(f);
    CHECK(n > 24);
    CHECK_EQ(truncate(path.c_str(), n - 8), 0);

    Kfixture rd;
    CHECK_EQ(rd.load(path), RAD_E_FORMAT);
    std::remove(path.c_str());
}

/* ================================================================== a weight's planes */

namespace {

/* Select every plane of `e` over a [rows, cols] weight, as a declaration taking all of them. */
KfOperand all_planes(const RadEncoding& e, int64_t rows, int64_t cols) {
    KfOperand d;
    d.role = RAD_OPD_WEIGHT;
    d.has_enc = true;
    d.enc = e;
    d.dtype = RAD_F32;
    d.shape = { rows, cols };
    for (int i = 0; i < e.n_planes; ++i) {
        int64_t pr = 0, pc = 0;
        rad_enc_plane_dims(&e.plane[i], rows, cols, &pr, &pc);
        d.sel.push_back(i);
        d.sel_rows.push_back(pr);
        d.sel_cols.push_back(pc);
    }
    return d;
}

}  /* namespace */

/* A DRAWN PLANE HOLDS ONLY WHAT ITS ROLE CAN. A code past its codebook is a read past a table; a
 * permutation that repeats a column is a different weight on every reader; an exponent scale
 * drawn over every exponent overflows the reference. Any of them turns a replay into a
 * comparison of two ways of handling garbage. */
TEST(a_weights_planes_are_drawn_inside_what_each_can_hold) {
    const int64_t R = 8, C = 64;
    RadEncoding e;
    rad_enc_clear(&e);
    rad_enc_copy_str(e.scheme, "affine");
    rad_enc_copy_str(e.transform, "perm");
    rad_enc_add_plane(&e, "codes", RAD_U8, 1, 1);
    rad_enc_add_plane(&e, "scale", RAD_E8M0, 1, 32);
    rad_enc_add_table(&e, "table", RAD_F32, 1, 10);
    rad_enc_add_table(&e, "t.perm", RAD_I32, 1, C);
    REQUIRE(rad_enc_valid(&e));
    const KfOperand d = all_planes(e, R, C);

    std::vector<std::vector<uint8_t>> a, b, x;
    kf_draw_planes(d, 7, &a);
    kf_draw_planes(d, 7, &b);
    kf_draw_planes(d, 8, &x);
    REQUIRE_EQ(a.size(), (size_t)4);
    CHECK(a == b);                                   /* a seed is the planes */
    CHECK(a[0] != x[0]);
    for (int64_t i = 0; i < R * C; ++i) CHECK(a[0][(size_t)i] < 10);
    for (uint8_t s8 : a[1]) CHECK(s8 >= 123 && s8 <= 131);
    std::vector<int32_t> perm((size_t)C);
    std::memcpy(perm.data(), a[3].data(), (size_t)C * 4);
    std::sort(perm.begin(), perm.end());
    for (int64_t j = 0; j < C; ++j) CHECK_EQ(perm[(size_t)j], (int32_t)j);
}

/* THE REFERENCE READS WHAT THE PLANES DECODE TO: the codes times their block's scale for the whole
 * encoding, and one plane as its own values -- widened where the operand is wider. */
TEST(the_reference_reads_what_the_planes_decode_to) {
    const int64_t R = 256, C = 384;
    KfOperand both = fp8_weight(R, C, { "codes", "scale" });
    std::vector<std::vector<uint8_t>> pl;
    kf_draw_planes(both, 3, &pl);
    std::vector<uint8_t> out;
    std::string why;
    REQUIRE_EQ(kf_planes_logical(both, pl, &out, &why), RAD_OK);
    REQUIRE_EQ(out.size(), (size_t)(R * C * 4));
    const float* f = (const float*)out.data();
    for (int64_t r : { (int64_t)0, (int64_t)130, R - 1 })
        for (int64_t c : { (int64_t)0, (int64_t)200, C - 1 }) {
            const float code = rad_load_f32(pl[0].data(), RAD_F8E4M3, r * C + c);
            const float scale = rad_load_f32(pl[1].data(), RAD_BF16, (r / 128) * 3 + c / 128);
            CHECK_EQ(f[r * C + c], code * scale);
        }

    KfOperand sc = fp8_weight(R, C, { "scale" });
    sc.dtype = RAD_F32;                              /* a reference reading the scales in f32 */
    std::vector<std::vector<uint8_t>> sp;
    kf_draw_planes(sc, 3, &sp);
    REQUIRE_EQ(kf_planes_logical(sc, sp, &out, &why), RAD_OK);
    REQUIRE_EQ(out.size(), (size_t)(2 * 3 * 4));
    for (int i = 0; i < 6; ++i)
        CHECK_EQ(((const float*)out.data())[i], rad_load_f32(sp[0].data(), RAD_BF16, i));
}

/* A SCALE WITH THE LEVEL ABOVE IT, selected without the codes -- the scale view of a w4nl64a8h
 * expert, an E4M3 scale a (row, 64 of K) under one f32 for the weight. The reference reads each
 * scale times the level, in the scale plane's own extents. Part of a chain that does not start at
 * its first level, a level finer than the scale it would multiply, and a reader wanting another
 * count are refused rather than read as something. */
TEST(the_reference_reads_a_scale_times_the_level_above_it) {
    const int64_t R = 32, C = 256;
    RadEncoding e;
    rad_enc_clear(&e);
    rad_enc_copy_str(e.scheme, "affine");
    rad_enc_copy_str(e.transform, "fwht128");
    rad_enc_add_plane(&e, "codes", RAD_U4, 1, 1);
    rad_enc_add_plane(&e, "scale", RAD_F8E4M3, 1, 64);
    rad_enc_add_plane(&e, "scale.1", RAD_F32, 0, 0);
    rad_enc_add_table(&e, "table", RAD_F32, 1, 16);
    REQUIRE(rad_enc_valid(&e));
    auto view = [&](const RadEncoding& enc, std::vector<const char*> roles) {
        KfOperand d;
        d.role = RAD_OPD_WEIGHT;
        d.has_enc = true;
        d.enc = enc;
        d.dtype = RAD_F32;
        for (const char* r : roles) {
            const int i = rad_enc_find(&enc, r);
            int64_t pr = 0, pc = 0;
            rad_enc_plane_dims(&enc.plane[i], R, C, &pr, &pc);
            d.sel.push_back(i);
            d.sel_rows.push_back(pr);
            d.sel_cols.push_back(pc);
        }
        d.shape = { d.sel_rows[0], d.sel_cols[0] };
        return d;
    };
    const KfOperand d = view(e, { "scale", "scale.1" });
    REQUIRE_EQ(numel(d), R * C / 64);
    std::vector<std::vector<uint8_t>> pl;
    kf_draw_planes(d, 9, &pl);
    REQUIRE_EQ(pl.size(), (size_t)2);
    std::vector<uint8_t> out;
    std::string why;
    REQUIRE_EQ(kf_planes_logical(d, pl, &out, &why), RAD_OK);
    const float level = rad_load_f32(pl[1].data(), RAD_F32, 0);
    const float* f = (const float*)out.data();
    for (int64_t i = 0; i < R * C / 64; ++i)
        CHECK_EQ(f[i], rad_load_f32(pl[0].data(), RAD_F8E4M3, i) * level);

    CHECK_EQ(kf_planes_logical(view(e, { "scale.1", "scale" }), pl, &out, &why), RAD_E_SHAPE);
    KfOperand more = d;
    more.shape = { R, C / 32 };
    CHECK_EQ(kf_planes_logical(more, pl, &out, &why), RAD_E_SHAPE);
    RadEncoding fine = e;
    fine.plane[rad_enc_find(&e, "scale.1")].block[1] = 32;
    const KfOperand df = view(fine, { "scale", "scale.1" });
    std::vector<std::vector<uint8_t>> pf;
    kf_draw_planes(df, 9, &pf);
    CHECK_EQ(kf_planes_logical(df, pf, &out, &why), RAD_E_SHAPE);
    CHECK(why.find("does not cover") != std::string::npos);
}

namespace {

int lay_identity(const RadParam*, int, int, const RadEncoding*, const int*, const RadTensor*, int,
                 RadLayout*) { return RAD_E_UNSUPPORTED; }
int lay_refuse(const RadParam*, int, int, const RadEncoding*, const int*, const RadTensor*, int,
               RadLayout*) { return RAD_E_DTYPE; }
int lay_reverse(const RadParam*, int, int, const RadEncoding*, const int*, const RadTensor* pl,
                int n, RadLayout* out) {
    if (n != 1) return RAD_E_SHAPE;
    out->dtype = pl[0].dtype;
    out->bytes = rad_dtype_bytes(pl[0].dtype, pl[0].shape[0] * pl[0].shape[1]);
    return RAD_OK;
}
int relay_reverse(const RadParam*, int, int, const RadEncoding*, const int*, const RadTensor* pl,
                  int, void* dst, int64_t bytes) {
    const uint8_t* s = (const uint8_t*)pl[0].data;
    for (int64_t i = 0; i < bytes; ++i) ((uint8_t*)dst)[i] = s[bytes - 1 - i];
    return RAD_OK;
}

}  /* namespace */

/* A KERNEL IS HANDED WHAT ITS OWN LAYOUT HOOKS MAKE OF THE PLANES, and the three answers a hook
 * gives are three different things: read as stored, not this encoding, or these bytes. */
TEST(a_kernel_is_handed_what_its_own_layout_makes) {
    const KfOperand d = fp8_weight(128, 256, { "codes" });
    std::vector<std::vector<uint8_t>> pl;
    kf_draw_planes(d, 5, &pl);
    RadKernelInfo ki{};
    RadLayout lay{};
    std::vector<uint8_t> out;

    CHECK_EQ(kf_lay_weight(&ki, nullptr, 0, 0, d, pl, &lay, &out), RAD_E_UNSUPPORTED);
    ki.layout = lay_identity;
    CHECK_EQ(kf_lay_weight(&ki, nullptr, 0, 0, d, pl, &lay, &out), RAD_E_UNSUPPORTED);
    ki.layout = lay_refuse;
    CHECK_EQ(kf_lay_weight(&ki, nullptr, 0, 0, d, pl, &lay, &out), RAD_E_DTYPE);
    ki.layout = lay_reverse;
    CHECK_EQ(kf_lay_weight(&ki, nullptr, 0, 0, d, pl, &lay, &out), RAD_E_STATE);  /* no relayout */
    ki.relayout = relay_reverse;
    REQUIRE_EQ(kf_lay_weight(&ki, nullptr, 0, 0, d, pl, &lay, &out), RAD_OK);
    REQUIRE_EQ(out.size(), pl[0].size());
    CHECK(std::equal(out.begin(), out.end(), pl[0].rbegin()));
    CHECK_EQ(lay.dtype, (uint32_t)RAD_F8E4M3);
}

/* ================================================================== the artifact in the tree
 *
 * data/kernels.rkb is checked in and is the whole of the replay's coverage, so it is a thing the
 * tests have an opinion about rather than an input they trust. A fixture that became unreadable,
 * lost its cases or gained an operand dtype the harness cannot draw would otherwise present as a
 * rad-kbench run that quietly checks less than the fixture records.
 */
namespace {

/* False when the file is not there, which is a skip, or does not load, which is a failure: a stale
 * recording is the replay checking nothing. */
bool load_checked_in(const char* leaf, Kfixture* fx) {
    const char* home = std::getenv("RADIANCE_HOME");
    if (!home || !*home) return false;
    const std::string path = std::string(home) + "/" + leaf;
    const int rc = fx->load(path);
    if (rc == RAD_E_NOTFOUND) {
        std::fprintf(stderr, "  SKIP %s is not in $RADIANCE_HOME\n", leaf);
        return false;
    }
    if (rc != RAD_OK)
        std::fprintf(stderr, "    %s does not load (%s) -- record it again with rad-kbench -m "
                             "<model.rad> --max-cases 1 --record\n", path.c_str(),
                     rad_strerror(rc));
    CHECK_OK(rc);
    return rc == RAD_OK;
}

}  /* namespace */

TEST(the_checked_in_fixtures_load_and_describe_real_cases) {
    for (const char* leaf : { "kernels.rkb", "kernels_moe.rkb" }) {
        Kfixture fx;
        if (!load_checked_in(leaf, &fx)) continue;
        /* THE VERSION THE RECORDINGS ARE HELD AT, pinned so that a fixture re-recorded by an
         * older tool is caught here rather than as a page of confident kernel failures. */
        CHECK_EQ(fx.version, (uint32_t)4);
        CHECK(!fx.cases.empty());
        CHECK(!fx.source.empty());
        CHECK(!fx.oracle.empty());
        int with_ref = 0, with_enc = 0;
        for (const KfCase& c : fx.cases) {
            CHECK(!c.op.empty());
            CHECK(!c.opd.empty());
            for (const KfOperand& d : c.opd) {
                if (d.absent) continue;
                if (d.role == RAD_OPD_WEIGHT) {
                    /* every weight a recording draws carries its encoding */
                    CHECK(d.has_enc);
                    if (d.has_enc) ++with_enc;
                }
                CHECK(!d.shape.empty());
                if (d.have_ref) {
                    ++with_ref;
                    CHECK(d.ref_n > 0);
                    CHECK((int64_t)d.sample.size() ==
                          (d.ref_n < KF_SAMPLES ? d.ref_n : (int64_t)KF_SAMPLES));
                }
            }
        }
        CHECK(with_ref > 0);
        CHECK(with_enc > 0);
    }
}

/* EVERY DTYPE A RECORDED CASE USES MUST BE ONE THE HARNESS CAN BOTH WRITE AND READ. An operand
 * whose dtype rad_narrow does not handle is handed to the kernel as zeros; one rad_widen does not
 * handle is compared as zeros on both sides, which agrees perfectly and reports a pass. Neither
 * shows up as an error anywhere, so the fixtures are swept for it here. */
TEST(the_checked_in_fixtures_use_no_dtype_the_harness_cannot_draw_or_read) {
    const float probe[4] = { 1.0f, 2.0f, 3.0f, 4.0f };
    for (const char* leaf : { "kernels.rkb", "kernels_moe.rkb" }) {
        Kfixture fx;
        if (!load_checked_in(leaf, &fx)) continue;
        std::set<uint32_t> seen;
        for (const KfCase& c : fx.cases)
            for (const KfOperand& d : c.opd)
                if (!d.absent) seen.insert(d.dtype);

        for (uint32_t dt : seen) {
            /* A plugin-private dtype is nobody else's to interpret and is excluded by design. */
            if (dt >= RAD_DT_PLUGIN_BASE) continue;
            std::vector<uint8_t> packed((size_t)rad_dtype_bytes(dt, 4), 0);
            rad_narrow(probe, dt, 4, packed.data());
            bool wrote = false;
            for (uint8_t b : packed) if (b) { wrote = true; break; }
            std::vector<float> back;
            rad_widen(packed.data(), dt, 4, &back);
            bool read = false;
            for (float x : back) if (x != 0.0f) { read = true; break; }
            if (!wrote || !read)
                std::fprintf(stderr, "    %s: dtype %s (%u) narrow=%s widen=%s\n", leaf,
                             rad_dtype_name(dt), dt, wrote ? "yes" : "NO", read ? "yes" : "NO");
            CHECK(wrote);
            CHECK(read);
        }
    }
}

RAD_TEST_MAIN()
