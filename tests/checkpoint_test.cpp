/* checkpoint_test.cpp -- a checkpoint as a source of logical weights (core/format/checkpoint.h).
 *
 * rad-convert writes a container from a checkpoint and the loader serves a trivial one directly,
 * and both go through ckpt_resolve: which tensors make a declared weight, what encoding they
 * already are, and the rows of it. A wrong answer there is a model that loads and is wrong -- an
 * expert that is every expert, a scale applied to the next piece's rows -- so each case below is
 * one of the ways a checkpoint is laid out in practice, built small and read back.
 */
#include "rad_test.h"
#include "format/checkpoint.h"
#include "rad_plugin.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <unordered_set>
#include <vector>

using namespace rad;

namespace {

struct T {
    std::string name, dtype;          /* safetensors spelling: BF16, F32, F8_E4M3 */
    std::vector<int64_t> shape;
    std::vector<uint8_t> bytes;
};

std::string dir_path() {
    const char* d = std::getenv("TMPDIR");
    return std::string(d && *d ? d : "/tmp") + "/radiance_checkpoint_test_" +
           std::to_string((long long)getpid());
}

/* One model.safetensors and a config.json in a fresh directory. */
std::string write_checkpoint(const std::vector<T>& ts, const std::string& config) {
    const std::string dir = dir_path();
    ::mkdir(dir.c_str(), 0755);
    std::string hdr = "{";
    uint64_t off = 0;
    for (size_t i = 0; i < ts.size(); ++i) {
        std::string shape;
        for (size_t k = 0; k < ts[i].shape.size(); ++k)
            shape += (k ? "," : "") + std::to_string(ts[i].shape[k]);
        hdr += (i ? "," : "") + std::string("\"") + ts[i].name + "\":{\"dtype\":\"" + ts[i].dtype +
               "\",\"shape\":[" + shape + "],\"data_offsets\":[" + std::to_string(off) + "," +
               std::to_string(off + ts[i].bytes.size()) + "]}";
        off += ts[i].bytes.size();
    }
    hdr += "}";
    while (hdr.size() % 8) hdr += ' ';
    FILE* f = std::fopen((dir + "/model.safetensors").c_str(), "wb");
    const uint64_t n = hdr.size();
    std::fwrite(&n, 8, 1, f);
    std::fwrite(hdr.data(), 1, hdr.size(), f);
    for (const T& t : ts) std::fwrite(t.bytes.data(), 1, t.bytes.size(), f);
    std::fclose(f);
    f = std::fopen((dir + "/config.json").c_str(), "wb");
    std::fwrite(config.data(), 1, config.size(), f);
    std::fclose(f);
    return dir;
}

void remove_checkpoint(const std::string& dir) {
    std::remove((dir + "/model.safetensors").c_str());
    std::remove((dir + "/config.json").c_str());
    ::rmdir(dir.c_str());
}

/* Element i of a tensor is a value that names it: tensor tag + i/1000, exact in bf16 for small i. */
T bf16(const std::string& name, std::vector<int64_t> shape, float tag) {
    T t{ name, "BF16", shape, {} };
    int64_t n = 1;
    for (int64_t s : shape) n *= s;
    t.bytes.resize((size_t)n * 2);
    for (int64_t i = 0; i < n; ++i) rad_store_f32(t.bytes.data(), RAD_BF16, i, tag + (float)(i % 64));
    return t;
}

T fp8(const std::string& name, std::vector<int64_t> shape) {
    T t{ name, "F8_E4M3", shape, {} };
    int64_t n = 1;
    for (int64_t s : shape) n *= s;
    t.bytes.resize((size_t)n);
    for (int64_t i = 0; i < n; ++i) rad_store_f32(t.bytes.data(), RAD_F8E4M3, i, (float)(i % 7) - 3.0f);
    return t;
}

RadNameMap map(const char* declared, int mode, std::vector<const char*> src,
               std::vector<int> index = {}, int concat_dim = 0) {
    RadNameMap m{};
    m.declared = declared;
    m.mode = mode;
    m.n_src = (int)src.size();
    m.concat_dim = concat_dim;
    for (int i = 0; i < 8; ++i) m.src_index[i] = -1;
    for (size_t i = 0; i < src.size(); ++i) m.src[i] = src[i];
    for (size_t i = 0; i < index.size(); ++i) m.src_index[i] = index[i];
    return m;
}

const char* kConfig =
    "{\"model_type\":\"qwen3_5\",\"hidden_size\":256,\"num_hidden_layers\":2,"
    "\"quantization_config\":{\"quant_method\":\"fp8\",\"fmt\":\"e4m3\","
    "\"weight_block_size\":[128,128]}}";

}  /* namespace */

TEST(the_metadata_names_the_architecture_and_the_quantisation) {
    const std::string dir = write_checkpoint({ bf16("x", { 2, 2 }, 0) }, kConfig);
    Checkpoint ck;
    REQUIRE_EQ(ck.open(dir), RAD_OK);
    CHECK_EQ(ck.meta().arch, std::string("qwen35"));
    CHECK_EQ(ck.meta().quant, std::string("fp8_e4m3"));
    CHECK_EQ(ck.meta().m.n_embd, (int64_t)256);
    CHECK_EQ(ck.fp8_block_rows(), (int64_t)128);
    CHECK_EQ(ck.dir(), dir);
    remove_checkpoint(dir);
}

TEST(a_plain_tensor_is_its_own_encoding_and_reads_back_as_itself) {
    const std::string dir = write_checkpoint({ bf16("a.weight", { 4, 8 }, 1.0f) }, "{}");
    Checkpoint ck;
    REQUIRE_EQ(ck.open(dir), RAD_OK);
    Program prog;
    prog.name_map.push_back(map("a", RAD_MAP_COPY, { "a.weight" }));
    CkptWeight w;
    std::string why;
    REQUIRE_EQ(ckpt_resolve(ck, prog, "a", false, &w, &why), RAD_OK);
    CHECK(w.trivial);
    CHECK(rad_enc_is(&w.enc, "plain"));
    CHECK_EQ(w.enc.plane[0].dtype, (uint32_t)RAD_BF16);
    CHECK_EQ(w.rows, (int64_t)4);
    CHECK_EQ(w.cols, (int64_t)8);
    std::vector<float> f(8);
    REQUIRE_EQ(ckpt_rows_f32(w, 2, 1, f.data(), &why), RAD_OK);
    for (int j = 0; j < 8; ++j) CHECK_EQ(f[(size_t)j], 1.0f + (float)((16 + j) % 64));
    std::vector<uint8_t> raw(8 * 2);
    REQUIRE_EQ(ckpt_plane_rows(w, 0, 2, 1, raw.data(), &why), RAD_OK);
    CHECK_EQ(std::memcmp(raw.data(), (const uint8_t*)ck.find("a.weight")->data + 32, 16), 0);
    remove_checkpoint(dir);
}

/* BLOCK-SCALED FP8 IS ONE WEIGHT OF TWO PLANES, and its value is the code times its block's
 * scale -- the scale of the block the element is in, along both axes. */
TEST(block_fp8_is_the_codes_and_their_scale_plane) {
    const int64_t N = 256, K = 384;
    T s = bf16("p.weight_scale_inv", { 2, 3 }, 0.0f);
    for (int i = 0; i < 6; ++i) rad_store_f32(s.bytes.data(), RAD_BF16, i, 0.5f * (float)(i + 1));
    const std::string dir = write_checkpoint({ fp8("p.weight", { N, K }), s }, kConfig);
    Checkpoint ck;
    REQUIRE_EQ(ck.open(dir), RAD_OK);
    Program prog;
    prog.name_map.push_back(map("p", RAD_MAP_COPY, { "p.weight" }));
    CkptWeight w;
    std::string why;
    REQUIRE_EQ(ckpt_resolve(ck, prog, "p", false, &w, &why), RAD_OK);
    REQUIRE(w.trivial);
    CHECK(rad_enc_is(&w.enc, "affine"));
    REQUIRE_EQ(w.enc.n_planes, 2);
    CHECK_EQ(w.enc.plane[1].block[0], (int64_t)128);
    CHECK_EQ(w.enc.plane[1].block[1], (int64_t)128);
    CHECK(rad_enc_valid(&w.enc));
    std::vector<float> f((size_t)(2 * K));
    REQUIRE_EQ(ckpt_rows_f32(w, 129, 2, f.data(), &why), RAD_OK);
    for (int64_t r = 0; r < 2; ++r)
        for (int64_t c : { (int64_t)0, (int64_t)130, (int64_t)383 }) {
            const int64_t i = (129 + r) * K + c;
            const float code = (float)(i % 7) - 3.0f;
            const float scale = 0.5f * (float)(3 + c / 128 + 1);   /* block row 1 */
            CHECK_EQ(f[(size_t)(r * K + c)], code * scale);
        }
    std::vector<uint8_t> sp(3 * 2);
    REQUIRE_EQ(ckpt_plane_rows(w, 1, 1, 1, sp.data(), &why), RAD_OK);
    CHECK_EQ(std::memcmp(sp.data(), s.bytes.data() + 6, 6), 0);
    remove_checkpoint(dir);
}

/* A FUSION JOINS THE PIECES -- and their scale planes -- along dim 0, and refuses a piece that ends
 * inside a scale block, whose scales could not follow it. */
TEST(a_fusion_joins_pieces_and_their_scales) {
    const std::string dir = write_checkpoint(
        { fp8("g.weight", { 256, 128 }), bf16("g.weight_scale_inv", { 2, 1 }, 10.0f),
          fp8("u.weight", { 128, 128 }), bf16("u.weight_scale_inv", { 1, 1 }, 20.0f),
          fp8("odd.weight", { 100, 128 }), bf16("odd.weight_scale_inv", { 1, 1 }, 30.0f) },
        kConfig);
    Checkpoint ck;
    REQUIRE_EQ(ck.open(dir), RAD_OK);
    Program prog;
    prog.name_map.push_back(map("gu", RAD_MAP_CONCAT, { "g.weight", "u.weight" }));
    prog.name_map.push_back(map("bad", RAD_MAP_CONCAT, { "odd.weight", "u.weight" }));
    CkptWeight w;
    std::string why;
    REQUIRE_EQ(ckpt_resolve(ck, prog, "gu", false, &w, &why), RAD_OK);
    CHECK(w.trivial);
    CHECK_EQ(w.shape[0], (int64_t)384);
    std::vector<uint8_t> sp(3 * 2);
    REQUIRE_EQ(ckpt_plane_rows(w, 1, 0, 3, sp.data(), &why), RAD_OK);
    CHECK_EQ(rad_load_f32(sp.data(), RAD_BF16, 0), 10.0f);
    CHECK_EQ(rad_load_f32(sp.data(), RAD_BF16, 1), 11.0f);
    CHECK_EQ(rad_load_f32(sp.data(), RAD_BF16, 2), 20.0f);
    std::vector<float> f(128);
    REQUIRE_EQ(ckpt_rows_f32(w, 300, 1, f.data(), &why), RAD_OK);   /* u's row 44, u's scale */
    CHECK_EQ(f[5], ((float)((44 * 128 + 5) % 7) - 3.0f) * 20.0f);

    REQUIRE_EQ(ckpt_resolve(ck, prog, "bad", false, &w, &why), RAD_OK);
    CHECK(!w.trivial);
    CHECK(w.why.find("inside a scale block") != std::string::npos);
    remove_checkpoint(dir);
}

/* A join along dim 1 interleaves the pieces row by row. */
TEST(a_join_along_columns_interleaves_rows) {
    const std::string dir = write_checkpoint({ bf16("l", { 3, 2 }, 0.0f),
                                               bf16("r", { 3, 4 }, 100.0f) }, "{}");
    Checkpoint ck;
    REQUIRE_EQ(ck.open(dir), RAD_OK);
    Program prog;
    prog.name_map.push_back(map("lr", RAD_MAP_CONCAT, { "l", "r" }, {}, 1));
    CkptWeight w;
    std::string why;
    REQUIRE_EQ(ckpt_resolve(ck, prog, "lr", false, &w, &why), RAD_OK);
    CHECK_EQ(w.shape[1], (int64_t)6);
    std::vector<float> f(6);
    REQUIRE_EQ(ckpt_rows_f32(w, 1, 1, f.data(), &why), RAD_OK);
    const float want[6] = { 2, 3, 104, 105, 106, 107 };
    for (int j = 0; j < 6; ++j) CHECK_EQ(f[(size_t)j], want[j]);
    std::vector<uint8_t> raw(6 * 2);
    REQUIRE_EQ(ckpt_plane_rows(w, 0, 2, 1, raw.data(), &why), RAD_OK);
    CHECK_EQ(rad_load_f32(raw.data(), RAD_BF16, 0), 4.0f);
    CHECK_EQ(rad_load_f32(raw.data(), RAD_BF16, 2), 108.0f);
    remove_checkpoint(dir);
}

/* AN EXPERT IS ITS OWN SLICE OF A STACKED TENSOR -- never the whole stack -- and a stacked FP8
 * tensor's scales are sliced at the same index. */
TEST(an_expert_is_its_slice_of_a_stack) {
    const std::string dir = write_checkpoint(
        { bf16("exps", { 4, 2, 3 }, 0.0f), fp8("q.weight", { 2, 128, 128 }),
          bf16("q.weight_scale_inv", { 2, 1, 1 }, 7.0f) }, kConfig);
    Checkpoint ck;
    REQUIRE_EQ(ck.open(dir), RAD_OK);
    Program prog;
    prog.name_map.push_back(map("e2", RAD_MAP_COPY, { "exps" }, { 2 }));
    prog.name_map.push_back(map("q1", RAD_MAP_COPY, { "q.weight" }, { 1 }));
    prog.name_map.push_back(map("e9", RAD_MAP_COPY, { "exps" }, { 9 }));
    CkptWeight w;
    std::string why;
    REQUIRE_EQ(ckpt_resolve(ck, prog, "e2", false, &w, &why), RAD_OK);
    CHECK_EQ(w.rank, 2u);
    std::vector<float> f(3);
    REQUIRE_EQ(ckpt_rows_f32(w, 1, 1, f.data(), &why), RAD_OK);
    CHECK_EQ(f[0], (float)(2 * 6 + 3));                       /* stack index 2, row 1, col 0 */
    REQUIRE_EQ(ckpt_resolve(ck, prog, "q1", false, &w, &why), RAD_OK);
    CHECK(w.trivial);
    std::vector<uint8_t> sp(2);
    REQUIRE_EQ(ckpt_plane_rows(w, 1, 0, 1, sp.data(), &why), RAD_OK);
    CHECK_EQ(rad_load_f32(sp.data(), RAD_BF16, 0), 8.0f);    /* the second expert's scale */
    CHECK(ckpt_resolve(ck, prog, "e9", false, &w, &why) < 0);
    remove_checkpoint(dir);
}

/* A CHANGED TENSOR IS NAMED AS THE CHECKPOINT NAMES IT. An expert is a slice "exps[2]" of its
 * stack, and it reads "exps"; a block-FP8 weight reads its scale plane as well as its codes, so a
 * changed scale changes the weight. */
TEST(a_weight_reads_the_checkpoint_tensors_its_pieces_and_scales_come_from) {
    const std::string dir = write_checkpoint(
        { bf16("exps", { 4, 2, 3 }, 0.0f), fp8("q.weight", { 2, 128, 128 }),
          bf16("q.weight_scale_inv", { 2, 1, 1 }, 7.0f), bf16("a.weight", { 2, 3 }, 0.0f),
          bf16("b.weight", { 2, 3 }, 0.0f) }, kConfig);
    Checkpoint ck;
    REQUIRE_EQ(ck.open(dir), RAD_OK);
    Program prog;
    prog.name_map.push_back(map("e2", RAD_MAP_COPY, { "exps" }, { 2 }));
    prog.name_map.push_back(map("q1", RAD_MAP_COPY, { "q.weight" }, { 1 }));
    prog.name_map.push_back(map("ab", RAD_MAP_CONCAT, { "a.weight", "b.weight" }));
    CkptWeight e2, q1, ab;
    std::string why;
    REQUIRE_EQ(ckpt_resolve(ck, prog, "e2", false, &e2, &why), RAD_OK);
    REQUIRE_EQ(ckpt_resolve(ck, prog, "q1", false, &q1, &why), RAD_OK);
    REQUIRE_EQ(ckpt_resolve(ck, prog, "ab", false, &ab, &why), RAD_OK);
    CHECK_EQ(ckpt_tensor_of(*e2.pieces[0]), std::string("exps"));

    std::unordered_set<std::string> hit;
    CHECK(ckpt_reads_any(e2, { "exps" }, &hit));
    CHECK(!ckpt_reads_any(e2, { "exps[2]" }));        /* a slice is not a checkpoint name */
    CHECK(ckpt_reads_any(q1, { "q.weight_scale_inv" }, &hit));
    CHECK(ckpt_reads_any(ab, { "b.weight" }, &hit));   /* one changed piece of a join */
    CHECK(!ckpt_reads_any(ab, { "q.weight", "exps" }));
    CHECK_EQ(hit.size(), (size_t)3);
    remove_checkpoint(dir);
}

/* A COPY's sources are alternatives, first present wins: a tied head reads the embedding where
 * the checkpoint ships no head of its own. */
TEST(alternatives_take_the_first_tensor_present) {
    const std::string dir = write_checkpoint({ bf16("embed", { 2, 2 }, 5.0f) }, "{}");
    Checkpoint ck;
    REQUIRE_EQ(ck.open(dir), RAD_OK);
    Program prog;
    prog.name_map.push_back(map("out", RAD_MAP_COPY, { "lm_head", "embed" }));
    prog.name_map.push_back(map("need", RAD_MAP_COPY, { "missing" }));
    CkptWeight w;
    std::string why;
    REQUIRE_EQ(ckpt_resolve(ck, prog, "out", false, &w, &why), RAD_OK);
    REQUIRE_EQ(w.pieces.size(), (size_t)1);
    CHECK_EQ(w.pieces[0]->name, std::string("embed"));
    CHECK_EQ(ckpt_resolve(ck, prog, "need", false, &w, &why), RAD_E_NOTFOUND);
    CHECK(why.find("missing") != std::string::npos);
    CHECK_EQ(ckpt_resolve(ck, prog, "need", true, &w, &why), RAD_E_NOTFOUND);
    CHECK_EQ(ckpt_resolve(ck, prog, "unmapped", false, &w, &why), RAD_E_NOTFOUND);
    remove_checkpoint(dir);
}

/* NOT TRIVIAL, AND SAID WHY: FP8 with no scale beside it, and pieces of different dtypes. */
TEST(what_is_not_served_as_it_is_says_why) {
    const std::string dir = write_checkpoint({ fp8("bare", { 128, 128 }),
                                               bf16("b16", { 128, 128 }, 0.0f) }, kConfig);
    Checkpoint ck;
    REQUIRE_EQ(ck.open(dir), RAD_OK);
    Program prog;
    prog.name_map.push_back(map("bare", RAD_MAP_COPY, { "bare" }));
    prog.name_map.push_back(map("mixed", RAD_MAP_CONCAT, { "bare", "b16" }));
    CkptWeight w;
    std::string why;
    REQUIRE_EQ(ckpt_resolve(ck, prog, "bare", false, &w, &why), RAD_OK);
    CHECK(!w.trivial);
    CHECK(w.why.find("no scale plane") != std::string::npos);
    REQUIRE_EQ(ckpt_resolve(ck, prog, "mixed", false, &w, &why), RAD_OK);
    CHECK(!w.trivial);
    remove_checkpoint(dir);
}

RAD_TEST_MAIN()

/* ================================================================== imported quantisations */
namespace {

/* A GPTQ-packed weight as auto_gptq's packer writes it: code (row n, column k) is the signed value
 * `q[n][k]`, stored unsigned with zero point 2^(bits-1) as field k % (32/bits) of int32 word
 * [k / (32/bits), n]; `qzeros` holds the v1 zero point (less one) in every field; `scales` is
 * f16 [K / group, N]. */
struct Gptq {
    std::vector<std::vector<int>> q;     /* [N][K] signed codes */
    std::vector<float> scale;            /* [G * N], row g of the scale tensor first */
    int64_t N = 0, K = 0, group = 0;
    int bits = 4;
};

Gptq make_gptq(int64_t N, int64_t K, int64_t group, int bits, int seed) {
    Gptq g;
    g.N = N; g.K = K; g.group = group; g.bits = bits;
    const int lo = -(1 << (bits - 1)), span = 1 << bits;
    g.q.assign((size_t)N, std::vector<int>((size_t)K));
    uint32_t x = 2463534242u ^ (uint32_t)seed;
    for (auto& row : g.q)
        for (int& v : row) {
            x ^= x << 13; x ^= x >> 17; x ^= x << 5;
            v = lo + (int)(x % (uint32_t)span);
        }
    /* f16 scales with low mantissa bits set and both signs, so the bf16 rounding and the sign
     * are both seen. */
    for (int64_t i = 0; i < (K / group) * N; ++i)
        g.scale.push_back((i % 3 == 1 ? -1.0f : 1.0f) * (0.0123f + 0.00071f * (float)i));
    return g;
}

void add_gptq(std::vector<T>* ts, const std::string& base, const Gptq& g, bool with_gidx = false) {
    const int64_t per = 32 / g.bits, G = g.K / g.group;
    T qw{ base + ".qweight", "I32", { g.K / per, g.N }, {} };
    qw.bytes.assign((size_t)(g.K / per * g.N * 4), 0);
    for (int64_t n = 0; n < g.N; ++n)
        for (int64_t k = 0; k < g.K; ++k) {
            const uint32_t u = (uint32_t)(g.q[(size_t)n][(size_t)k] + (1 << (g.bits - 1)));
            uint32_t* w = (uint32_t*)qw.bytes.data() + (k / per) * g.N + n;
            *w |= u << ((k % per) * g.bits);
        }
    T sc{ base + ".scales", "F16", { G, g.N }, {} };
    sc.bytes.resize((size_t)(G * g.N * 2));
    for (int64_t i = 0; i < G * g.N; ++i) rad_store_f32(sc.bytes.data(), RAD_F16, i, g.scale[(size_t)i]);
    T qz{ base + ".qzeros", "I32", { G, g.N * g.bits / 32 }, {} };
    qz.bytes.resize((size_t)(G * g.N * g.bits / 32 * 4));
    for (size_t i = 0; i < qz.bytes.size() / 4; ++i)
        ((uint32_t*)qz.bytes.data())[i] = g.bits == 4 ? 0x77777777u : 0x7f7f7f7fu;
    ts->push_back(qw);
    ts->push_back(sc);
    ts->push_back(qz);
    if (with_gidx) {
        T gi{ base + ".g_idx", "I32", { g.K }, {} };
        gi.bytes.resize((size_t)g.K * 4);
        for (int64_t k = 0; k < g.K; ++k) ((int32_t*)gi.bytes.data())[k] = (int32_t)(k / g.group);
        ts->push_back(gi);
    }
}

/* The scale the f16 tensor actually holds. */
float f16_of(float v) {
    uint16_t h;
    rad_store_f32(&h, RAD_F16, 0, v);
    return rad_load_f32(&h, RAD_F16, 0);
}

const char* kAutoRound =
    "{\"model_type\":\"qwen4_exp\",\"quantization_config\":{\"quant_method\":\"auto-round\","
    "\"packing_format\":\"auto_round:auto_gptq\",\"bits\":4,\"group_size\":128,\"sym\":true}}";

}  /* namespace */

/* AUTOROUND'S GPTQ EXPORT IS IMPORTED: gate and up -- each its own GPTQ-packed int4 weight --
 * become one `i4*bf16[1x128]` weight whose codes are the shipped ones, signed, row by row, and
 * whose scales are the shipped f16 ones rounded to bf16. Read as f32 it is the shipped value:
 * the code times the f16 scale. Codeberg #29. */
TEST(an_autoround_int4_weight_is_imported_as_its_codes_and_scales) {
    const Gptq gate = make_gptq(8, 256, 128, 4, 1), up = make_gptq(16, 256, 128, 4, 2);
    std::vector<T> ts;
    add_gptq(&ts, "e.0.gate_proj", gate);
    add_gptq(&ts, "e.0.up_proj", up, /*with_gidx=*/true);
    const std::string dir = write_checkpoint(ts, kAutoRound);
    Checkpoint ck;
    REQUIRE_EQ(ck.open(dir), RAD_OK);
    CHECK(ck.imports() == Checkpoint::Import::Gptq);
    CHECK_EQ(ck.meta().quant, std::string("auto-round"));
    Program prog;
    prog.name_map.push_back(map("gu", RAD_MAP_CONCAT, { "e.0.gate_proj.weight", "e.0.up_proj.weight" }));
    CkptWeight w;
    std::string why;
    REQUIRE_EQ(ckpt_resolve(ck, prog, "gu", false, &w, &why), RAD_OK);
    CHECK(!w.trivial);
    REQUIRE(w.imported);
    CHECK_EQ(w.rows, (int64_t)24);
    CHECK_EQ(w.cols, (int64_t)256);
    REQUIRE_EQ(w.enc.n_planes, 2);
    CHECK_EQ(w.enc.plane[0].dtype, (uint32_t)RAD_I4);
    CHECK_EQ(w.enc.plane[1].dtype, (uint32_t)RAD_BF16);
    CHECK_EQ(w.enc.plane[1].block[0], (int64_t)1);
    CHECK_EQ(w.enc.plane[1].block[1], (int64_t)128);
    CHECK(rad_enc_valid(&w.enc));

    /* every row, both pieces: codes, scales and the f32 value */
    std::vector<uint8_t> codes((size_t)(24 * 128)), scales((size_t)(24 * 2 * 2));
    REQUIRE_EQ(ckpt_plane_rows(w, 0, 0, 24, codes.data(), &why), RAD_OK);
    REQUIRE_EQ(ckpt_plane_rows(w, 1, 0, 24, scales.data(), &why), RAD_OK);
    std::vector<float> f((size_t)(24 * 256));
    REQUIRE_EQ(ckpt_rows_f32(w, 0, 24, f.data(), &why), RAD_OK);
    int bad_code = 0, bad_scale = 0, bad_f = 0;
    for (int64_t r = 0; r < 24; ++r) {
        const Gptq& g = r < 8 ? gate : up;
        const int64_t n = r < 8 ? r : r - 8;
        for (int64_t k = 0; k < 256; ++k) {
            const int q = g.q[(size_t)n][(size_t)k];
            if ((int)rad_load_f32(codes.data(), RAD_I4, r * 256 + k) != q) ++bad_code;
            const float s = f16_of(g.scale[(size_t)((k / 128) * g.N + n)]);
            if (f[(size_t)(r * 256 + k)] != (float)q * s) ++bad_f;
        }
        for (int64_t c = 0; c < 2; ++c) {
            const float s = f16_of(g.scale[(size_t)(c * g.N + n)]);
            if (rad_load_f32(scales.data(), RAD_BF16, r * 2 + c) !=
                rad_load_f32(std::vector<uint16_t>{ rad_f32_to_bf16(s) }.data(), RAD_BF16, 0))
                ++bad_scale;
        }
    }
    CHECK_EQ(bad_code, 0);
    CHECK_EQ(bad_scale, 0);
    CHECK_EQ(bad_f, 0);
    /* a part of the plane, from the middle of the second piece */
    std::vector<uint8_t> part(128);
    REQUIRE_EQ(ckpt_plane_rows(w, 0, 13, 1, part.data(), &why), RAD_OK);
    CHECK_EQ(std::memcmp(part.data(), codes.data() + 13 * 128, 128), 0);
    remove_checkpoint(dir);
}

/* AN INT8 PER-CHANNEL WEIGHT (one scale a row, g_idx all zero) is `i8*bf16[1x128]` with its row's
 * scale in every block. */
TEST(a_gptq_int8_per_channel_weight_repeats_its_scale) {
    const Gptq g = make_gptq(4, 384, 384, 8, 3);
    std::vector<T> ts;
    add_gptq(&ts, "o_proj", g, /*with_gidx=*/true);
    const std::string dir = write_checkpoint(ts, kAutoRound);
    Checkpoint ck;
    REQUIRE_EQ(ck.open(dir), RAD_OK);
    Program prog;
    prog.name_map.push_back(map("o", RAD_MAP_COPY, { "o_proj.weight" }));
    CkptWeight w;
    std::string why;
    REQUIRE_EQ(ckpt_resolve(ck, prog, "o", false, &w, &why), RAD_OK);
    REQUIRE(w.imported);
    CHECK_EQ(w.enc.plane[0].dtype, (uint32_t)RAD_I8);
    std::vector<uint8_t> codes(4 * 384), scales(4 * 3 * 2);
    REQUIRE_EQ(ckpt_plane_rows(w, 0, 0, 4, codes.data(), &why), RAD_OK);
    REQUIRE_EQ(ckpt_plane_rows(w, 1, 0, 4, scales.data(), &why), RAD_OK);
    int bad = 0;
    for (int64_t n = 0; n < 4; ++n) {
        for (int64_t k = 0; k < 384; ++k)
            if ((int8_t)codes[(size_t)(n * 384 + k)] != g.q[(size_t)n][(size_t)k]) ++bad;
        for (int c = 0; c < 3; ++c)
            if (rad_load_f32(scales.data(), RAD_BF16, n * 3 + c) !=
                rad_load_f32(scales.data(), RAD_BF16, n * 3))
                ++bad;
    }
    CHECK_EQ(bad, 0);
    remove_checkpoint(dir);
}

/* WHAT AN IMPORT CANNOT READ IT REFUSES BY NAME: an asymmetric zero point, an act-order group
 * index, a group narrower than the served 128, and a packing it does not know. */
TEST(an_import_refuses_what_it_cannot_read) {
    auto refused = [](void (*edit)(std::vector<T>*), const char* config, const char* expect) {
        Gptq g = make_gptq(8, 256, 128, 4, 4);
        std::vector<T> ts;
        add_gptq(&ts, "x", g, true);
        edit(&ts);
        const std::string dir = write_checkpoint(ts, config);
        Checkpoint ck;
        REQUIRE_EQ(ck.open(dir), RAD_OK);
        Program prog;
        prog.name_map.push_back(map("x", RAD_MAP_COPY, { "x.weight" }));
        CkptWeight w;
        std::string why;
        const int rc = ckpt_resolve(ck, prog, "x", false, &w, &why);
        CHECK(rc < 0);
        CHECK(why.find(expect) != std::string::npos);
        remove_checkpoint(dir);
    };
    refused([](std::vector<T>* ts) { (*ts)[2].bytes[0] = 0x76; }, kAutoRound, "symmetric");
    refused([](std::vector<T>* ts) { ((int32_t*)(*ts)[3].bytes.data())[0] = 1; }, kAutoRound,
            "act-order");

    /* group 64: four scale rows for 256 columns */
    {
        Gptq g = make_gptq(8, 256, 64, 4, 5);
        std::vector<T> ts;
        add_gptq(&ts, "x", g);
        const std::string dir = write_checkpoint(ts, kAutoRound);
        Checkpoint ck;
        REQUIRE_EQ(ck.open(dir), RAD_OK);
        Program prog;
        prog.name_map.push_back(map("x", RAD_MAP_COPY, { "x.weight" }));
        CkptWeight w;
        std::string why;
        CHECK(ckpt_resolve(ck, prog, "x", false, &w, &why) < 0);
        CHECK(why.find("multiple of 128") != std::string::npos);
        remove_checkpoint(dir);
    }
    /* AutoRound's own packing is another layout, and the checkpoint is not imported at all. */
    {
        const std::string dir = write_checkpoint(
            { bf16("y", { 2, 2 }, 0) },
            "{\"model_type\":\"qwen4_exp\",\"quantization_config\":{\"quant_method\":"
            "\"auto-round\",\"packing_format\":\"auto_round\"}}");
        Checkpoint ck;
        REQUIRE_EQ(ck.open(dir), RAD_OK);
        CHECK(ck.imports() == Checkpoint::Import::None);
        CHECK(ck.import_refusal().find("auto_round:auto_gptq") != std::string::npos);
        remove_checkpoint(dir);
    }
}

/* QUARK'S MXFP4 IS `fp4_e2m1*e8m0[1x32]` BYTE FOR BYTE, and so trivial: the U8 codes and U8
 * exponents are the planes, gate and up join along rows, and the f32 value is the e2m1 code times
 * 2^(e - 127). */
TEST(a_quark_mxfp4_weight_is_its_codes_and_exponents) {
    const int64_t K = 64;
    auto codes = [&](const std::string& n, int64_t N, int seed) {
        T t{ n, "U8", { N, K / 2 }, {} };
        for (int64_t i = 0; i < N * K / 2; ++i) t.bytes.push_back((uint8_t)((i * 37 + seed) & 0xff));
        return t;
    };
    auto exps = [&](const std::string& n, int64_t N, int seed) {
        T t{ n, "U8", { N, K / 32 }, {} };
        for (int64_t i = 0; i < N * K / 32; ++i) t.bytes.push_back((uint8_t)(120 + (i + seed) % 9));
        return t;
    };
    const T g = codes("g.weight", 2, 1), gs = exps("g.weight_scale", 2, 1);
    const T u = codes("u.weight", 3, 5), us = exps("u.weight_scale", 3, 4);
    const std::string dir = write_checkpoint(
        { g, gs, u, us, bf16("n.weight", { 4 }, 1.0f) },
        "{\"model_type\":\"qwen3_5\",\"quantization_config\":{\"quant_method\":\"quark\","
        "\"global_quant_config\":{\"weight\":{\"dtype\":\"fp4\",\"qscheme\":\"per_group\","
        "\"group_size\":32,\"scale_format\":\"e8m0\"}},\"layer_quant_config\":{}}}");
    Checkpoint ck;
    REQUIRE_EQ(ck.open(dir), RAD_OK);
    CHECK(ck.imports() == Checkpoint::Import::QuarkMx);
    Program prog;
    prog.name_map.push_back(map("gu", RAD_MAP_CONCAT, { "g.weight", "u.weight" }));
    prog.name_map.push_back(map("n", RAD_MAP_COPY, { "n.weight" }));
    CkptWeight w;
    std::string why;
    REQUIRE_EQ(ckpt_resolve(ck, prog, "gu", false, &w, &why), RAD_OK);
    REQUIRE(w.trivial);
    CHECK(!w.imported);
    CHECK_EQ(w.rows, (int64_t)5);
    CHECK_EQ(w.cols, K);
    REQUIRE_EQ(w.enc.n_planes, 2);
    CHECK_EQ(w.enc.plane[0].dtype, (uint32_t)RAD_FP4E2M1);
    CHECK_EQ(w.enc.plane[1].dtype, (uint32_t)RAD_E8M0);
    CHECK_EQ(w.enc.plane[1].block[1], (int64_t)32);
    CHECK(rad_enc_valid(&w.enc));
    std::vector<uint8_t> cp(5 * K / 2), sp(5 * K / 32);
    REQUIRE_EQ(ckpt_plane_rows(w, 0, 0, 5, cp.data(), &why), RAD_OK);
    REQUIRE_EQ(ckpt_plane_rows(w, 1, 0, 5, sp.data(), &why), RAD_OK);
    CHECK_EQ(std::memcmp(cp.data(), g.bytes.data(), g.bytes.size()), 0);
    CHECK_EQ(std::memcmp(cp.data() + g.bytes.size(), u.bytes.data(), u.bytes.size()), 0);
    CHECK_EQ(std::memcmp(sp.data() + gs.bytes.size(), us.bytes.data(), us.bytes.size()), 0);
    std::vector<float> f((size_t)K);
    REQUIRE_EQ(ckpt_rows_f32(w, 3, 1, f.data(), &why), RAD_OK);       /* u's row 1 */
    int bad = 0;
    for (int64_t k = 0; k < K; ++k) {
        const uint8_t b = u.bytes[(size_t)(K / 2 + k / 2)];
        const float code = rad_fp4e2m1_to_f32((uint8_t)(k % 2 ? b >> 4 : b & 15));
        const float sc = std::ldexp(1.0f, (int)us.bytes[(size_t)(K / 32 + k / 32)] - 127);
        if (f[(size_t)k] != code * sc) ++bad;
    }
    CHECK_EQ(bad, 0);
    /* an excluded layer is a plain tensor */
    REQUIRE_EQ(ckpt_resolve(ck, prog, "n", false, &w, &why), RAD_OK);
    CHECK(w.trivial && rad_enc_is(&w.enc, "plain"));
    remove_checkpoint(dir);
}

/* One NVFP4 linear: packed e2m1 codes, an e4m3 table, an f32 divisor. */
void add_nvfp4(std::vector<T>* ts, const std::string& base, int64_t N, int64_t K, float d, int seed,
               bool with_global = true) {
    T p{ base + ".weight_packed", "U8", { N, K / 2 }, {} };
    for (int64_t i = 0; i < N * K / 2; ++i) p.bytes.push_back((uint8_t)((i * 37 + seed) & 0xff));
    ts->push_back(p);
    T s{ base + ".weight_scale", "F8_E4M3", { N, K / 16 }, {} };
    for (int64_t i = 0; i < N * K / 16; ++i) {
        float v = 1.0f + (float)((i * 7 + seed) % 23) / 8.0f;
        uint8_t b;
        rad_store_f32(&b, RAD_F8E4M3, 0, v);
        s.bytes.push_back(b);
    }
    ts->push_back(s);
    if (with_global) {
        T g{ base + ".weight_global_scale", "F32", { 1 }, {} };
        g.bytes.resize(4);
        std::memcpy(g.bytes.data(), &d, 4);
        ts->push_back(g);
    }
}

/* NVFP4 AS SHIPPED IS IMPORTED: the packed e2m1 codes beside the e4m3 table divided by the
 * tensor's f32 global -- which the checkpoint stores as the divisor its quantiser scaled by, so
 * the value a row decodes to is the code times e4m3-over-divisor, a piece's own: a fused weight
 * joins the pieces' tables along rows and keeps each piece's global, the way vLLM's online
 * requantisation dequantises per partition of a merged linear. A tensor without its three reads
 * stock; the codes keep the file's low-nibble-first order, the value a code times its table's. */
TEST(an_nvfp4_weight_is_its_codes_and_divided_table) {
    std::vector<T> ts;
    add_nvfp4(&ts, "g", 2, 64, 8.0f, 1);
    add_nvfp4(&ts, "u", 3, 64, 4.0f, 5);
    ts.push_back(bf16("n.weight", { 4 }, 1.0f));
    const std::string dir = write_checkpoint(
        ts,
        "{\"model_type\":\"qwen3_5\",\"quantization_config\":{\"quant_method\":"
        "\"compressed-tensors\",\"weights\":{\"type\":\"fp4\",\"group_size\":16}}}");
    Checkpoint ck;
    REQUIRE_EQ(ck.open(dir), RAD_OK);
    CHECK(ck.imports() == Checkpoint::Import::Nvfp4);
    Program prog;
    prog.name_map.push_back(map("gu", RAD_MAP_CONCAT, { "g.weight", "u.weight" }));
    prog.name_map.push_back(map("n", RAD_MAP_COPY, { "n.weight" }));
    CkptWeight w;
    std::string why;
    REQUIRE_EQ(ckpt_resolve(ck, prog, "gu", false, &w, &why), RAD_OK);
    CHECK(!w.trivial);
    CHECK(w.imported);
    CHECK_EQ(w.rows, (int64_t)5);
    CHECK_EQ(w.cols, (int64_t)64);
    REQUIRE_EQ(w.enc.n_planes, 2);
    CHECK_EQ(w.enc.plane[0].dtype, (uint32_t)RAD_FP4E2M1);
    CHECK_EQ(w.enc.plane[1].dtype, (uint32_t)RAD_F32);
    CHECK_EQ(w.enc.plane[1].block[0], (int64_t)1);
    CHECK_EQ(w.enc.plane[1].block[1], (int64_t)16);
    CHECK(rad_enc_valid(&w.enc));
    /* plane 0 is the file's bytes, the pieces in order; plane 1 is e4m3 over the piece's divisor */
    const T& gp = ts[0], gs = ts[1], up = ts[3], us = ts[4];
    std::vector<uint8_t> cp(5 * 32), tp(5 * 4 * 4);
    REQUIRE_EQ(ckpt_plane_rows(w, 0, 0, 5, cp.data(), &why), RAD_OK);
    REQUIRE_EQ(ckpt_plane_rows(w, 1, 0, 5, tp.data(), &why), RAD_OK);
    CHECK_EQ(std::memcmp(cp.data(), gp.bytes.data(), gp.bytes.size()), 0);
    CHECK_EQ(std::memcmp(cp.data() + gp.bytes.size(), up.bytes.data(), up.bytes.size()), 0);
    int bad_tab = 0;
    for (int64_t r = 0; r < 5; ++r) {
        const T& s = r < 2 ? gs : us;
        const int64_t n = r < 2 ? r : r - 2;
        const float d = r < 2 ? 8.0f : 4.0f;
        for (int64_t c = 0; c < 4; ++c)
            if (rad_load_f32(tp.data(), RAD_F32, r * 4 + c) !=
                rad_fp8e4m3_to_f32(s.bytes[(size_t)(n * 4 + c)]) / d)
                ++bad_tab;
    }
    CHECK_EQ(bad_tab, 0);
    /* the f32 value: the code -- in the file's low-nibble-first order -- times that table;
     * logical row 3 is u's row 1, so the reference indexes u's second row */
    std::vector<float> f((size_t)64);
    REQUIRE_EQ(ckpt_rows_f32(w, 3, 1, f.data(), &why), RAD_OK);
    int bad = 0;
    for (int64_t k = 0; k < 64; ++k) {
        const uint8_t b = up.bytes[(size_t)(32 + k / 2)];
        const float code = rad_fp4e2m1_to_f32((uint8_t)(k % 2 ? b >> 4 : b & 15));
        const float sc = rad_fp8e4m3_to_f32(us.bytes[(size_t)(4 + k / 16)]) / 4.0f;
        if (f[(size_t)k] != code * sc) ++bad;
    }
    CHECK_EQ(bad, 0);
    /* a tensor without its three is stock */
    REQUIRE_EQ(ckpt_resolve(ck, prog, "n", false, &w, &why), RAD_OK);
    CHECK(w.trivial && rad_enc_is(&w.enc, "plain"));
    remove_checkpoint(dir);
}

/* AN NVFP4 LINEAR WITHOUT ITS GLOBAL -- OR WITH A ZERO ONE -- IS REFUSED BY NAME AT THE WEIGHT:
 * the value cannot be read without the divisor, and a zero divisor cannot be folded. */
TEST(an_nvfp4_weight_missing_its_global_is_refused) {
    std::vector<T> ts;
    add_nvfp4(&ts, "g", 2, 64, 8.0f, 1, /*with_global=*/false);
    add_nvfp4(&ts, "z", 2, 64, 0.0f, 2);
    const std::string dir = write_checkpoint(
        ts,
        "{\"model_type\":\"qwen3_5\",\"quantization_config\":{\"quant_method\":"
        "\"compressed-tensors\",\"weights\":{\"type\":\"fp4\",\"group_size\":16}}}");
    Checkpoint ck;
    REQUIRE_EQ(ck.open(dir), RAD_OK);
    CHECK(ck.imports() == Checkpoint::Import::Nvfp4);
    Program prog;
    prog.name_map.push_back(map("g", RAD_MAP_COPY, { "g.weight" }));
    prog.name_map.push_back(map("z", RAD_MAP_COPY, { "z.weight" }));
    CkptWeight w;
    std::string why;
    CHECK(ckpt_resolve(ck, prog, "g", false, &w, &why) < 0);
    CHECK(why.find("weight_global_scale") != std::string::npos);
    why.clear();
    CHECK(ckpt_resolve(ck, prog, "z", false, &w, &why) < 0);
    CHECK(why.find("divisor") != std::string::npos);
    remove_checkpoint(dir);
}

/* THE CONFIG DECLARES THE CANDIDATE: a compressed-tensors / modelopt config that is not the NVFP4
 * signature -- fp8 weights, a 128-wide group -- is not imported at all, and says so by name. */
TEST(a_compressed_tensors_release_that_is_not_nvfp4_is_not_imported) {
    const char* cfg =
        "{\"model_type\":\"qwen3_5\",\"quantization_config\":{\"quant_method\":"
        "\"compressed-tensors\",\"weights\":{\"type\":\"fp8\",\"group_size\":128}}}";
    const std::string dir = write_checkpoint({ bf16("x", { 2, 2 }, 0) }, cfg);
    Checkpoint ck;
    REQUIRE_EQ(ck.open(dir), RAD_OK);
    CHECK(ck.imports() == Checkpoint::Import::None);
    CHECK(ck.import_refusal().find("fp4") != std::string::npos);
    remove_checkpoint(dir);
}

/* THE SAME IMPORT IN COMPRESSED-TENSORS' OWN SCHEMA: the real releases state the NVFP4 group as
 * `format: nvfp4-pack-quantized` beside an fp8 `float-quantized` group under a top-level
 * `format` of `mixed-precision` -- SOME group, not all, is the test, and the decode is the one
 * the flat schema already proves. */
TEST(an_nvfp4_release_in_the_config_groups_schema_is_imported) {
    std::vector<T> ts;
    add_nvfp4(&ts, "g", 2, 64, 8.0f, 1);
    add_nvfp4(&ts, "u", 3, 64, 4.0f, 5);
    ts.push_back(bf16("n.weight", { 4 }, 1.0f));
    const char* cfg =
        "{\"model_type\":\"qwen3_5\",\"quantization_config\":{\"quant_method\":"
        "\"compressed-tensors\",\"format\":\"mixed-precision\",\"config_groups\":{"
        "\"group_0\":{\"format\":\"float-quantized\",\"weights\":{\"num_bits\":8}},"
        "\"group_1\":{\"format\":\"nvfp4-pack-quantized\",\"weights\":{\"num_bits\":4,"
        "\"group_size\":16,\"scale_dtype\":\"torch.float8_e4m3fn\"}}}}}";
    const std::string dir = write_checkpoint(ts, cfg);
    Checkpoint ck;
    REQUIRE_EQ(ck.open(dir), RAD_OK);
    CHECK(ck.imports() == Checkpoint::Import::Nvfp4);
    Program prog;
    prog.name_map.push_back(map("gu", RAD_MAP_CONCAT, { "g.weight", "u.weight" }));
    prog.name_map.push_back(map("n", RAD_MAP_COPY, { "n.weight" }));
    CkptWeight w;
    std::string why;
    REQUIRE_EQ(ckpt_resolve(ck, prog, "gu", false, &w, &why), RAD_OK);
    CHECK(w.imported && !w.trivial);
    CHECK_EQ(w.rows, (int64_t)5);
    REQUIRE_EQ(w.enc.n_planes, 2);
    CHECK_EQ(w.enc.plane[0].dtype, (uint32_t)RAD_FP4E2M1);
    CHECK_EQ(w.enc.plane[1].dtype, (uint32_t)RAD_F32);
    CHECK_EQ(w.enc.plane[1].block[1], (int64_t)16);
    /* the folded table: u's e4m3 divided by u's divisor, row 3 being u's row 1 */
    const T& us = ts[4];
    std::vector<uint8_t> tp(5 * 4 * 4);
    REQUIRE_EQ(ckpt_plane_rows(w, 1, 0, 5, tp.data(), &why), RAD_OK);
    int bad = 0;
    for (int64_t c = 0; c < 4; ++c)
        if (rad_load_f32(tp.data(), RAD_F32, 3 * 4 + c) !=
            rad_fp8e4m3_to_f32(us.bytes[(size_t)(1 * 4 + c)]) / 4.0f)
            ++bad;
    CHECK_EQ(bad, 0);
    /* a tensor without its three reads stock */
    REQUIRE_EQ(ckpt_resolve(ck, prog, "n", false, &w, &why), RAD_OK);
    CHECK(w.trivial && rad_enc_is(&w.enc, "plain"));
    remove_checkpoint(dir);
}

/* A config_groups RELEASE IN NO NVFP4 FORMAT IS NOT IMPORTED, and the refusal names the
 * formats it does carry. */
TEST(an_nvfp4_candidate_without_an_nvfp4_group_is_not_imported) {
    const char* cfg =
        "{\"model_type\":\"qwen3_5\",\"quantization_config\":{\"quant_method\":"
        "\"compressed-tensors\",\"format\":\"float-quantized\",\"config_groups\":{"
        "\"group_0\":{\"format\":\"float-quantized\",\"weights\":{\"num_bits\":8}}}}}";
    const std::string dir = write_checkpoint({ bf16("x", { 2, 2 }, 0) }, cfg);
    Checkpoint ck;
    REQUIRE_EQ(ck.open(dir), RAD_OK);
    CHECK(ck.imports() == Checkpoint::Import::None);
    CHECK(ck.import_refusal().find("float-quantized") != std::string::npos);
    remove_checkpoint(dir);
}

/* PER-CHANNEL FP8 IS NEVER TRIVIAL, and a recipe quantises from the value the core decodes:
 * the code times the row's scale, read from the [N, 1] plane beside the codes. */
TEST(a_per_channel_fp8_weight_is_decoded_but_never_trivial) {
    T w8{ "a.weight", "F8_E4M3", { 2, 128 }, {} };
    for (int64_t i = 0; i < 256; ++i) {
        uint8_t b;
        rad_store_f32(&b, RAD_F8E4M3, 0, (float)((i % 7) - 3));
        w8.bytes.push_back(b);
    }
    T s{ "a.weight_scale", "BF16", { 2, 1 }, {} };
    const float sv[2] = { 4.0f, -2.0f };
    for (int64_t r = 0; r < 2; ++r) {
        /* THE TOP HALF OF AN f32. A 2-byte memcpy of the float takes the BOTTOM two bytes,
         * which are 0 for 4.0 and -2.0 -- the scale the weight is read as, silently zero. */
        const uint16_t h = rad_f32_to_bf16(sv[r]);
        s.bytes.push_back((uint8_t)(h & 0xff));
        s.bytes.push_back((uint8_t)(h >> 8));
    }
    const std::string dir = write_checkpoint({ w8, s }, kConfig);
    Checkpoint ck;
    REQUIRE_EQ(ck.open(dir), RAD_OK);
    Program prog;
    prog.name_map.push_back(map("a", RAD_MAP_COPY, { "a.weight" }));
    CkptWeight w;
    std::string why;
    REQUIRE_EQ(ckpt_resolve(ck, prog, "a", false, &w, &why), RAD_OK);
    CHECK(!w.trivial);
    CHECK(w.why.find("per-channel") != std::string::npos);
    REQUIRE_EQ(w.enc.n_planes, 2);
    CHECK_EQ(w.enc.plane[1].block[0], (int64_t)1);
    CHECK_EQ(w.enc.plane[1].block[1], (int64_t)128);
    /* row 1's value: its code -- exact in e4m3 over this range -- times its row's scale */
    std::vector<float> f(128);
    REQUIRE_EQ(ckpt_rows_f32(w, 1, 1, f.data(), &why), RAD_OK);
    int bad = 0;
    for (int64_t k = 0; k < 128; ++k)
        if (f[k] != (float)(((128 + k) % 7) - 3) * -2.0f) ++bad;
    CHECK_EQ(bad, 0);
    remove_checkpoint(dir);
}

/* A FUSION OF PER-CHANNEL PIECES DECODES EACH PIECE'S OWN SCALE: the pieces' [N, 1] planes are
 * distinct, indexed at the piece's own rows. */
TEST(a_fusion_of_per_channel_fp8_pieces_decodes_each_scale) {
    auto piece = [](const char* n, int64_t N, int seed) {
        T c{ n, "F8_E4M3", { N, 8 }, {} };
        for (int64_t i = 0; i < N * 8; ++i) {
            uint8_t b;
            rad_store_f32(&b, RAD_F8E4M3, 0, (float)((i % 5) - 2));
            c.bytes.push_back(b);
        }
        T s{ std::string(n).substr(0, std::string(n).size() - 7) + ".weight_scale", "BF16", { N, 1 }, {} };
        for (int64_t r = 0; r < N; ++r) {
            const float v = 1.0f + (float)(r % 3) + (float)seed;   /* 1..4, exact in bf16 */
            const uint16_t h = rad_f32_to_bf16(v);
            s.bytes.push_back((uint8_t)(h & 0xff));
            s.bytes.push_back((uint8_t)(h >> 8));
        }
        return std::pair<T, T>(c, s);
    };
    auto p0 = piece("g.weight", 4, 0), p1 = piece("u.weight", 3, 1);
    const std::string dir = write_checkpoint({ p0.first, p0.second, p1.first, p1.second }, kConfig);
    Checkpoint ck;
    REQUIRE_EQ(ck.open(dir), RAD_OK);
    Program prog;
    prog.name_map.push_back(map("gu", RAD_MAP_CONCAT, { "g.weight", "u.weight" }));
    CkptWeight w;
    std::string why;
    REQUIRE_EQ(ckpt_resolve(ck, prog, "gu", false, &w, &why), RAD_OK);
    CHECK(!w.trivial);
    /* u's row 2: its own scale -- 1 + 2 + 1 -- not g's; u's own element 16 + k, a value u's
     * bytes index from 0, not from where g's numbering would continue */
    std::vector<float> f(8);
    REQUIRE_EQ(ckpt_rows_f32(w, 6, 1, f.data(), &why), RAD_OK);
    int bad = 0;
    for (int64_t k = 0; k < 8; ++k)
        if (f[k] != (float)(((16 + k) % 5) - 2) * 4.0f) ++bad;
    CHECK_EQ(bad, 0);
    remove_checkpoint(dir);
}

/* A COLUMN JOIN OF PER-CHANNEL PIECES HAS NO ENCODING THAT READS IT -- the pieces' widths differ,
 * which one [1, K] plane cannot cover -- so the read refuses with the reason rather than index a
 * [N, 1] plane as a [N, K/128] one. */
TEST(a_column_join_of_per_channel_fp8_has_no_encoding_and_refuses_the_read) {
    T l = fp8("l.weight", { 4, 8 }), r = fp8("r.weight", { 4, 16 });
    const std::string dir = write_checkpoint(
        { l, bf16("l.weight_scale", { 4, 1 }, 1.0f), r, bf16("r.weight_scale", { 4, 1 }, 1.0f) },
        kConfig);
    Checkpoint ck;
    REQUIRE_EQ(ck.open(dir), RAD_OK);
    Program prog;
    prog.name_map.push_back(map("lr", RAD_MAP_CONCAT, { "l.weight", "r.weight" }, {}, 1));
    CkptWeight w;
    std::string why;
    REQUIRE_EQ(ckpt_resolve(ck, prog, "lr", false, &w, &why), RAD_OK);
    CHECK(!w.trivial);
    CHECK(w.why.find("per-channel") != std::string::npos);
    CHECK_EQ(w.enc.n_planes, (int)0);
    std::vector<float> f(24);
    CHECK(ckpt_rows_f32(w, 0, 1, f.data(), &why) < 0);
    remove_checkpoint(dir);
}

/* A JOIN THAT ENDS INSIDE THE CHECKPOINT'S DECLARED BLOCK is not served as it stands, but it is
 * still READ with the block the checkpoint states -- not a guessed 128 -- so the value a recipe
 * quantises from is the piece's own. */
TEST(a_fp8_join_that_ends_inside_a_custom_block_reads_with_that_block) {
    const char* cfg =
        "{\"model_type\":\"qwen3_5\",\"quantization_config\":{\"quant_method\":\"fp8\","
        "\"fmt\":\"e4m3\",\"weight_block_size\":[64,64]}}";
    T p = fp8("p.weight", { 100, 128 }), q = fp8("q.weight", { 64, 128 });
    const std::string dir = write_checkpoint(
        { p, bf16("p.weight_scale_inv", { 2, 2 }, 10.0f), q, bf16("q.weight_scale_inv", { 1, 2 }, 10.0f) },
        cfg);
    Checkpoint ck;
    REQUIRE_EQ(ck.open(dir), RAD_OK);
    CHECK_EQ(ck.fp8_block_rows(), (int64_t)64);
    Program prog;
    prog.name_map.push_back(map("pq", RAD_MAP_CONCAT, { "p.weight", "q.weight" }));
    CkptWeight w;
    std::string why;
    REQUIRE_EQ(ckpt_resolve(ck, prog, "pq", false, &w, &why), RAD_OK);
    CHECK(!w.trivial);
    CHECK(w.why.find("inside a scale block") != std::string::npos);
    REQUIRE_EQ(w.enc.n_planes, 2);
    CHECK_EQ(w.enc.plane[1].block[0], (int64_t)64);
    CHECK_EQ(w.enc.plane[1].block[1], (int64_t)64);
    /* p's row 90 sits in its second 64-row block (row 1 of the plane), which a 128 guess would
     * put in the first: its scale is element 2 of p's [2, 2] plane, 12, not 10. Column 100 is
     * in block column 1, element 1 of that block row, 13, not 11. */
    std::vector<float> f(128);
    REQUIRE_EQ(ckpt_rows_f32(w, 90, 1, f.data(), &why), RAD_OK);
    CHECK_EQ(f[0], ((float)((90 * 128) % 7) - 3.0f) * 12.0f);
    CHECK_EQ(f[100], ((float)((90 * 128 + 100) % 7) - 3.0f) * 13.0f);
    /* q's row 50: element 1 of its [1, 2] plane, 11. */
    REQUIRE_EQ(ckpt_rows_f32(w, 150, 1, f.data(), &why), RAD_OK);
    CHECK_EQ(f[100], ((float)((50 * 128 + 100) % 7) - 3.0f) * 11.0f);
    remove_checkpoint(dir);
}

/* A WEIGHT THAT MIXES A BLOCK-SCALED PIECE WITH A PER-CHANNEL ONE has no encoding that reads it:
 * one [br, bc] plane cannot serve both, so the read refuses with the reason. */
TEST(a_fp8_weight_mixed_block_and_per_channel_refuses_the_read) {
    T b = fp8("b.weight", { 128, 128 }), c = fp8("c.weight", { 32, 128 });
    const std::string dir = write_checkpoint(
        { b, bf16("b.weight_scale_inv", { 1, 1 }, 1.0f), c, bf16("c.weight_scale", { 32, 1 }, 1.0f) },
        kConfig);
    Checkpoint ck;
    REQUIRE_EQ(ck.open(dir), RAD_OK);
    Program prog;
    prog.name_map.push_back(map("bc", RAD_MAP_CONCAT, { "b.weight", "c.weight" }));
    CkptWeight w;
    std::string why;
    REQUIRE_EQ(ckpt_resolve(ck, prog, "bc", false, &w, &why), RAD_OK);
    CHECK(!w.trivial);
    CHECK(w.why.find("disagree") != std::string::npos);
    CHECK_EQ(w.enc.n_planes, (int)0);
    std::vector<float> f(128);
    CHECK(ckpt_rows_f32(w, 0, 1, f.data(), &why) < 0);
    remove_checkpoint(dir);
}
