/* encoding_test.cpp -- weight encodings: the struct, its spelling, its geometry, the core's decoder
 * and the plane cuts tensor parallelism takes. spec.md §4.1.
 *
 * The decoder is checked against values worked by hand rather than against a second
 * implementation of itself: a scale that lands on the wrong block, a zero point subtracted after
 * the multiply instead of before, a nibble read from the wrong half of a byte -- each gives a
 * plausible number, and only an expected value that did not come from the code can tell.
 */
#include "rad_test.h"
#include "rad_types.h"
#include "rad_plugin.h"
#include "format/encoding.h"

#include <cstring>
#include <string>
#include <vector>

using namespace rad;

TEST(encodings_spell_themselves_canonically) {
    CHECK_EQ(enc_name(rad_enc_plain(RAD_BF16)), std::string("bf16"));

    RadEncoding fp8 = rad_enc_affine(RAD_F8E4M3, RAD_BF16, 128, 128);
    CHECK_EQ(enc_name(fp8), std::string("fp8_e4m3*bf16[128x128]"));

    RadEncoding w4 = rad_enc_affine(RAD_I4, RAD_BF16, 1, 128);
    rad_enc_copy_str(w4.transform, "fwht128");
    CHECK_EQ(enc_name(w4), std::string("i4*bf16[1x128]/fwht128"));

    RadEncoding w2 = rad_enc_affine(RAD_U2, RAD_F16, 1, 128);
    rad_enc_add_plane(&w2, "zero", RAD_U8, 1, 128);
    CHECK_EQ(enc_name(w2), std::string("u2*f16[1x128]-u8[1x128]"));

    RadEncoding tensor = rad_enc_affine(RAD_F8E4M3, RAD_BF16, 0, 0);
    CHECK_EQ(enc_name(tensor), std::string("fp8_e4m3*bf16[*x*]"));

    RadEncoding own;
    rad_enc_clear(&own);
    rad_enc_copy_str(own.scheme, "lattice");
    rad_enc_add_plane(&own, "codes", RAD_U8, 1, 1);
    rad_enc_add_plane(&own, "book", RAD_F16, 0, 0);
    CHECK_EQ(enc_name(own), std::string("lattice:codes=u8[1x1],book=f16[*x*]"));

    /* NVFP4: an e4m3 scale a 16 and an f32 a tensor, read in the order they multiply */
    RadEncoding nv = rad_enc_affine(RAD_FP4E2M1, RAD_F8E4M3, 1, 16);
    rad_enc_add_plane(&nv, "scale.1", RAD_F32, 0, 0);
    CHECK(rad_enc_valid(&nv));
    CHECK_EQ(enc_name(nv), std::string("fp4_e2m1*fp8_e4m3[1x16]*f32[*x*]"));

    /* Q4_K's min chain has no short form */
    RadEncoding qk = rad_enc_affine(RAD_U4, RAD_U6, 1, 32);
    rad_enc_add_plane(&qk, "scale.1", RAD_F16, 1, 256);
    rad_enc_add_plane(&qk, "min", RAD_U6, 1, 32);
    rad_enc_add_plane(&qk, "min.1", RAD_F16, 1, 256);
    CHECK(rad_enc_valid(&qk));
    CHECK_EQ(enc_name(qk), std::string("affine:codes=u4[1x1],scale=u6[1x32],scale.1=f16[1x256],"
                                       "min=u6[1x32],min.1=f16[1x256]"));

    /* a TABLE plane is written with its own extents */
    RadEncoding nl = rad_enc_affine(RAD_U4, RAD_F16, 1, 32);
    rad_enc_add_table(&nl, "table", RAD_I8, 1, 16);
    CHECK(rad_enc_valid(&nl));
    CHECK_EQ(enc_name(nl), std::string("affine:codes=u4[1x1],scale=f16[1x32],table=i8{1x16}"));
}

TEST(only_well_formed_encodings_are_valid) {
    RadEncoding e = rad_enc_plain(RAD_BF16);
    CHECK(rad_enc_valid(&e));
    e = rad_enc_affine(RAD_F8E4M3, RAD_BF16, 128, 128);
    CHECK(rad_enc_valid(&e));

    /* a rotation on a plain weight says nothing a reader could undo */
    e = rad_enc_plain(RAD_BF16);
    rad_enc_copy_str(e.transform, "fwht128");
    CHECK(!rad_enc_valid(&e));
    /* not a power of two */
    e = rad_enc_affine(RAD_I4, RAD_BF16, 1, 96);
    rad_enc_copy_str(e.transform, "fwht96");
    CHECK(!rad_enc_valid(&e));
    /* affine with nothing beyond the codes is plain, and a weight has one spelling */
    e = rad_enc_plain(RAD_I4);
    rad_enc_copy_str(e.scheme, "affine");
    CHECK(!rad_enc_valid(&e));
    /* a plugin-private dtype in a container would be unreadable without that plugin */
    e = rad_enc_plain(RAD_DT_PLUGIN_BASE + 3);
    CHECK(!rad_enc_valid(&e));
    /* the codes come first and cover one value each -- unless a grid expands them */
    e = rad_enc_affine(RAD_I4, RAD_BF16, 1, 128);
    e.plane[0].block[1] = 2;
    CHECK(!rad_enc_valid(&e));
    rad_enc_add_table(&e, "grid", RAD_I8, 16, 2);
    CHECK(rad_enc_valid(&e));
    e.plane[2].extent[1] = 4;                         /* a grid of 4-vectors for 2-wide codes */
    CHECK(!rad_enc_valid(&e));
    /* a scale.2 needs a scale.1 under it */
    e = rad_enc_affine(RAD_I4, RAD_BF16, 1, 32);
    rad_enc_add_plane(&e, "scale.2", RAD_F32, 0, 0);
    CHECK(!rad_enc_valid(&e));
    /* a role the scheme does not read is a mistake, not an extension */
    e = rad_enc_affine(RAD_I4, RAD_BF16, 1, 32);
    rad_enc_add_plane(&e, "scales", RAD_F32, 0, 0);
    CHECK(!rad_enc_valid(&e));
    /* a transform needs its tables */
    e = rad_enc_affine(RAD_I4, RAD_BF16, 1, 32);
    rad_enc_copy_str(e.transform, "givens");
    CHECK(!rad_enc_valid(&e));
    /* signs are one bit a value */
    e = rad_enc_affine(RAD_U4, RAD_BF16, 1, 32);
    rad_enc_add_plane(&e, "signs", RAD_U2, 1, 1);
    CHECK(!rad_enc_valid(&e));
}

TEST(plane_geometry_rounds_blocks_up_and_rows_to_a_byte) {
    const RadEncoding e = rad_enc_affine(RAD_I4, RAD_BF16, 1, 128);
    int64_t pr = 0, pc = 0;
    rad_enc_plane_dims(&e.plane[0], 10, 300, &pr, &pc);
    CHECK_EQ(pr, 10);
    CHECK_EQ(pc, 300);
    CHECK_EQ(rad_enc_plane_bytes(&e.plane[0], 10, 300), 10 * 150);
    rad_enc_plane_dims(&e.plane[1], 10, 300, &pr, &pc);
    CHECK_EQ(pr, 10);
    CHECK_EQ(pc, 3);                                  /* 128 + 128 + a short last block */
    CHECK_EQ(rad_enc_plane_bytes(&e.plane[1], 10, 300), 10 * 3 * 2);

    /* 2-bit codes, 7 a row: two bytes a row, not seven quarter-bytes run together */
    const RadEncoding u2 = rad_enc_plain(RAD_U2);
    CHECK_EQ(rad_enc_plane_bytes(&u2.plane[0], 3, 7), 3 * 2);

    const RadEncoding t = rad_enc_affine(RAD_F8E4M3, RAD_BF16, 0, 0);
    rad_enc_plane_dims(&t.plane[1], 320001536, 160, &pr, &pc);
    CHECK_EQ(pr, 1);
    CHECK_EQ(pc, 1);

    int64_t rows = 0, cols = 0;
    const int64_t shape3[3] = { 4, 5, 6 };
    rad_enc_view(3, shape3, &rows, &cols);
    CHECK_EQ(rows, 20);
    CHECK_EQ(cols, 6);
    const int64_t shape1[1] = { 9 };
    rad_enc_view(1, shape1, &rows, &cols);
    CHECK_EQ(rows, 1);
    CHECK_EQ(cols, 9);
}

TEST(sub_byte_codes_pack_low_bits_first) {
    uint8_t b[4] = { 0, 0, 0, 0 };
    CHECK(rad_store_code(b, RAD_I4, 0, -3));          /* 0xd */
    CHECK(rad_store_code(b, RAD_I4, 1, 7));           /* 0x7 */
    CHECK_EQ((int)b[0], 0x7d);
    CHECK_EQ(rad_load_f32(b, RAD_I4, 0), -3.0f);
    CHECK_EQ(rad_load_f32(b, RAD_I4, 1), 7.0f);
    CHECK_EQ(rad_load_f32(b, RAD_U4, 0), 13.0f);

    std::memset(b, 0, sizeof b);
    for (int i = 0; i < 4; ++i) CHECK(rad_store_code(b, RAD_U2, i, i));
    CHECK_EQ((int)b[0], 0xe4);                        /* 3 2 1 0 */
    for (int i = 0; i < 4; ++i) CHECK_EQ(rad_load_f32(b, RAD_U2, i), (float)i);
    CHECK_EQ(rad_load_f32(b, RAD_I2, 3), -1.0f);
    CHECK_EQ(rad_load_f32(b, RAD_I2, 2), -2.0f);

    /* fp4 e2m1: 0x7 is 6, 0x9 is -0.5 */
    b[1] = 0x97;
    CHECK_EQ(rad_load_f32(b + 1, RAD_FP4E2M1, 0), 6.0f);
    CHECK_EQ(rad_load_f32(b + 1, RAD_FP4E2M1, 1), -0.5f);

    b[2] = 127;
    CHECK_EQ(rad_load_f32(b + 2, RAD_E8M0, 0), 1.0f);
    b[2] = 130;
    CHECK_EQ(rad_load_f32(b + 2, RAD_E8M0, 0), 8.0f);
    b[2] = 0xff;
    CHECK(std::isnan(rad_load_f32(b + 2, RAD_E8M0, 0)));

    CHECK(!rad_store_code(b, RAD_F32, 0, 1));         /* wider than 16 bits is not a code */
}

TEST(odd_widths_are_a_bit_stream_that_straddles_bytes) {
    /* 3-bit codes 0..7 in a row: 24 bits, element 2 straddling bytes 0 and 1 */
    uint8_t b[8] = {};
    for (int i = 0; i < 8; ++i) CHECK(rad_store_code(b, RAD_U3, i, i));
    CHECK_EQ((int)b[0], 0x88);                        /* 0b10 001 000: elements 0, 1, low of 2 */
    for (int i = 0; i < 8; ++i) CHECK_EQ(rad_load_f32(b, RAD_U3, i), (float)i);
    CHECK_EQ(rad_load_f32(b, RAD_I3, 7), -1.0f);
    CHECK_EQ(rad_load_f32(b, RAD_I3, 4), -4.0f);

    /* every width, a pattern that cannot pass by accident */
    const uint32_t widths[] = { RAD_U1, RAD_U2, RAD_U3, RAD_U4, RAD_U5, RAD_U6, RAD_U7,
                                RAD_U9, RAD_U10, RAD_U11, RAD_U12 };
    for (uint32_t dt : widths) {
        const int bits = rad_dtype_bits(dt);
        std::vector<uint8_t> row((size_t)(37 * bits + 7) / 8 + 2, 0xa5);
        for (int i = 0; i < 37; ++i)
            rad_store_code(row.data(), dt, i, (int)((i * 2654435761u) & ((1u << bits) - 1)));
        int off = 0;
        for (int i = 0; i < 37; ++i)
            off += rad_load_f32(row.data(), dt, i) !=
                   (float)((i * 2654435761u) & ((1u << bits) - 1));
        CHECK_EQ(off, 0);
    }
    /* signed widths: the extremes */
    for (uint32_t dt : { RAD_I3, RAD_I5, RAD_I6, RAD_I7 }) {
        const int bits = rad_dtype_bits(dt), lo = -(1 << (bits - 1)), hi = (1 << (bits - 1)) - 1;
        uint8_t r[4] = {};
        rad_store_code(r, dt, 0, lo);
        rad_store_code(r, dt, 1, hi);
        CHECK_EQ(rad_load_f32(r, dt, 0), (float)lo);
        CHECK_EQ(rad_load_f32(r, dt, 1), (float)hi);
    }
}

TEST(fp6_decodes_both_forms) {
    uint8_t b[4] = {};
    rad_store_code(b, RAD_FP6E2M3, 0, 0x1f);          /* e=3 m=7: 1.875 * 4 */
    rad_store_code(b, RAD_FP6E2M3, 1, 0x21);          /* negative subnormal: -1/8 */
    CHECK_EQ(rad_load_f32(b, RAD_FP6E2M3, 0), 7.5f);
    CHECK_EQ(rad_load_f32(b, RAD_FP6E2M3, 1), -0.125f);
    std::memset(b, 0, sizeof b);
    rad_store_code(b, RAD_FP6E3M2, 0, 0x1f);          /* e=7 m=3: 1.75 * 16 */
    rad_store_code(b, RAD_FP6E3M2, 1, 0x0c);          /* e=3 m=0: 1 */
    CHECK_EQ(rad_load_f32(b, RAD_FP6E3M2, 0), 28.0f);
    CHECK_EQ(rad_load_f32(b, RAD_FP6E3M2, 1), 1.0f);
}

TEST(affine_decodes_through_the_right_block_and_subtracts_the_zero_first) {
    /* A 2 x 4 weight, u4 codes, a scale per [1 x 2] and a zero per [2 x 4]. */
    RadEncoding e = rad_enc_affine(RAD_U4, RAD_F32, 1, 2);
    rad_enc_add_plane(&e, "zero", RAD_U8, 2, 4);
    CHECK(rad_enc_valid(&e));
    uint8_t codes[4] = { 0, 0, 0, 0 };
    const int q[8] = { 1, 2, 3, 4, 5, 6, 7, 8 };
    for (int r = 0; r < 2; ++r)
        for (int c = 0; c < 4; ++c) rad_store_code(codes + r * 2, RAD_U4, c, q[r * 4 + c]);
    const float scale[4] = { 1.0f, 10.0f, 100.0f, 1000.0f };   /* [2][2] */
    const uint8_t zero[1] = { 3 };
    const void* planes[3] = { codes, scale, zero };
    EncodedView v;
    enc_view(e, 2, 4, planes, &v);
    std::vector<float> out(8);
    CHECK_EQ(enc_decode_rows(v, 0, 2, out.data()), RAD_OK);
    const float want[8] = { -2, -1, 0, 10, 200, 300, 4000, 5000 };
    for (int i = 0; i < 8; ++i) CHECK_EQ(out[(size_t)i], want[i]);

    /* one row alone decodes the same */
    std::vector<float> one(4);
    CHECK_EQ(enc_decode_rows(v, 1, 1, one.data()), RAD_OK);
    for (int i = 0; i < 4; ++i) CHECK_EQ(one[(size_t)i], want[4 + i]);
}

TEST(a_rotation_decodes_back_to_the_weight) {
    /* Store v = H w / 4 for a 4-wide group; decoding applies H once more, H H = 4 I, so the
     * weight comes back exactly -- the convention docs/MOE-W4.md folds into the stored scale. */
    const float w[8] = { 1, -2, 3, 0.5f, 0, 0, 2, -1 };
    float v[8];
    std::memcpy(v, w, sizeof v);
    fwht_inplace(v, 4);
    fwht_inplace(v + 4, 4);
    for (float& x : v) x /= 4.0f;
    RadEncoding e = rad_enc_plain(RAD_F32);
    rad_enc_copy_str(e.scheme, "affine");
    rad_enc_add_plane(&e, "scale", RAD_F32, 0, 0);
    rad_enc_copy_str(e.transform, "fwht4");
    CHECK(rad_enc_valid(&e));
    const float one = 1.0f;
    const void* planes[2] = { v, &one };
    EncodedView ev;
    enc_view(e, 1, 8, planes, &ev);
    float out[8];
    CHECK_EQ(enc_decode_rows(ev, 0, 1, out), RAD_OK);
    for (int i = 0; i < 8; ++i) CHECK_EQ(out[i], w[i]);

    /* the Sylvester order: H_4 e_1 is the second column, (1, -1, 1, -1) */
    float e1[4] = { 0, 1, 0, 0 };
    fwht_inplace(e1, 4);
    CHECK_EQ(e1[0], 1.0f);
    CHECK_EQ(e1[1], -1.0f);
    CHECK_EQ(e1[2], 1.0f);
    CHECK_EQ(e1[3], -1.0f);

    /* a group that does not divide the row is refused, not decoded short */
    enc_view(e, 1, 6, planes, &ev);
    std::string why;
    CHECK_EQ(enc_decode_rows(ev, 0, 1, out, &why), RAD_E_SHAPE);
}

TEST(multilevel_scales_and_minimums_decode_as_the_k_quants_do) {
    /* One row of 8: u4 codes, u6 scale and min per [1 x 4], f16 super-scale and super-min per row.
     * w = d * sc * q - dmin * m, Q4_K's formula. */
    RadEncoding e = rad_enc_affine(RAD_U4, RAD_U6, 1, 4);
    rad_enc_add_plane(&e, "scale.1", RAD_F16, 1, 0);
    rad_enc_add_plane(&e, "min", RAD_U6, 1, 4);
    rad_enc_add_plane(&e, "min.1", RAD_F16, 1, 0);
    REQUIRE(rad_enc_valid(&e));
    uint8_t q[4] = {}, sc[2] = {}, mn[2] = {};
    for (int i = 0; i < 8; ++i) rad_store_code(q, RAD_U4, i, i + 1);
    rad_store_code(sc, RAD_U6, 0, 3);
    rad_store_code(sc, RAD_U6, 1, 5);
    rad_store_code(mn, RAD_U6, 0, 2);
    rad_store_code(mn, RAD_U6, 1, 7);
    const uint16_t d = rad_f32_to_f16(0.5f), dm = rad_f32_to_f16(0.25f);
    const void* planes[5] = { q, sc, &d, mn, &dm };
    EncodedView v;
    enc_view(e, 1, 8, planes, &v);
    float out[8];
    REQUIRE_EQ(enc_decode_rows(v, 0, 1, out), RAD_OK);
    for (int i = 0; i < 8; ++i) {
        const float want = 0.5f * (i < 4 ? 3 : 5) * (float)(i + 1) - 0.25f * (i < 4 ? 2 : 7);
        CHECK_EQ(out[i], want);
    }
}

TEST(nvfp4_multiplies_both_scale_levels) {
    RadEncoding e = rad_enc_affine(RAD_FP4E2M1, RAD_F8E4M3, 1, 16);
    rad_enc_add_plane(&e, "scale.1", RAD_F32, 0, 0);
    uint8_t q[16] = {};
    for (int i = 0; i < 32; ++i) rad_store_code(q, RAD_FP4E2M1, i, i & 15);
    const uint8_t s[2] = { rad_f32_to_fp8e4m3(2.0f), rad_f32_to_fp8e4m3(0.5f) };
    const float t = 3.0f;
    const void* planes[3] = { q, s, &t };
    EncodedView v;
    enc_view(e, 1, 32, planes, &v);
    float out[32];
    REQUIRE_EQ(enc_decode_rows(v, 0, 1, out), RAD_OK);
    for (int i = 0; i < 32; ++i)
        CHECK_EQ(out[i], rad_fp4e2m1_to_f32((uint8_t)(i & 15)) * (i < 16 ? 2.0f : 0.5f) * 3.0f);
}

TEST(a_scalar_codebook_maps_each_code_before_the_scale) {
    /* IQ4_NL: the code indexes sixteen non-linear levels */
    static const int8_t kvalues[16] = { -127, -104, -83, -65, -49, -35, -22, -10,
                                        1, 13, 25, 38, 53, 69, 89, 113 };
    RadEncoding e = rad_enc_affine(RAD_U4, RAD_F16, 1, 4);
    rad_enc_add_table(&e, "table", RAD_I8, 1, 16);
    REQUIRE(rad_enc_valid(&e));
    uint8_t q[2] = {};
    const int codes[4] = { 0, 7, 8, 15 };
    for (int i = 0; i < 4; ++i) rad_store_code(q, RAD_U4, i, codes[i]);
    const uint16_t d = rad_f32_to_f16(0.25f);
    const void* planes[3] = { q, &d, kvalues };
    EncodedView v;
    enc_view(e, 1, 4, planes, &v);
    float out[4];
    REQUIRE_EQ(enc_decode_rows(v, 0, 1, out), RAD_OK);
    for (int i = 0; i < 4; ++i) CHECK_EQ(out[i], 0.25f * kvalues[codes[i]]);
}

TEST(a_vector_codebook_expands_a_code_into_its_grid_row_and_signs) {
    /* IQ2-shaped: a u8 code per 4 values into a grid of 4-vectors, a sign bit a value, a u4
     * scale per [1 x 8] and an f16 per row. */
    static const uint8_t grid[3][4] = { { 8, 8, 8, 8 }, { 8, 25, 43, 8 }, { 43, 43, 25, 8 } };
    RadEncoding e = rad_enc_affine(RAD_U8, RAD_U4, 1, 8);
    e.plane[0].block[1] = 4;
    rad_enc_add_plane(&e, "scale.1", RAD_F16, 1, 0);
    rad_enc_add_table(&e, "grid", RAD_U8, 3, 4);
    rad_enc_add_plane(&e, "signs", RAD_U1, 1, 1);
    REQUIRE(rad_enc_valid(&e));
    const uint8_t codes[2] = { 1, 2 };
    uint8_t sc[1] = {};
    rad_store_code(sc, RAD_U4, 0, 3);
    const uint16_t d = rad_f32_to_f16(0.125f);
    const uint8_t signs[1] = { 0x81 };                /* values 0 and 7 negative */
    const void* planes[5] = { codes, sc, &d, grid, signs };
    EncodedView v;
    enc_view(e, 1, 8, planes, &v);
    float out[8];
    REQUIRE_EQ(enc_decode_rows(v, 0, 1, out), RAD_OK);
    for (int i = 0; i < 8; ++i) {
        float g = (float)grid[codes[i / 4]][i % 4];
        if (i == 0 || i == 7) g = -g;
        CHECK_EQ(out[i], g * 3.0f * 0.125f);
    }
    /* a code past the grid's end is a corrupt file, said so */
    const uint8_t bad[2] = { 1, 9 };
    const void* bp[5] = { bad, sc, &d, grid, signs };
    enc_view(e, 1, 8, bp, &v);
    std::string why;
    CHECK_EQ(enc_decode_rows(v, 0, 1, out, &why), RAD_E_FORMAT);
}

TEST(an_act_order_permutation_puts_columns_back) {
    RadEncoding e = rad_enc_plain(RAD_F32);
    rad_enc_copy_str(e.scheme, "affine");
    rad_enc_add_plane(&e, "scale", RAD_F32, 0, 0);
    rad_enc_add_table(&e, "t.perm", RAD_I32, 1, 4);
    rad_enc_copy_str(e.transform, "perm");
    REQUIRE(rad_enc_valid(&e));
    const float w[4] = { 10, 20, 30, 40 };
    const int32_t perm[4] = { 2, 0, 3, 1 };
    float vq[4];
    for (int j = 0; j < 4; ++j) vq[j] = w[perm[j]];   /* stored in the order it was quantised */
    const float one = 1.0f;
    const void* planes[3] = { vq, &one, perm };
    EncodedView v;
    enc_view(e, 1, 4, planes, &v);
    float out[4];
    REQUIRE_EQ(enc_decode_rows(v, 0, 1, out), RAD_OK);
    for (int j = 0; j < 4; ++j) CHECK_EQ(out[j], w[j]);
}

TEST(a_givens_transform_with_channel_scales_inverts) {
    /* Two stages over 4 columns and a channel scale: build v = w . D . G1 . G2 the way a
     * quantiser would, store v exactly, and decode back to w. */
    const int K = 4;
    const float w[4] = { 1.0f, -2.0f, 0.5f, 3.0f };
    const float dsc[4] = { 2.0f, 0.5f, 1.0f, 4.0f };
    const int32_t pairs[2][4] = { { 0, 1, 2, 3 }, { 0, 2, 1, 3 } };
    const float ang[2][2] = { { 0.3f, -1.1f }, { 0.7f, 2.0f } };
    double x[4];
    for (int k = 0; k < K; ++k) x[k] = (double)w[k] * dsc[k];
    for (int s = 0; s < 2; ++s)
        for (int p = 0; p < 2; ++p) {
            const int a = pairs[s][2 * p], b = pairs[s][2 * p + 1];
            const double c = std::cos((double)ang[s][p]), sn = std::sin((double)ang[s][p]);
            const double xa = x[a], xb = x[b];
            x[a] = xa * c + xb * sn;
            x[b] = xb * c - xa * sn;
        }
    float vq[4];
    for (int k = 0; k < K; ++k) vq[k] = (float)x[k];

    RadEncoding e = rad_enc_plain(RAD_F32);
    rad_enc_copy_str(e.scheme, "affine");
    rad_enc_add_plane(&e, "scale", RAD_F32, 0, 0);
    rad_enc_add_table(&e, "t.pairs", RAD_I32, 2, 4);
    rad_enc_add_table(&e, "t.angle", RAD_F32, 2, 2);
    rad_enc_add_table(&e, "t.scale", RAD_F32, 1, 4);
    rad_enc_copy_str(e.transform, "givens");
    REQUIRE(rad_enc_valid(&e));
    const float one = 1.0f;
    const void* planes[5] = { vq, &one, pairs, ang, dsc };
    EncodedView v;
    enc_view(e, 1, 4, planes, &v);
    float out[4];
    REQUIRE_EQ(enc_decode_rows(v, 0, 1, out), RAD_OK);
    for (int k = 0; k < K; ++k) CHECK(std::fabs(out[k] - w[k]) < 1e-5f);
}

TEST(givens_forward_turns_a_base_activation_into_the_stored_code_domain) {
    /* The identity path A leans on: with the same tables, a = enc_givens_forward(x) satisfies
     * dot(a, v) == dot(x, w) for every stored code row v, w being what givens_inverse decodes from
     * v. A GEMM that reads v then reproduces exactly the base weight's output. */
    const int K = 4;
    const float v[4] = { 1.0f, -2.0f, 0.5f, 3.0f };
    const float dsc[4] = { 2.0f, 0.5f, 1.0f, 4.0f };          /* t.scale = 1/channel_scales */
    const int32_t pairs[2][4] = { { 0, 1, 2, 3 }, { 0, 2, 1, 3 } };
    const float ang[2][2] = { { 0.3f, -1.1f }, { 0.7f, 2.0f } };
    RadEncoding e = rad_enc_plain(RAD_F32);
    rad_enc_copy_str(e.scheme, "affine");
    rad_enc_add_plane(&e, "scale", RAD_F32, 0, 0);
    rad_enc_add_table(&e, "t.pairs", RAD_I32, 2, 4);
    rad_enc_add_table(&e, "t.angle", RAD_F32, 2, 2);
    rad_enc_add_table(&e, "t.scale", RAD_F32, 1, 4);
    rad_enc_copy_str(e.transform, "givens");
    const float one = 1.0f;
    const void* planes[5] = { v, &one, pairs, ang, dsc };
    EncodedView ev;
    enc_view(e, 1, 4, planes, &ev);
    float w[4];
    REQUIRE_EQ(enc_decode_rows(ev, 0, 1, w), RAD_OK);
    const float x[4] = { 0.7f, -1.3f, 2.1f, 0.4f };
    float a[4];
    for (int k = 0; k < K; ++k) a[k] = x[k];
    REQUIRE_EQ(enc_givens_forward(ev, a), RAD_OK);
    double dot_av = 0, dot_xw = 0;
    for (int k = 0; k < K; ++k) {
        dot_av += (double)a[k] * v[k];
        dot_xw += (double)x[k] * w[k];
    }
    CHECK(std::fabs(dot_av - dot_xw) < 1e-5);
}

TEST(the_core_declines_a_scheme_it_did_not_define) {
    RadEncoding e;
    rad_enc_clear(&e);
    rad_enc_copy_str(e.scheme, "lattice");
    rad_enc_add_plane(&e, "codes", RAD_U8, 1, 1);
    uint8_t c = 0;
    const void* planes[1] = { &c };
    EncodedView v;
    enc_view(e, 1, 1, planes, &v);
    float out = 0;
    std::string why;
    CHECK_EQ(enc_decode_rows(v, 0, 1, &out, &why), RAD_E_UNSUPPORTED);
    CHECK(why.find("lattice") != std::string::npos);
}

TEST(a_plane_is_cut_on_its_own_blocks) {
    const RadEncoding e = rad_enc_affine(RAD_F8E4M3, RAD_BF16, 128, 128);
    PlaneRect r;
    std::string why;
    /* a row shard of a [5120, 17408] weight at tp 2 */
    CHECK_EQ(enc_plane_rect(e.plane[0], 5120, 17408, 2560, 5120, 0, 17408, &r, &why), RAD_OK);
    CHECK_EQ(r.row0, 2560);
    CHECK_EQ(r.rows, 2560);
    CHECK_EQ(enc_plane_rect(e.plane[1], 5120, 17408, 2560, 5120, 0, 17408, &r, &why), RAD_OK);
    CHECK_EQ(r.row0, 20);
    CHECK_EQ(r.rows, 20);
    CHECK_EQ(r.cols, 136);
    /* a 640-wide weight split 3:2 in 128-column blocks */
    CHECK_EQ(enc_plane_rect(e.plane[1], 2560, 640, 0, 2560, 384, 640, &r, &why), RAD_OK);
    CHECK_EQ(r.col0, 3);
    CHECK_EQ(r.cols, 2);
    /* half of 640 splits a block */
    CHECK_EQ(enc_plane_rect(e.plane[1], 2560, 640, 0, 2560, 320, 640, &r, &why), RAD_E_SHAPE);
    CHECK(why.find("splits") != std::string::npos);
    /* the weight's own end is not a split: 300 columns end inside a block */
    CHECK_EQ(enc_plane_rect(e.plane[1], 4, 300, 0, 4, 256, 300, &r, &why), RAD_OK);
    CHECK_EQ(r.col0, 2);
    CHECK_EQ(r.cols, 1);

    /* a whole-extent scale belongs to every slice */
    const RadEncoding t = rad_enc_affine(RAD_F8E4M3, RAD_BF16, 0, 0);
    CHECK_EQ(enc_plane_rect(t.plane[1], 1000, 160, 500, 1000, 0, 160, &r, &why), RAD_OK);
    CHECK_EQ(r.row0, 0);
    CHECK_EQ(r.rows, 1);

    /* a table goes whole into every slice */
    RadEncoding nl = rad_enc_affine(RAD_U4, RAD_F16, 1, 32);
    rad_enc_add_table(&nl, "table", RAD_I8, 1, 16);
    CHECK_EQ(enc_plane_rect(nl.plane[2], 64, 256, 32, 64, 128, 256, &r, &why), RAD_OK);
    CHECK_EQ(r.rows, 1);
    CHECK_EQ(r.cols, 16);

    /* a 4-bit column cut must land on a byte */
    const RadEncoding w4 = rad_enc_affine(RAD_I4, RAD_BF16, 1, 1);
    CHECK_EQ(enc_plane_rect(w4.plane[0], 2, 10, 0, 2, 3, 10, &r, &why), RAD_E_SHAPE);
    CHECK_EQ(enc_plane_rect(w4.plane[0], 2, 10, 0, 2, 4, 10, &r, &why), RAD_OK);
}

RAD_TEST_MAIN()
