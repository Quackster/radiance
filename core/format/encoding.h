/* encoding.h -- the core's half of a weight encoding: decoding one to f32, and cutting a plane for
 * tensor parallelism. The encoding itself is abi/rad_encoding.h; spec.md §4.1.
 *
 * The core decodes the two schemes it defines, "plain" and "affine", under every transform the
 * ABI names, and nothing else: a quantiser that invents a scheme decodes it (abi/rad_quant.h). The
 * decoder is what the oracle checks a kernel through, what rad-convert measures a quantiser's
 * error with, and what a weight is read back through to be quantised again -- three consumers
 * that must agree on what a stored weight means, which is why there is one of it.
 */
#pragma once
#include "../rad_internal.h"
#include "rad_encoding.h"
#include "rad_quant.h"

#include <cstdint>
#include <string>
#include <vector>

namespace rad {

/* One whole plane of a weight, canonical: row-major, `row_bytes` a row. */
struct PlaneView {
    const void* data = nullptr;
    int64_t     rows = 0, cols = 0;   /* elements */
    int64_t     row_bytes = 0;
};

/* The planes of an encoding over a [rows, cols] weight, in the encoding's order. */
struct EncodedView {
    RadEncoding enc{};
    int64_t     rows = 0, cols = 0;   /* the logical view */
    PlaneView   plane[RAD_ENC_MAX_PLANES];
};

/* Fill `v`'s plane geometry for a weight of [rows, cols] in `e`, pointing plane i at `data[i]`. */
void enc_view(const RadEncoding& e, int64_t rows, int64_t cols, const void* const* data,
              EncodedView* v);

/* Decode logical rows [row0, row0 + n) into `out`, f32 [n, cols]. Returns RAD_OK, or
 * RAD_E_UNSUPPORTED for a scheme the core does not decode (with `why` saying which), RAD_E_SHAPE
 * when a transform does not fit the row, or RAD_E_FORMAT for a malformed encoding or a code its
 * codebook has no entry for -- a corrupt file, said so, rather than a read past a table. */
int enc_decode_rows(const EncodedView& v, int64_t row0, int64_t n, float* out,
                    std::string* why = nullptr);

/* THE DECODERS FOR SCHEMES THE CORE DOES NOT DEFINE: quantisers' own (RadQuantDecodeFn). The
 * plugin loader adds each loaded quantiser's and removes them when it unloads; enc_decode_rows
 * asks them in that order for any scheme but plain and affine. */
void enc_add_decoder(RadQuantDecodeFn fn);
void enc_remove_decoder(RadQuantDecodeFn fn);

/* The unnormalised Sylvester Walsh-Hadamard transform of `n` values in place, `n` a power of two.
 * The butterfly keeps (a + b) in the lower index and (a - b) in the upper, which is H_n in
 * natural (Sylvester) order -- the transform libr4d's r4d_fwht_wave computes in registers, and the
 * one an "fwhtN" encoding names. H_n is symmetric and H_n H_n = n I. */
void fwht_inplace(float* v, int64_t n);

/* The canonical spelling of an encoding (rad_enc_format), as a string. */
std::string enc_name(const RadEncoding& e);

/* ================================================================== ParoQuant's rotation tables
 *
 * A `paroquant_mxfp6` CHECKPOINT'S ROTATION, AS THE ENCODING'S. The published checkpoint describes
 * its transform as w = R(v) . D: `channel_scales` is the diagonal D applied to the RIGHT of the
 * rotation, and `pairs` are indices LOCAL to a `group`-wide rotation group (0..group-1), one group
 * after another along the row. The encoding's convention is the other way round -- v = w . D . G1
 * ... GS, decoded by givens_inverse as w = v . GS^T ... G1^T . D^-1 (abi/rad_encoding.h) -- so
 * this globalizes the pairs (column = group base + local index) and stores the RECIPROCAL scale,
 * 1/channel_scales. givens_inverse's divide then multiplies by D and reconstructs exactly R(v). D.
 *
 * `pairs` is I16 [stages * cols]; `theta` F16 [stages * cols/2] in radians; `channel_scales` F16
 * [cols]. The three outputs receive the tables' canonical bytes: i32 global pairs [stages, cols],
 * f32 angles [stages, cols/2], f32 1/D [1, cols]. RAD_OK, or RAD_E_SHAPE when the geometry is not
 * a whole number of rotation groups / RAD_E_FORMAT when a local pair is outside its group or a
 * channel scale is zero. */
int enc_givens_paro_tables(int64_t stages, int64_t cols, int64_t group, const int16_t* pairs,
                           const uint16_t* theta, const uint16_t* channel_scales,
                           std::vector<uint8_t>* pairs_i32, std::vector<uint8_t>* angle_f32,
                           std::vector<uint8_t>* scale_f32, std::string* why);

/* THE FORWARD READING OF THOSE TABLES: the activation-side rotation path A serves with. Given the
 * encoding's `t.pairs`/`t.angle`/`t.scale` (scale = 1/channel_scales) and an input row x, computes
 * a = (x / t.scale) . M_dec^T in place -- givens_inverse's stages in ascending order, transposed --
 * so that for any stored code row v, dot(a, v) == dot(x, w) with w the base weight givens_inverse
 * decodes from v. A givens-fused producer runs this on the bf16 activation before its fp8 quantiser.
 * RAD_OK, or RAD_E_SHAPE when the tables do not fit the row. */
int enc_givens_forward(const EncodedView& v, float* row, std::string* why = nullptr);

/* ================================================================== a selection, as a reader reads it
 *
 * The `n_sel` planes of `e` a declaration selects (RadWeightDecl::planes) -- plane `sel[k]` at
 * `data[k]`, dense [rows[k], cols[k]] in its own elements -- as `n` elements of dtype `dt` at
 * `out`: the tensor a reference implementation reads where a kernel reads the stored form. One
 * plane is converted element by element (a bf16 scale plane into an f32 reference, codes as their
 * values); codes with the scalar codebook they index, each code as its table entry; a scale with
 * the levels of its chain above it, each scale as their product; the whole encoding is decoded.
 * RAD_OK, or RAD_E_SHAPE with `why` when the planes do not make that tensor -- part of an
 * encoding, which decodes to nothing on its own, or extents that disagree with `n`; or the
 * decoder's own refusal. */
int enc_selection_logical(const RadEncoding& e, const int* sel, int n_sel, const int64_t* rows,
                          const int64_t* cols, const void* const* data, uint32_t dt, int64_t n,
                          void* out, std::string* why);

/* ================================================================== slicing */
/* A rectangle of a plane, in that plane's elements. */
struct PlaneRect {
    int64_t row0 = 0, rows = 0;
    int64_t col0 = 0, cols = 0;
};

/* The part of plane `p` that covers logical rows [r0, r1) and columns [c0, c1) of a [rows, cols]
 * weight. Refused, with `why`, when a bound would split one of the plane's blocks -- a scale that
 * covers rows two ranks own between them belongs to neither, and taking it twice gives each a
 * plane that decodes to the right numbers for the wrong rows. The weight's own end is never a
 * split: a last block may be short. */
int enc_plane_rect(const RadEncPlane& p, int64_t rows, int64_t cols, int64_t r0, int64_t r1,
                   int64_t c0, int64_t c1, PlaneRect* out, std::string* why);

}  /* namespace rad */
