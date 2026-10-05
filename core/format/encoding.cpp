/* encoding.cpp -- decoding a weight encoding to f32 and cutting its planes. See encoding.h, and
 * abi/rad_encoding.h for what each plane means. */
#include "encoding.h"
#include "rad_plugin.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <vector>

namespace rad {

void enc_view(const RadEncoding& e, int64_t rows, int64_t cols, const void* const* data,
              EncodedView* v) {
    v->enc  = e;
    v->rows = rows;
    v->cols = cols;
    for (int i = 0; i < RAD_ENC_MAX_PLANES; ++i) v->plane[i] = PlaneView{};
    for (int i = 0; i < e.n_planes && i < RAD_ENC_MAX_PLANES; ++i) {
        PlaneView& p = v->plane[i];
        rad_enc_plane_dims(&e.plane[i], rows, cols, &p.rows, &p.cols);
        p.row_bytes = rad_enc_row_bytes(e.plane[i].dtype, p.cols);
        p.data = data ? data[i] : nullptr;
    }
}

void fwht_inplace(float* v, int64_t n) {
    for (int64_t h = 1; h < n; h <<= 1)
        for (int64_t i = 0; i < n; i += h << 1)
            for (int64_t j = i; j < i + h; ++j) {
                const float a = v[j], b = v[j + h];
                v[j] = a + b;
                v[j + h] = a - b;
            }
}

std::string enc_name(const RadEncoding& e) {
    char buf[512];
    rad_enc_format(&e, buf, sizeof buf);
    return buf;
}

namespace {

std::string fmt(const char* f, long long a = 0, long long b = 0, long long c = 0) {
    char m[256];
    std::snprintf(m, sizeof m, f, a, b, c);
    return m;
}

/* A plane and the means to read element (i, j) of it as f32. */
struct Rd {
    const PlaneView* v = nullptr;
    const RadEncPlane* d = nullptr;
    explicit operator bool() const { return v != nullptr; }
    float at(int64_t i, int64_t j) const {
        return rad_load_f32((const uint8_t*)v->data + i * v->row_bytes, d->dtype, j);
    }
    /* The element of a TILED plane covering logical (r, c). */
    float cover(int64_t r, int64_t c) const {
        return at(d->block[0] > 0 ? r / d->block[0] : 0, d->block[1] > 0 ? c / d->block[1] : 0);
    }
};

/* Row `r` of a code plane as f32, its dtype resolved once for the row rather than per element.
 * The values are rad_load_f32's: the same nibble order, the same sign extension. */
void codes_row(const Rd& p, int64_t r, int64_t n, float* out) {
    const uint8_t* b = (const uint8_t*)p.v->data + r * p.v->row_bytes;
    switch (p.d->dtype) {
        case RAD_I4:
            for (int64_t i = 0; i < n; ++i) {
                const int c = (b[i >> 1] >> ((i & 1) * 4)) & 15;
                out[i] = (float)((c ^ 8) - 8);
            }
            return;
        case RAD_U4:
            for (int64_t i = 0; i < n; ++i) out[i] = (float)((b[i >> 1] >> ((i & 1) * 4)) & 15);
            return;
        case RAD_I8:
            for (int64_t i = 0; i < n; ++i) out[i] = (float)(int8_t)b[i];
            return;
        case RAD_U8:
            for (int64_t i = 0; i < n; ++i) out[i] = (float)b[i];
            return;
        case RAD_F8E4M3:
            for (int64_t i = 0; i < n; ++i) out[i] = rad_fp8e4m3_to_f32(b[i]);
            return;
        default:
            for (int64_t i = 0; i < n; ++i) out[i] = rad_load_f32(b, p.d->dtype, i);
            return;
    }
}

Rd plane(const EncodedView& v, int idx) {
    Rd r;
    if (idx >= 0) {
        r.v = &v.plane[idx];
        r.d = &v.enc.plane[idx];
    }
    return r;
}

/* Undo a "givens" transform on one row in place: w = v . G_S^T ... G_1^T . D^-1. */
int givens_inverse(const EncodedView& v, float* row, std::string* why) {
    const Rd pairs = plane(v, rad_enc_find(&v.enc, "t.pairs"));
    const Rd angle = plane(v, rad_enc_find(&v.enc, "t.angle"));
    const Rd scale = plane(v, rad_enc_find(&v.enc, "t.scale"));
    const int64_t S = pairs.v->rows, K = v.cols;
    if (pairs.v->cols != K || angle.v->rows != S || angle.v->cols != K / 2 || (K & 1) ||
        (scale && (scale.v->rows != 1 || scale.v->cols != K))) {
        if (why) *why = fmt("a givens transform's tables do not fit a %lld-column row", K);
        return RAD_E_SHAPE;
    }
    for (int64_t s = S - 1; s >= 0; --s)
        for (int64_t p = 0; p < K / 2; ++p) {
            const int64_t a = (int64_t)pairs.at(s, 2 * p), b = (int64_t)pairs.at(s, 2 * p + 1);
            if (a < 0 || a >= K || b < 0 || b >= K || a == b) {
                if (why) *why = fmt("givens stage %lld pairs columns %lld and %lld", s, a, b);
                return RAD_E_SHAPE;
            }
            const double t = angle.at(s, p), c = std::cos(t), sn = std::sin(t);
            const double x = row[a], y = row[b];
            row[a] = (float)(c * x - sn * y);
            row[b] = (float)(sn * x + c * y);
        }
    if (scale)
        for (int64_t k = 0; k < K; ++k) row[k] /= scale.at(0, k);
    return RAD_OK;
}

}  /* namespace */

namespace {
std::mutex                     g_dec_mu;
std::vector<RadQuantDecodeFn>  g_decoders;
}  // namespace

void enc_add_decoder(RadQuantDecodeFn fn) {
    if (!fn) return;
    std::lock_guard<std::mutex> lk(g_dec_mu);
    g_decoders.push_back(fn);
}

void enc_remove_decoder(RadQuantDecodeFn fn) {
    std::lock_guard<std::mutex> lk(g_dec_mu);
    g_decoders.erase(std::remove(g_decoders.begin(), g_decoders.end(), fn), g_decoders.end());
}

int enc_decode_rows(const EncodedView& v, int64_t row0, int64_t n, float* out, std::string* why) {
    const RadEncoding& e = v.enc;
    const bool plain  = rad_enc_is(&e, "plain");
    const bool affine = rad_enc_is(&e, "affine");
    if (!plain && !affine) {
        /* A QUANTISER'S OWN SCHEME, read by the quantiser that wrote it -- whichever loaded one
         * answers for it (abi/rad_quant.h). The weight is described as far as the caller knows it:
         * a [rows, cols] view and the planes from their starts. */
        std::vector<RadQuantDecodeFn> fns;
        {
            std::lock_guard<std::mutex> lk(g_dec_mu);
            fns = g_decoders;
        }
        RadQuantWeight w{};
        w.name = "";
        w.rank = 2;
        w.shape[0] = v.rows;
        w.shape[1] = v.cols;
        w.rows = v.rows;
        w.cols = v.cols;
        w.layer = w.expert = -1;
        const void* planes[RAD_ENC_MAX_PLANES] = {};
        for (int k = 0; k < e.n_planes && k < RAD_ENC_MAX_PLANES; ++k) planes[k] = v.plane[k].data;
        for (RadQuantDecodeFn fn : fns) {
            const int st = fn(&e, &w, planes, row0, n, out);
            if (st == RAD_E_UNSUPPORTED) continue;
            if (st != RAD_OK && why)
                *why = "the quantiser that decodes scheme '" + std::string(e.scheme) +
                       "' refused: " + rad_strerror(st);
            return st;
        }
        if (why) *why = "scheme '" + std::string(e.scheme) + "' is a quantiser's own, and no "
                        "loaded quantiser decodes it; the core decodes only plain and affine";
        return RAD_E_UNSUPPORTED;
    }
    if (!rad_enc_valid(&e)) {
        if (why) *why = "the encoding " + enc_name(e) + " is not well formed";
        return RAD_E_FORMAT;
    }
    const int64_t fw = rad_enc_fwht(&e);
    if (fw > 0 && v.cols % fw != 0) {
        if (why) *why = fmt("an fwht%lld transform does not divide a %lld-column row", fw, v.cols);
        return RAD_E_SHAPE;
    }
    if (row0 < 0 || n < 0 || row0 + n > v.rows) return RAD_E_SHAPE;

    const Rd codes = plane(v, 0);
    const Rd grid  = plane(v, rad_enc_find(&e, "grid"));
    const Rd table = plane(v, rad_enc_find(&e, "table"));
    const Rd signs = plane(v, rad_enc_find(&e, "signs"));
    const Rd zero  = plane(v, rad_enc_find(&e, "zero"));
    std::vector<Rd> scales, mins;
    for (int lv = 0;; ++lv) {
        const int i = rad_enc_find_level(&e, "scale", lv);
        if (i < 0) break;
        scales.push_back(plane(v, i));
    }
    for (int lv = 0;; ++lv) {
        const int i = rad_enc_find_level(&e, "min", lv);
        if (i < 0) break;
        mins.push_back(plane(v, i));
    }
    const int64_t V = rad_enc_code_width(&e);
    const bool perm = !std::strcmp(e.transform, "perm");
    const bool giv  = !std::strcmp(e.transform, "givens");
    const Rd pm = plane(v, perm ? rad_enc_find(&e, "t.perm") : -1);
    if (perm && (pm.v->rows != 1 || pm.v->cols != v.cols)) {
        if (why) *why = fmt("a perm transform's table is not one entry a column (%lld)", v.cols);
        return RAD_E_SHAPE;
    }
    std::vector<float> tmp(perm ? (size_t)v.cols : 0);
    /* THE COMMON FORM, A ROW AT A TIME: codes with no codebook, sign, zero or minimum and at most
     * one scale level -- int4 experts, fp8 rows, a fixed-scale table. The element walk below
     * resolves every element's dtype and its scale's index through a switch and two divisions,
     * and on a Flash-Next conversion that walk is a third of the convert; here the codes' dtype is
     * resolved once a row and the scale read once a group. The same products, so the same
     * floats. */
    const bool direct = V <= 1 && !table && !signs && !zero && mins.empty() && scales.size() <= 1;

    for (int64_t i = 0; i < n; ++i) {
        const int64_t r = row0 + i;
        float* o = out + i * v.cols;
        if (direct) {
            codes_row(codes, r, v.cols, o);
            if (!scales.empty()) {
                const Rd& s = scales[0];
                const int64_t bc = s.d->block[1] > 0 ? s.d->block[1] : v.cols;
                const int64_t sr = s.d->block[0] > 0 ? r / s.d->block[0] : 0;
                for (int64_t c0 = 0; c0 < v.cols; c0 += bc) {
                    const float k = s.at(sr, s.d->block[1] > 0 ? c0 / bc : 0);
                    const int64_t c1 = std::min(c0 + bc, v.cols);
                    for (int64_t c = c0; c < c1; ++c) o[c] *= k;
                }
            }
        }
        for (int64_t c = 0; !direct && c < v.cols; ++c) {
            float q;
            if (V > 1) {
                const int64_t code = (int64_t)codes.at(r, c / V);
                if (code < 0 || code >= grid.v->rows) {
                    if (why) *why = fmt("row %lld: code %lld is outside a %lld-entry grid", r, code,
                                        grid.v->rows);
                    return RAD_E_FORMAT;
                }
                q = grid.at(code, c % V);
            } else {
                q = codes.at(r, c);
                if (table) {
                    const int64_t code = (int64_t)q;
                    if (code < 0 || code >= table.v->cols) {
                        if (why) *why = fmt("row %lld: code %lld is outside a %lld-entry table", r,
                                            code, table.v->cols);
                        return RAD_E_FORMAT;
                    }
                    q = table.at(0, code);
                }
            }
            if (signs && signs.cover(r, c) != 0.0f) q = -q;
            if (zero) q -= zero.cover(r, c);
            for (const Rd& s : scales) q *= s.cover(r, c);
            if (!mins.empty()) {
                float m = 1.0f;
                for (const Rd& s : mins) m *= s.cover(r, c);
                q -= m;
            }
            o[c] = q;
        }
        if (fw > 0) {
            for (int64_t g = 0; g < v.cols; g += fw) fwht_inplace(o + g, fw);
        } else if (perm) {
            for (int64_t c = 0; c < v.cols; ++c) {
                const int64_t to = (int64_t)pm.at(0, c);
                if (to < 0 || to >= v.cols) {
                    if (why) *why = fmt("perm sends column %lld to %lld", c, to);
                    return RAD_E_FORMAT;
                }
                tmp[(size_t)to] = o[c];
            }
            std::memcpy(o, tmp.data(), (size_t)v.cols * sizeof(float));
        } else if (giv) {
            const int st = givens_inverse(v, o, why);
            if (st != RAD_OK) return st;
        }
    }
    return RAD_OK;
}

int enc_selection_logical(const RadEncoding& e, const int* sel, int n_sel, const int64_t* rows,
                          const int64_t* cols, const void* const* data, uint32_t dt, int64_t n,
                          void* out, std::string* why) {
    auto refuse = [&](std::string m) {
        if (why) *why = std::move(m);
        return RAD_E_SHAPE;
    };
    if (n_sel < 1 || n_sel > e.n_planes) return refuse("no planes are selected");
    if (n_sel == 1) {
        const RadEncPlane& p = e.plane[sel[0]];
        const int64_t r = rows[0], c = cols[0];
        if (r * c != n)
            return refuse("its plane holds " + std::to_string(r * c) + " elements and the reader "
                          "wants " + std::to_string(n));
        const int64_t rb = rad_enc_row_bytes(p.dtype, c);
        const uint8_t* src = (const uint8_t*)data[0];
        if (p.dtype == dt && rb == rad_dtype_bytes(dt, c)) {
            std::memcpy(out, src, (size_t)(r * rb));
            return RAD_OK;
        }
        for (int64_t i = 0; i < r; ++i)
            for (int64_t j = 0; j < c; ++j)
                rad_store_f32(out, dt, i * c + j, rad_load_f32(src + i * rb, p.dtype, j));
        return RAD_OK;
    }
    /* CODES AND THE SCALAR CODEBOOK THEY INDEX, without the scale: each code as its table entry.
     * It is what a kernel that folds the scale itself reads -- the codes view of a w4nla8h expert
     * -- and the entries are the values the reference multiplies, where the raw indices would be
     * other numbers. */
    if (n_sel == 2 && !std::strcmp(e.plane[sel[0]].role, "codes") &&
        !std::strcmp(e.plane[sel[1]].role, "table") && e.plane[sel[0]].kind == RAD_PLANE_TILED &&
        e.plane[sel[0]].block[0] == 1 && e.plane[sel[0]].block[1] == 1 &&
        e.plane[sel[1]].kind == RAD_PLANE_TABLE && e.plane[sel[1]].extent[0] == 1) {
        const RadEncPlane& pc = e.plane[sel[0]];
        const RadEncPlane& pt = e.plane[sel[1]];
        const int64_t r = rows[0], c = cols[0], entries = pt.extent[1];
        if (r * c != n)
            return refuse("its codes hold " + std::to_string(r * c) + " elements and the reader "
                          "wants " + std::to_string(n));
        const int64_t rb = rad_enc_row_bytes(pc.dtype, c);
        const uint8_t* src = (const uint8_t*)data[0];
        for (int64_t i = 0; i < r; ++i)
            for (int64_t j = 0; j < c; ++j) {
                const int64_t code = (int64_t)rad_load_f32(src + i * rb, pc.dtype, j);
                if (code < 0 || code >= entries)
                    return refuse("code " + std::to_string(code) + " indexes a table of " +
                                  std::to_string(entries));
                rad_store_f32(out, dt, i * c + j, rad_load_f32(data[1], pt.dtype, code));
            }
        return RAD_OK;
    }
    /* A SCALE AND THE LEVELS ABOVE IT, without the codes: each first-level scale times every level
     * that covers it. It is what a kernel that takes its scales as an operand of their own reads
     * where the encoding states them in two levels -- the scale view of a w4nl64a8h expert, an
     * E4M3 scale a (row, 64 of K) under one fixed f32 -- and the product is the scale the
     * reference multiplies, where the first level on its own is that scale off by the second. One
     * element a first-level scale, in that plane's extents. A level has to cover whole
     * first-level blocks -- the whole weight, or a block a multiple of the first's -- or a scale
     * would straddle two of its elements and have no one product. */
    if (n_sel >= 2 && !std::strcmp(e.plane[sel[0]].role, "scale")) {
        bool chain = true;
        for (int k = 0; k < n_sel && chain; ++k) {
            const RadEncPlane& p = e.plane[sel[k]];
            chain = p.kind == RAD_PLANE_TILED && rad_enc_role_in_chain(p.role, "scale") &&
                    (k == 0 || std::strcmp(p.role, "scale") != 0);
        }
        if (chain) {
            const RadEncPlane& p0 = e.plane[sel[0]];
            const int64_t r = rows[0], c = cols[0];
            if (r * c != n)
                return refuse("its scales hold " + std::to_string(r * c) + " elements and the "
                              "reader wants " + std::to_string(n));
            auto covers = [](int64_t b0, int64_t b) { return b <= 0 || (b0 > 0 && b % b0 == 0); };
            for (int k = 1; k < n_sel; ++k) {
                const RadEncPlane& p = e.plane[sel[k]];
                if (!covers(p0.block[0], p.block[0]) || !covers(p0.block[1], p.block[1]))
                    return refuse(std::string("its '") + p.role + "' level does not cover whole '" +
                                  p0.role + "' blocks");
            }
            const int64_t rb0 = rad_enc_row_bytes(p0.dtype, c);
            for (int64_t i = 0; i < r; ++i)
                for (int64_t j = 0; j < c; ++j) {
                    float v = rad_load_f32((const uint8_t*)data[0] + i * rb0, p0.dtype, j);
                    for (int k = 1; k < n_sel; ++k) {
                        const RadEncPlane& p = e.plane[sel[k]];
                        const int64_t li = p.block[0] > 0 ? i * p0.block[0] / p.block[0] : 0;
                        const int64_t lj = p.block[1] > 0 ? j * p0.block[1] / p.block[1] : 0;
                        if (li >= rows[k] || lj >= cols[k])
                            return refuse(std::string("its '") + p.role + "' level is " +
                                          std::to_string(rows[k]) + "x" +
                                          std::to_string(cols[k]) + " and does not reach scale " +
                                          std::to_string(i) + "," + std::to_string(j));
                        v *= rad_load_f32((const uint8_t*)data[k] +
                                              li * rad_enc_row_bytes(p.dtype, cols[k]),
                                          p.dtype, lj);
                    }
                    rad_store_f32(out, dt, i * c + j, v);
                }
            return RAD_OK;
        }
    }
    if (n_sel != e.n_planes)
        return refuse("it takes " + std::to_string(n_sel) + " of its encoding's " +
                      std::to_string(e.n_planes) + " planes, which decode to nothing on their own");
    /* The logical rows from the first selected plane and its block height; the columns are what
     * is left of `n`, since a codes plane of vector codes does not carry them. */
    const RadEncPlane& p0 = e.plane[sel[0]];
    const int64_t r = rows[0] * (p0.kind == RAD_PLANE_TILED && p0.block[0] > 0 ? p0.block[0] : 1);
    if (r <= 0 || n % r)
        return refuse("its planes do not tile the " + std::to_string(n) +
                      " elements the reader wants");
    const int64_t c = n / r;
    const void* d[RAD_ENC_MAX_PLANES] = {};
    for (int k = 0; k < n_sel; ++k) d[sel[k]] = data[k];
    EncodedView v;
    enc_view(e, r, c, d, &v);
    std::vector<float> f((size_t)n);
    const int rc = enc_decode_rows(v, 0, r, f.data(), why);
    if (rc != RAD_OK) return rc;
    for (int64_t i = 0; i < n; ++i) rad_store_f32(out, dt, i, f[(size_t)i]);
    return RAD_OK;
}

int enc_plane_rect(const RadEncPlane& p, int64_t rows, int64_t cols, int64_t r0, int64_t r1,
                   int64_t c0, int64_t c1, PlaneRect* out, std::string* why) {
    int64_t pr = 0, pc = 0;
    rad_enc_plane_dims(&p, rows, cols, &pr, &pc);
    if (p.kind == RAD_PLANE_TABLE) {          /* every slice carries the whole table */
        *out = PlaneRect{ 0, pr, 0, pc };
        return RAD_OK;
    }
    auto cut = [&](int64_t lo, int64_t hi, int64_t extent, int64_t b, int64_t pext,
                   int64_t* e0, int64_t* en, const char* axis) -> bool {
        if (lo < 0 || hi > extent || lo > hi) {
            if (why) *why = std::string(axis) + fmt(" [%lld, %lld) is outside the weight's %lld",
                                                    lo, hi, extent);
            return false;
        }
        if (b <= 0) {                 /* one element covers the whole extent: every slice has it */
            *e0 = 0;
            *en = pext;
            return true;
        }
        if (lo % b != 0 || (hi % b != 0 && hi != extent)) {
            if (why) *why = std::string(axis) + fmt(" [%lld, %lld) splits ", lo, hi) + "the '" +
                            p.role + fmt("' plane's %lld-wide blocks", b);
            return false;
        }
        *e0 = lo / b;
        *en = (hi + b - 1) / b - lo / b;
        return true;
    };
    PlaneRect r;
    if (!cut(r0, r1, rows, p.block[0], pr, &r.row0, &r.rows, "rows")) return RAD_E_SHAPE;
    if (!cut(c0, c1, cols, p.block[1], pc, &r.col0, &r.cols, "columns")) return RAD_E_SHAPE;
    /* A slice is copied as bytes, so a packed plane's cut has to land on a byte. */
    const int bits = rad_dtype_bits(p.dtype);
    if (bits > 0 && (r.col0 * bits) % 8 != 0) {
        if (why) *why = fmt("column %lld of the ", r.col0) + rad_dtype_name(p.dtype) + " '" +
                        p.role + "' plane is not on a byte";
        return RAD_E_SHAPE;
    }
    *out = r;
    return RAD_OK;
}

}  /* namespace rad */
