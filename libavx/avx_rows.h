/* avx_rows.h -- the quantised embedding gather's two halves: the row source, compiled once at the
 * baseline, and the row decode, compiled per ISA level.
 *
 * embed_lookup_q over an E4M3 table into bf16 is a gather of rows that are, most of the time, not in
 * memory at all: the n-gram table it exists for is tens of gigabytes in a container that is mapped
 * and never loaded. Getting the bytes is the whole cost and is the same work at every ISA level --
 * a row cache every thread shares, direct reads for its misses, a reader thread that runs a step
 * ahead -- so that half is avx_ngram.cpp, compiled baseline, with one row cache for the process
 * whatever level runs. Turning a row's codes into bf16 is the half that differs per level, so the
 * level's kernel hands its own decoder in. */
#ifndef RAD_AVX_ROWS_H
#define RAD_AVX_ROWS_H

#include "rad_abi.h"

#include <cstdint>

namespace avx {

/* dst[i][0, n) = lut[src[i][0, n)] for every i < count: E4M3 codes through a table of all 256
 * codes already scaled and rounded to bf16. */
typedef void (*RowDecode)(const uint8_t* const* src, uint16_t* const* dst, int64_t count,
                          int64_t n, const uint16_t* lut);

/* One call's gather, validated by the caller: `ids` holds M row ids, `tab` n_vocab rows of n_embd
 * elements -- E4M3 codes, or bf16 as stored -- `row_bytes` apart, `out` M rows of n_embd bf16
 * `x_ld` elements apart. The row cache and the reads deal in `row_bytes`; the decode in elements.
 * `ahead` holds n_ahead ids a later call will gather, or is null. rank and world_size are the
 * call's. `resident` is the op's parameter of that name: keep a copy of the table in RAM and read
 * that, once it is there (avx_ngram.cpp). */
struct RowGather {
    const int32_t* ids;
    int64_t        M;
    const uint8_t* tab;
    int64_t        n_vocab, n_embd, voff;
    int64_t        row_bytes;
    float          scale;
    uint16_t*      out;
    int64_t        x_ld;
    const int32_t* ahead;
    int64_t        n_ahead;
    int            rank, world_size;
    int            resident;
};

/* The gather, through the row cache when a file backs the table, from its copy in RAM once
 * `resident` has made one, and from the table's memory when no file backs it. Returns a RAD_*
 * status; on anything but RAD_OK the output is not complete. */
int row_gather(const RowGather& g, RowDecode decode);

}  /* namespace avx */

/* What the table must be: the E4M3 plane with ONE scale, or a plain bf16 one, stored as it is
 * (avx_ngram.cpp). */
extern "C" int avx_layout_ngram(const RadParam*, int, int, const RadEncoding*, const int*,
                                const RadTensor*, int, RadLayout*);

/* Whether this process can read the table's rows: refused, with the reason on stderr, where it may
 * not make an io_uring ring (avx_ngram.cpp). */
extern "C" int avx_init_ngram(const RadParam*, int, int, int, void**);

#endif /* RAD_AVX_ROWS_H */
