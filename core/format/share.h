/* share.h -- which bytes of a container entry one rank's declaration takes.
 *
 * A container entry is a logical weight's planes, whole and canonical (rad_format.h). A rank
 * declares a VIEW of it (RadWeightDecl::source/planes) and, under tensor parallelism, only its
 * share: a ROW shard is rows of every selected plane and a COL shard is columns, each cut on the
 * plane's own block boundary and refused where a block would be split (spec §4.3). This is that
 * arithmetic, and nothing else -- the loader turns its rectangles into reads.
 *
 * Pure, so a test can hold it to the shapes tensor parallelism actually produces: a stacked
 * [gate|up] whose every part is split, an uneven span, a block-fp8 scale plane cut at 1/128 of
 * its weight's rows, a per-tensor scale every rank carries whole.
 */
#pragma once
#include "../rad_core.h"

#include <string>
#include <vector>

namespace rad {

/* One plane's share, as the rectangles of the plane that make it, in the order they lie end to
 * end at the destination. A rectangle is `rows` rows from `row0`, and of each row the `bytes`
 * bytes from `byte0`; a row of the plane is `row_bytes` long in the file. */
struct PlaneShare {
    struct Rect { int64_t row0 = 0, rows = 0, byte0 = 0, bytes = 0; };
    std::vector<Rect> rects;
    uint32_t dtype = RAD_DT_INVALID;
    int64_t rows = 0, cols = 0;      /* the share's extents, in the plane's elements */
    int64_t row_bytes = 0;
    /* rad_weight_shard_zero: the share has its extents and no rectangles; the loader writes
     * dense_bytes() of zeros and reads nothing. */
    bool    zero = false;
    int64_t dense_bytes() const;     /* the share laid out dense: rows * a row of `cols` */
};

/* This rank's share of every plane `w` selects, from an entry of encoding `enc` and logical
 * shape (`rank`, `shape`), at tensor-parallel rank `index` of `world`. RAD_OK, or RAD_E_SHAPE with
 * the reason in `why`: an extent that does not divide, a cut that splits a block or a byte, a
 * share that is not what the declaration says. */
int weight_share(const WeightInfo& w, const RadEncoding& enc, uint32_t rank, const int64_t* shape,
                 int world, int index, PlaneShare out[RAD_ENC_MAX_PLANES], std::string* why);

}  /* namespace rad */
