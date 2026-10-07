/* item.cpp -- the parts of an Item the step path uses, in a unit of their own. The batch builder
 * links against this and nothing else in core/mm, so a test that fakes the device layer does not
 * pull the processor -- and through its metadata reader, the real device layer -- in with it. */
#include "mm/processor.h"

namespace rad {
namespace mm {

void Item::coords(int64_t first, int64_t count, int32_t* planes) const {
    /* Merge-block order inverted: a patch's index within its segment is
     * ((block_row * blocks_w + block_col) * merge + in_row) * merge + in_col. */
    const int32_t merge = merge_size();
    int segment = 0;
    for (int64_t k = 0; k < count; ++k) {
        while (segment + 1 < n_seg && first + k >= patch_at(segment + 1)) ++segment;
        const int64_t within = first + k - patch_at(segment);
        const int32_t gh = segment_h(segment), gw = segment_w(segment);
        const int64_t bw = gw / merge;
        const int64_t in_col = within % merge;
        const int64_t in_row = (within / merge) % merge;
        const int64_t blk = within / (merge * merge);
        const int64_t row = (blk / bw) * merge + in_row;
        const int64_t col = (blk % bw) * merge + in_col;
        planes[k]             = (int32_t)row;
        planes[count + k]     = (int32_t)col;
        planes[2 * count + k] = gh;
        planes[3 * count + k] = gw;
    }
}

}  /* namespace mm */
}  /* namespace rad */
