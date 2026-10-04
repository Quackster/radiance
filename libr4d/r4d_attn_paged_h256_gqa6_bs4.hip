// R4D attention at SIX queries per KV head, head_dim 256 -- the same QSA layers of
// Qwen3.8-Flash-Next that r4d_attn_paged_h256_gqa12.hip serves at two ranks, served at FOUR: the
// two KV heads are replicated across the rank pairs, so each rank answers 6 queries for the one KV
// head it holds and the ratio the constraint table gates on is 6 rather than 12.
//
// Four lines of geometry and the dispatch itself, because r4d_attn_paged_gqa8.hip's kernels are
// templates on `gqa` and that unit is the parameterised body. What the ratio changes is
// exactly two bounds, and both are in the ROW table rather than here:
//
//   the decode verify width   64 / gqa = 10, against 5 at gqa 12 and 8 at gqa 8. A workgroup holds
//                             64 (query, head) rows and carries q_len * gqa of them, so this is
//                             the WIDEST of the three legs: MTP's K=3 verify is q_len 4 and fits
//                             either ratio, and a draft block of six or seven positions -- which
//                             the gqa12 leg has to send to the prefill kernel -- fits the decode
//                             kernel here.
//   a prefill block's width   384 / gqa = 64 query positions, against 32 at gqa 12 and 48 at
//                             gqa 8.
//
// NO head_dim 128 LEG, for the gqa12 file's reason: nothing in scope is gqa 6 at 128, and an
// instantiation nothing resolves is a kernel nobody has run. R4D_ATTN_H128 leaves those four entry
// points out of this translation unit entirely rather than shipping them untested.
#define R4D_ATTN_GQA  6
#define R4D_ATTN_H128 0
// THE KV PAGE IS FOUR TOKENS HERE, AND IT IS THE ONLY GEOMETRY IN THIS LIBRARY WHERE IT IS NOT 16.
//
// Qwen3.8-Flash-Next's QSA indexer selects 4-TOKEN BLOCKS of the key history. With a page of 4 a
// selected block IS a page, so the selection is a paged block table and the sparse attention is
// this kernel with a different table -- instead of a new attention kernel, which is the most
// expensive kind of kernel in this tree to get right. docs/QSA.md is the decomposition. The page
// is 4 at this ratio for exactly the reason it is 4 at gqa 12 -- the model's page, not the
// parallelism's; what TP4 changes is the queries a head, not the storage.
//
// IT COSTS NOTHING. A page size is pure storage, so the only thing it can move is the block-table
// walk -- `NB = TILE / BS` entries a tile instead of one, four instead of one at TILE 16 -- and at
// a realistic context that is inside the noise of the step, with identical output either way.
//
// THE ROWS HAVE TO AGREE. `C_EQ("block_size", 4)` on all four gqa6-bs4 rows is what makes the core
// size the pages at 4 (RadKVGroupDecl carries no block size on purpose -- it is taken off the
// resolved kernel). A row that says 16 against this define is a kernel that refuses at issue
// rather than at declare.
#define G8_BLOCK_SIZE 4

#include "r4d_attn_paged_gqa8.hip"
