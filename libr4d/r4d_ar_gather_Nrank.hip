// r4d_ar_gather_Nrank.hip: r4d_ar_gather_4rank, r4d_ar_gather_8rank -- one-shot P2P all-GATHER,
// 4 or 8 ranks, gfx1201. WS is compile-time; the 2-rank gather keeps its own file, exactly as the
// one-shot all-reduce's does.
//
// WHY IT EXISTS. r4d_ar_gather_2rank.hip is the engine's only device all-gather and its handshake
// is ONE PEER FLAG per block, not a tree (r4d_ar_entry.hip:492), so it is compiled for two ranks
// and refuses every other width. The vocab-parallel sampler declares an all_gather whenever the
// vocabulary is sharded, so at world 4 nothing but libref's host ref_all_gather resolves and the
// serve is refused by name. This is the wide one-shot all-reduce with the reduction REMOVED: same
// push, same per-block handshake, same double-buffered (parity, source) scratch, and the fp32 sum
// replaced by placement.
//
// THE PAYLOAD IS MOVED AND NEVER SUMMED. What the sampler gathers is (u32 global token id, f32
// logit) pairs; a token id's bit pattern read as f32 is a denormal, so adding two lanes would
// work on the logit and destroy the id. `dtype` is taken for symmetry with the family and used
// only to turn n_elem into bytes.
//
// THE PLACEMENT. row16 is one ROW of this rank's contribution in 16-byte words and r4d_g_dst_ws
// says where a rank's words land: the plain concatenation when there is one row, and otherwise
// ROW-INTERLEAVED -- input [rows][row16], output [rows][WS][row16] -- because the merge that
// consumes this reads one row of gathered candidates per sampled position at a single row stride.
// See r4d_ar_gather_2rank.hip, whose comment states why that is not a convenience.
//
// The IPC helpers and the handshake live in r4d_ar_wide.h; this unit links beside r4d_ar_entry.hip,
// which owns the instance and the peer arrays.
#include "r4d.h"
#include "r4d_ar_wide.h"
#include <cstdint>
#include <stdexcept>
#include <string>

// The launchers this unit provides, in the family's shape: the peer arguments are ascending global
// rank with the current rank removed, and `out` is world_size * n_elem elements. Declared extern
// "C" here, at arm's length from the definitions below, so those definitions get C linkage; the
// caller is r4d_ar_entry.hip, which owns the instance and repeats the two prototypes at its call
// site (r4d.h declares the rest of the family and is not this unit's file).
extern "C" void r4d_ar_gather_4rank(long peer_scratch0, long peer_scratch1, long peer_scratch2,
                                    long peer_flags0, long peer_flags1, long peer_flags2,
                                    long my_scratch, long my_flags, long seq_ctrs,
                                    long slot_stride16, long inp, long out, long n_elem,
                                    long dtype, long rank, long row_elems, long stream,
                                    long nblocks, long nthreads, long drain, long acq, long pub);
extern "C" void r4d_ar_gather_8rank(long peer_scratch0, long peer_scratch1, long peer_scratch2,
                                    long peer_scratch3, long peer_scratch4, long peer_scratch5,
                                    long peer_scratch6, long peer_flags0, long peer_flags1,
                                    long peer_flags2, long peer_flags3, long peer_flags4,
                                    long peer_flags5, long peer_flags6, long my_scratch,
                                    long my_flags, long seq_ctrs, long slot_stride16, long inp,
                                    long out, long n_elem, long dtype, long rank, long row_elems,
                                    long stream, long nblocks, long nthreads, long drain, long acq,
                                    long pub);

// WHERE A RANK'S WORDS LAND -- the 2-rank r4d_g_dst at a compile-time width. Named apart from it
// on purpose: same rule, different WS, and two __device__ definitions of one name would be an ODR
// violation between the two units.
__device__ __forceinline__ int r4d_g_dst_ws(int k, int row16, int n16, int WS, int who) {
  if (row16 <= 0 || row16 >= n16) return who * n16 + k;
  const int m = k / row16, c = k - m * row16;
  return m * (WS * row16) + who * row16 + c;
}

// Block b owns words [start,end) of the 16B-word space and drives its own handshake; no cross-block
// sync, exactly as in the all-reduce this rides beside.
template <int WS>
__global__ void r4d_ar_gather_wide_kernel(
    const int4* __restrict__ in4, R4DArWidePeers<int4, WS - 1> peers,
    const int4* __restrict__ my_base, int4* __restrict__ out4,
    unsigned int* __restrict__ my_flags, unsigned int* __restrict__ seq_ctrs,
    int n16, int slot_stride16, int rank, int row16, int drain, int acq, int pub) {
  constexpr int NPEERS = WS - 1;
  // The placement holds WS source words live per output word, so the 2x unroll spills at ws=8 --
  // the same ceiling the one-shot all-reduce draws for the same reason.
  constexpr int RU = (WS <= 4) ? 2 : 1;

  const int b = blockIdx.x, nb = gridDim.x, tid = threadIdx.x, nt = blockDim.x;
  __shared__ unsigned int s_seq, s_slot;
  r4d_ar_seq_take(seq_ctrs, R4D_AR_WIDE_MAX_BLOCKS, s_seq, s_slot);  // device-resident, replay-safe
  const unsigned int s = s_seq;
  const int parity = (int)s_slot;

  const int w = (n16 + nb - 1) / nb;
  const int start = b * w;
  int end = start + w; if (end > n16) end = n16;

  // Every peer lays its scratch out identically, so one index serves all of them: the slot that
  // NAMES THE SOURCE RANK. We write our own words into the same slot number of every peer's
  // scratch, and every peer reads them out of the slot numbered by OUR rank.
  int4* ps[NPEERS];
#pragma unroll
  for (int p = 0; p < NPEERS; ++p)
    ps[p] = peers.scratch[p] + ((size_t)parity * WS + rank) * (size_t)slot_stride16;

  // push: 4x-unrolled, and the loaded words are reused across peers so the input is read once.
  // This rank's own slice is NOT placed here -- the pass after the handshake places every rank's
  // slice from one src[] table, and duplicating the local store to hide it here would be the only
  // thing the all-reduce's shape does not carry over.
  {
    int k = start + tid;
    for (; k + 3 * nt < end; k += 4 * nt) {
      const int4 v0 = in4[k], v1 = in4[k + nt], v2 = in4[k + 2 * nt], v3 = in4[k + 3 * nt];
#pragma unroll
      for (int p = 0; p < NPEERS; ++p) {
        ps[p][k] = v0; ps[p][k + nt] = v1; ps[p][k + 2 * nt] = v2; ps[p][k + 3 * nt] = v3;
      }
    }
    for (; k < end; k += nt) {
      const int4 v = in4[k];
#pragma unroll
      for (int p = 0; p < NPEERS; ++p) ps[p][k] = v;
    }
  }

  // The wide family's own protocol: publish 2*seq to every peer's flag word [b][our rank], then
  // wait on our own [b][source] for every source. Flag values, parity and the slot guard are
  // r4d_ar_wide.h's, shared with the all-reduces this instance's buffers serve, and a handshake
  // that cannot complete traps rather than pass stale bytes on as a peer's.
  r4d_ar_handshake<WS>(peers, my_flags, b, rank, 2u * s, drain, acq, pub);

  // The peers' slices, now staged in this rank's scratch, into the places that name them. Slot
  // [parity][our rank] of my own scratch is written by nobody, so -- exactly as in the all-reduce
  // -- the source that names OUR rank is the input itself and every lane is branch-free.
  const int4* src[WS];
#pragma unroll
  for (int r = 0; r < WS; ++r)
    src[r] = (r == rank) ? in4
                         : (my_base + ((size_t)parity * WS + r) * (size_t)slot_stride16);
  {
    int k = start + tid;
    if (RU == 2) {
      for (; k + nt < end; k += 2 * nt) {
#pragma unroll
        for (int r = 0; r < WS; ++r) {
          out4[r4d_g_dst_ws(k,      row16, n16, WS, r)] = src[r][k];
          out4[r4d_g_dst_ws(k + nt, row16, n16, WS, r)] = src[r][k + nt];
        }
      }
    }
    for (; k < end; k += nt) {
#pragma unroll
      for (int r = 0; r < WS; ++r)
        out4[r4d_g_dst_ws(k, row16, n16, WS, r)] = src[r][k];
    }
  }
}

// --- host side ---
// `row_elems` is the row width in ELEMENTS, 0 for the plain concatenation; the entry point has
// already checked that it divides the message and is a whole number of 16-byte words, and it is
// rechecked here because this is the half that throws.
template <int WS>
static void ar_gather_wide_launch(const long* peer_scratch, const long* peer_flags, long my_scratch,
                                  long my_flags, long seq_ctrs, long slot_stride16, long inp,
                                  long out, long n_elem, long dtype, long rank, long row_elems,
                                  long stream_i, long nblocks, long nthreads, long drain, long acq,
                                  long pub) {
  constexpr int NPEERS = WS - 1;
  static_assert(NPEERS <= R4D_AR_WIDE_MAX_PEERS, "npeers exceeds the peer bundle bound");

  if (rank < 0 || rank >= WS)
    throw std::runtime_error("all_gather: rank " + std::to_string(rank) +
                             " out of range for the " + std::to_string(WS) +
                             " ranks this entry point is compiled for");

  const int esize = (dtype == 2) ? 4 : 2;
  const long nbytes = (long)n_elem * esize;
  if (nbytes % 16 != 0) throw std::runtime_error("all_gather: n_elem*esize not 16B-aligned");
  const long n16l = nbytes / 16;
  if (n16l > (long)slot_stride16)
    throw std::runtime_error("all_gather: message (" + std::to_string(n16l) +
                             " 16B words) exceeds the per-source slot stride (" +
                             std::to_string(slot_stride16) + ")");
  const int n16 = (int)n16l;

  long rowb = (long)row_elems * esize;
  if (row_elems > 0 && (rowb % 16 != 0 || nbytes % rowb != 0))
    throw std::runtime_error("all_gather: row is not a whole number of 16B words");
  const int row16 = row_elems > 0 ? (int)(rowb / 16) : 0;

  int nb = (int)nblocks;
  if (nb < 1) nb = R4D_AR_NB_DESIGN;
  if (nb > n16) nb = n16;
  if (nb > R4D_AR_WIDE_MAX_BLOCKS) nb = R4D_AR_WIDE_MAX_BLOCKS;
  int nt = (int)nthreads;
  if (nt < 64) nt = 64;
  if (nt > 1024) nt = 1024;

  R4DArWidePeers<int4, NPEERS> peers{};
  for (int i = 0; i < NPEERS; ++i) {
    if (!peer_scratch[i] || !peer_flags[i])
      throw std::runtime_error("all_gather: null peer scratch/flag pointer at index " +
                               std::to_string(i));
    peers.scratch[i] = (int4*)peer_scratch[i];
    peers.flags[i] = (unsigned int*)peer_flags[i];
    peers.rank[i] = (i < (int)rank) ? i : i + 1;
  }

  // Same buffers, same rule as the all-reduce this shares an instance with: scratch backs 2*WS
  // slots of the declared stride, flags one word per (block, source rank), seq a handshake and a
  // parity word per block of capacity -- checked against the allocations, not the arguments.
  const long slot_bytes = 2L * WS * (long)slot_stride16 * 16;
  const long flag_bytes = (long)nb * WS * (long)sizeof(unsigned int);
  r4d_ar_check_extent(my_scratch, slot_bytes, "all_gather", "my_scratch");
  r4d_ar_check_extent(my_flags, flag_bytes, "all_gather", "my_flags");
  r4d_ar_check_extent(seq_ctrs, 2L * R4D_AR_WIDE_MAX_BLOCKS * (long)sizeof(unsigned int),
                      "all_gather", "seq_ctrs");
  for (int i = 0; i < NPEERS; ++i) {
    r4d_ar_check_extent(peer_scratch[i], slot_bytes, "all_gather", "peer scratch");
    r4d_ar_check_extent(peer_flags[i], flag_bytes, "all_gather", "peer flags");
  }

  hipStream_t st = (hipStream_t)stream_i;
  r4d_ar_gather_wide_kernel<WS><<<nb, nt, 0, st>>>(
      (const int4*)inp, peers, (const int4*)my_scratch, (int4*)out, (unsigned int*)my_flags,
      (unsigned int*)seq_ctrs, n16, (int)slot_stride16, (int)rank, row16, (int)drain, (int)acq,
      (int)pub);
}

void r4d_ar_gather_4rank(long peer_scratch0, long peer_scratch1, long peer_scratch2,
                         long peer_flags0, long peer_flags1, long peer_flags2, long my_scratch,
                         long my_flags, long seq_ctrs, long slot_stride16, long inp, long out,
                         long n_elem, long dtype, long rank, long row_elems, long stream,
                         long nblocks, long nthreads, long drain, long acq, long pub) {
  const long ps[3] = {peer_scratch0, peer_scratch1, peer_scratch2};
  const long pf[3] = {peer_flags0, peer_flags1, peer_flags2};
  ar_gather_wide_launch<4>(ps, pf, my_scratch, my_flags, seq_ctrs, slot_stride16, inp, out, n_elem,
                           dtype, rank, row_elems, stream, nblocks, nthreads, drain, acq, pub);
}

void r4d_ar_gather_8rank(long peer_scratch0, long peer_scratch1, long peer_scratch2,
                         long peer_scratch3, long peer_scratch4, long peer_scratch5,
                         long peer_scratch6, long peer_flags0, long peer_flags1, long peer_flags2,
                         long peer_flags3, long peer_flags4, long peer_flags5, long peer_flags6,
                         long my_scratch, long my_flags, long seq_ctrs, long slot_stride16,
                         long inp, long out, long n_elem, long dtype, long rank, long row_elems,
                         long stream, long nblocks, long nthreads, long drain, long acq,
                         long pub) {
  const long ps[7] = {peer_scratch0, peer_scratch1, peer_scratch2, peer_scratch3,
                      peer_scratch4, peer_scratch5, peer_scratch6};
  const long pf[7] = {peer_flags0, peer_flags1, peer_flags2, peer_flags3,
                      peer_flags4, peer_flags5, peer_flags6};
  ar_gather_wide_launch<8>(ps, pf, my_scratch, my_flags, seq_ctrs, slot_stride16, inp, out, n_elem,
                           dtype, rank, row_elems, stream, nblocks, nthreads, drain, acq, pub);
}
