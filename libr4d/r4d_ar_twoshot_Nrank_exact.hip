// r4d_ar_twoshot_Nrank_exact.hip: r4d_ar_twoshot_{3,4,8}rank_exact -- reduce-scatter +
// all-gather at 3, 4 or 8 ranks, exact bf16/fp16/fp32 wire. Per-block pipelined handshake
// instead of a grid arrival election. Sends 2*(ws-1)/ws*N bytes per rank in two hops, so it
// covers the sizes where the one-shot's (ws-1)*N send is too heavy.
//
// Shard s = a contiguous run of the message's 16-byte words, owned by rank s. Phase 1: push my
// contribution to each peer-owned shard into the owner's slot[me] region A, flag 2s-1; reduce my
// shard fp32 in ascending rank order (own contribution from the input). Phase 2: reduced words go
// to my out AND every peer's slot[me] region B, flag 2s; assemble out from region B. Region A at
// 0, B at sh_max; the slot stride must hold 2*sh_max.
//
// UNEVEN SHARDS. A message of n16 words splits into base = n16/ws words a shard, and the first
// n16 % ws shards take one more: shard s starts at s*base + min(s, rem). At 4 and 8 ranks every
// residual the engine issues divides evenly and this is the plain 1/ws split. At 3 it almost never
// does -- a 2048-wide bf16 row is 256 words, and 256 * rows is a multiple of 3 only when rows is --
// so the split has to tolerate it rather than refuse it. The sum per word is the same fp32 fold in
// ascending rank order whichever rank owns it, so the bits match the one-shot's.
//
// SHARES the wide family's flag/seq/scratch set: every wide kernel publishes in the 2*seq flag
// space (this one adds the 2s-1 half-step) and advances the per-block counter once per AR, so
// parity and the monotonic spin survive any mixed one-shot/two-shot stream on one buffer set.
// The slot stride must hold the largest requirement of every family in the mix.
#include "r4d.h"
#include "r4d_ar_wide.h"
#include <stdexcept>
#include <string>

template <typename T, int LANES, int WS>
__global__ void r4d_ar_twoshot_wide_exact_kernel(
    const int4* __restrict__ in4, R4DArWidePeers<int4, WS - 1> peers,
    const int4* __restrict__ my_base, unsigned int* __restrict__ my_flags,
    int4* __restrict__ out4, unsigned int* __restrict__ seq_ctrs, int n16,
    int slot_stride16, int rank, int drain, int acq, int pub) {
  constexpr int NPEERS = WS - 1;
  const int b = blockIdx.x, nb = gridDim.x, tid = threadIdx.x, nt = blockDim.x;
  __shared__ unsigned int s_seq, s_slot;
  r4d_ar_seq_take(seq_ctrs, R4D_AR_WIDE_MAX_BLOCKS, s_seq, s_slot);  // device-resident, replay-safe
  const unsigned int s = s_seq;
  const int parity = (int)s_slot;

  // shard s = words [s*base + min(s, rem), +base + (s < rem)); sh_max is the longest one
  const int base = n16 / WS, rem = n16 % WS;
  const int sh_max = base + (rem ? 1 : 0);
#define R4D_TS_OFF(s) ((s) * base + ((s) < rem ? (s) : rem))
#define R4D_TS_LEN(s) (base + ((s) < rem ? 1 : 0))

  const int w = (sh_max + nb - 1) / nb;    // block b owns words [w0,w1) of the SHARD space
  const int w0 = b * w;
  int w1 = w0 + w; if (w1 > sh_max) w1 = sh_max;

  // region A of my slot in every peer's scratch (B sits at +sh_max)
  int4* pa[NPEERS];
#pragma unroll
  for (int p = 0; p < NPEERS; ++p)
    pa[p] = peers.scratch[p] + ((size_t)parity * WS + rank) * (size_t)slot_stride16;

  // ---- phase 1 push: my contribution to each peer-owned shard -> owner's region A
  for (int p = 0; p < NPEERS; ++p) {
    const int owner = peers.rank[p];
    const int4* src = in4 + (size_t)R4D_TS_OFF(owner);
    int4* dst = pa[p];
    const int lim = w1 < R4D_TS_LEN(owner) ? w1 : R4D_TS_LEN(owner);
    for (int k = w0 + tid; k < lim; k += nt) dst[k] = src[k];
  }
  r4d_ar_handshake<WS>(peers, my_flags, b, rank, 2u * s - 1u, drain, acq, pub);

  // ---- reduce my shard (fp32, ascending rank, own contribution from the input), and phase 2
  // push: the reduced word goes to my out and every peer's region B in the same pass
  const int4* src[WS];
#pragma unroll
  for (int r = 0; r < WS; ++r)
    src[r] = (r == rank) ? (in4 + (size_t)R4D_TS_OFF(rank))
                         : (my_base + ((size_t)parity * WS + r) * (size_t)slot_stride16);
  int4* my_out = out4 + (size_t)R4D_TS_OFF(rank);
  const int my_lim = w1 < R4D_TS_LEN(rank) ? w1 : R4D_TS_LEN(rank);
  for (int k = w0 + tid; k < my_lim; k += nt) {
    float acc[LANES];
#pragma unroll
    for (int j = 0; j < LANES; ++j) acc[j] = 0.0f;
#pragma unroll
    for (int r = 0; r < WS; ++r) {
      const int4 v = src[r][k];
      const T* pv = reinterpret_cast<const T*>(&v);
#pragma unroll
      for (int j = 0; j < LANES; ++j) acc[j] += r4d_ar_to_f(pv[j]);
    }
    int4 o;
    T* po = reinterpret_cast<T*>(&o);
#pragma unroll
    for (int j = 0; j < LANES; ++j) r4d_ar_from_f(acc[j], po[j]);
    my_out[k] = o;
#pragma unroll
    for (int p = 0; p < NPEERS; ++p) pa[p][sh_max + k] = o;  // region B
  }
  r4d_ar_handshake<WS>(peers, my_flags, b, rank, 2u * s, drain, acq, pub);

  // ---- assemble: every peer-owned shard from my region B copies
  for (int p = 0; p < NPEERS; ++p) {
    const int owner = peers.rank[p];
    const int4* srcB = my_base + ((size_t)parity * WS + owner) * (size_t)slot_stride16 + sh_max;
    int4* dst = out4 + (size_t)R4D_TS_OFF(owner);
    const int lim = w1 < R4D_TS_LEN(owner) ? w1 : R4D_TS_LEN(owner);
    for (int k = w0 + tid; k < lim; k += nt) dst[k] = srcB[k];
  }
#undef R4D_TS_OFF
#undef R4D_TS_LEN
}

// --- host side ---
template <int WS>
static void ar_twoshot_launch(const long* peer_scratch, const long* peer_flags, long my_scratch,
                              long my_flags, long seq_ctrs, long slot_stride16, long inp,
                              long out, long n_elem, long dtype, long rank, long stream_i,
                              long nblocks, long nthreads, long drain, long acq, long pub) {
  constexpr int NPEERS = WS - 1;
  if (rank < 0 || rank >= WS)
    throw std::runtime_error("all_reduce: rank " + std::to_string(rank) +
                             " out of range for the " + std::to_string(WS) +
                             " ranks this entry point is compiled for");
  const int esize = (dtype == 2) ? 4 : 2;
  const long nbytes = (long)n_elem * esize;
  if (nbytes % 16 != 0) throw std::runtime_error("all_reduce: n_elem*esize not 16B-aligned");
  if (n_elem > R4D_AR_TWOSHOT_WIDE_MAX_ELEMS)
    throw std::runtime_error("all_reduce: n_elem " + std::to_string(n_elem) +
                             " exceeds the twoshot band (" +
                             std::to_string(R4D_AR_TWOSHOT_WIDE_MAX_ELEMS) +
                             " elems); use the fallback path");
  const long n16 = nbytes / 16;
  if (n16 < WS)
    throw std::runtime_error("all_reduce: message (" + std::to_string(n16) +
                             " 16B words) is shorter than one word a rank");
  // the longest shard (the kernel's uneven split); regions A and B each hold one
  const long sh_max = (n16 + WS - 1) / WS;
  if (2 * sh_max > (long)slot_stride16)
    throw std::runtime_error("all_reduce: shard (" + std::to_string(sh_max) +
                             " 16B words) needs 2x inside the per-source slot stride (" +
                             std::to_string(slot_stride16) + ")");

  int nb = (int)nblocks;
  if (nb < 1) nb = R4D_AR_NB_DESIGN;
  if (nb > sh_max) nb = (int)sh_max;
  if (nb > R4D_AR_WIDE_MAX_BLOCKS) nb = R4D_AR_WIDE_MAX_BLOCKS;
  int nt = (int)nthreads;
  if (nt < 64) nt = 64;
  if (nt > 1024) nt = 1024;

  R4DArWidePeers<int4, NPEERS> peers{};
  for (int i = 0; i < NPEERS; ++i) {
    if (!peer_scratch[i] || !peer_flags[i])
      throw std::runtime_error("all_reduce: null peer scratch/flag pointer at index " +
                               std::to_string(i));
    peers.scratch[i] = (int4*)peer_scratch[i];
    peers.flags[i] = (unsigned int*)peer_flags[i];
    peers.rank[i] = (i < (int)rank) ? i : i + 1;
  }

  // scratch must back 2*WS slots of the declared stride, flags one word per (block, source
  // rank), seq a handshake and a parity word per block of capacity (r4d_ar_seq.h) -- checked
  // against the allocations, not the arguments.
  const long slot_bytes = 2L * WS * (long)slot_stride16 * 16;
  const long flag_bytes = (long)nb * WS * (long)sizeof(unsigned int);
  r4d_ar_check_extent(my_scratch, slot_bytes, "all_reduce", "my_scratch");
  r4d_ar_check_extent(my_flags, flag_bytes, "all_reduce", "my_flags");
  r4d_ar_check_extent(seq_ctrs, 2L * R4D_AR_WIDE_MAX_BLOCKS * (long)sizeof(unsigned int),
                      "all_reduce", "seq_ctrs");
  for (int i = 0; i < NPEERS; ++i) {
    r4d_ar_check_extent(peer_scratch[i], slot_bytes, "all_reduce", "peer scratch");
    r4d_ar_check_extent(peer_flags[i], flag_bytes, "all_reduce", "peer flags");
  }

  hipStream_t st = (hipStream_t)stream_i;
#define R4D_AR_TS_LAUNCH(T, LN)                                                       \
  r4d_ar_twoshot_wide_exact_kernel<T, LN, WS><<<nb, nt, 0, st>>>(                     \
      (const int4*)inp, peers, (const int4*)my_scratch, (unsigned int*)my_flags,      \
      (int4*)out, (unsigned int*)seq_ctrs, (int)n16, (int)slot_stride16,              \
      (int)rank, (int)drain, (int)acq, (int)pub)
  if (dtype == 0) R4D_AR_TS_LAUNCH(__hip_bfloat16, 8);
  else if (dtype == 1) R4D_AR_TS_LAUNCH(__half, 8);
  else if (dtype == 2) R4D_AR_TS_LAUNCH(float, 4);
  else throw std::runtime_error("all_reduce: bad dtype");
#undef R4D_AR_TS_LAUNCH
}

void r4d_ar_twoshot_3rank_exact(long peer_scratch0, long peer_scratch1, long peer_flags0,
                                long peer_flags1, long my_scratch, long my_flags, long seq_ctrs,
                                long slot_stride16, long inp, long out, long n_elem, long dtype,
                                long rank, long stream, long nblocks, long nthreads, long drain,
                                long acq, long pub) {
  const long ps[2] = {peer_scratch0, peer_scratch1};
  const long pf[2] = {peer_flags0, peer_flags1};
  ar_twoshot_launch<3>(ps, pf, my_scratch, my_flags, seq_ctrs, slot_stride16, inp, out, n_elem,
                       dtype, rank, stream, nblocks, nthreads, drain, acq, pub);
}

void r4d_ar_twoshot_4rank_exact(long peer_scratch0, long peer_scratch1, long peer_scratch2,
                                long peer_flags0, long peer_flags1, long peer_flags2,
                                long my_scratch, long my_flags, long seq_ctrs,
                                long slot_stride16, long inp, long out, long n_elem, long dtype,
                                long rank, long stream, long nblocks, long nthreads, long drain,
                                long acq, long pub) {
  const long ps[3] = {peer_scratch0, peer_scratch1, peer_scratch2};
  const long pf[3] = {peer_flags0, peer_flags1, peer_flags2};
  ar_twoshot_launch<4>(ps, pf, my_scratch, my_flags, seq_ctrs, slot_stride16, inp, out, n_elem,
                       dtype, rank, stream, nblocks, nthreads, drain, acq, pub);
}

void r4d_ar_twoshot_8rank_exact(long peer_scratch0, long peer_scratch1, long peer_scratch2,
                                long peer_scratch3, long peer_scratch4, long peer_scratch5,
                                long peer_scratch6, long peer_flags0, long peer_flags1,
                                long peer_flags2, long peer_flags3, long peer_flags4,
                                long peer_flags5, long peer_flags6, long my_scratch,
                                long my_flags, long seq_ctrs, long slot_stride16, long inp,
                                long out, long n_elem, long dtype, long rank, long stream,
                                long nblocks, long nthreads, long drain, long acq, long pub) {
  const long ps[7] = {peer_scratch0, peer_scratch1, peer_scratch2, peer_scratch3,
                      peer_scratch4, peer_scratch5, peer_scratch6};
  const long pf[7] = {peer_flags0, peer_flags1, peer_flags2, peer_flags3,
                      peer_flags4, peer_flags5, peer_flags6};
  ar_twoshot_launch<8>(ps, pf, my_scratch, my_flags, seq_ctrs, slot_stride16, inp, out, n_elem,
                       dtype, rank, stream, nblocks, nthreads, drain, acq, pub);
}
