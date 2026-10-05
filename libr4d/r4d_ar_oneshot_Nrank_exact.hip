// r4d_ar_oneshot_Nrank_exact.hip: r4d_ar_oneshot_4rank_exact,
// r4d_ar_oneshot_8rank_exact -- one-shot P2P push all-reduce, 4 or 8
// ranks, gfx1201. WS is compile-time; the 2-rank kernel keeps its own file.
//
// Per-block handshake: block b waits only on the peers' block b, so push
// and reduce overlap. Scratch slot per (parity, source), flag per (block,
// source). Reduce is fp32 in ascending rank order, one round at the store
// -- required, or ranks diverge. Stuck handshake traps.
// Sends (ws-1)*N bytes per rank: for small messages only, where latency dominates.
// The two-shot covers larger sizes.
//
// Also r4d_ar_hc_write_4rank: the same one-shot with hc_write as its epilogue (ar_hc_write at
// width, exact wire only), and r4d_ar_gather_hc_write_4rank: that with moe_gather as its prologue
// (ar_gather_hc_write at width, exact wire only).
//
// The IPC helpers live in r4d_ar_oneshot_2rank_exact.hip; this unit links beside it.
#include "r4d.h"
#include "r4d_ar_wide.h"
#include "r4d_ar_hc.h"      // the gather and hc write words, shared with the two-rank fused forms
#include <stdexcept>
#include <string>

// src[] is pre-ordered by global rank with our own entry pointing at the local input, so the
// accumulate order is identical on every rank and there is no branch here.
template <typename T, int LANES, int WS>
__device__ __forceinline__ int4 ar_vsum(const int4* const* __restrict__ src, int k) {
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
  return o;
}

template <typename T, int LANES, int WS>
__global__ void r4d_ar_oneshot_wide_exact_kernel(
    const int4* __restrict__ in4, R4DArWidePeers<int4, WS - 1> peers,
    const int4* __restrict__ my_base,
    unsigned int* __restrict__ my_flags, int4* __restrict__ out4,
    unsigned int* __restrict__ seq_ctrs, int n16, int slot_stride16, int rank,
    int drain, int acq, int pub) {
  constexpr int NPEERS = WS - 1;
  // The reduce holds WS source words live per output word, so the 2-rank unroll spills at ws=8.
  constexpr int RU = (WS <= 4) ? 2 : 1;

  const int b = blockIdx.x, nb = gridDim.x, tid = threadIdx.x, nt = blockDim.x;
  __shared__ unsigned int s_seq, s_slot;
  r4d_ar_seq_take(seq_ctrs, R4D_AR_WIDE_MAX_BLOCKS, s_seq, s_slot);  // device-resident, replay-safe
  const unsigned int s = s_seq;
  const int parity = (int)s_slot;

  const int w = (n16 + nb - 1) / nb;
  const int start = b * w;
  int end = start + w; if (end > n16) end = n16;

  // Every peer lays its scratch out identically, so one index serves all of them.
  int4* ps[NPEERS];
#pragma unroll
  for (int p = 0; p < NPEERS; ++p)
    ps[p] = peers.scratch[p] + ((size_t)parity * WS + rank) * (size_t)slot_stride16;

  // push: 4x-unrolled, and the loaded words are reused across peers so the input is read once
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

  // Flag values live in the 2*seq space shared by the whole wide family (the two-shot's
  // half-step is 2s-1, its full step 2s), so one flag/seq/scratch set serves a mixed
  // kernel stream: every kernel advances seq_ctrs[b] once per AR, parity alternates
  // across the stream, and the monotonic wait stays coherent at a band crossover. The
  // slot guard is the same shared discipline: each AR's writes land in its (parity,
  // source) slot, the handshake keeps any rank at most one AR ahead, and the caller
  // sizes the stride for the largest family in the mix -- so a crossover neighbour can
  // never overlap this AR's bytes.
  r4d_ar_handshake<WS>(peers, my_flags, b, rank, 2u * s, drain, acq, pub);

  // Our own contribution comes from the input; slot[my rank] in my scratch is never written.
  const int4* src[WS];
#pragma unroll
  for (int r = 0; r < WS; ++r)
    src[r] = (r == rank) ? in4
                         : (my_base + ((size_t)parity * WS + r) * (size_t)slot_stride16);
  {
    int k = start + tid;
    if (RU == 2) {
      for (; k + nt < end; k += 2 * nt) {
        out4[k]      = ar_vsum<T, LANES, WS>(src, k);
        out4[k + nt] = ar_vsum<T, LANES, WS>(src, k + nt);
      }
    }
    for (; k < end; k += nt) out4[k] = ar_vsum<T, LANES, WS>(src, k);
  }
}

// THE SAME ONE-SHOT WITH THE GATED RESIDUAL'S WRITE AS ITS EPILOGUE, bf16 only. `y` is the
// message and is reduced IN PLACE; everything up to the reduce is the kernel above, step for step,
// because both run on one instance's buffers in one stream: the same seq take, the same handshake
// value 2*s, the same (parity, source) slots and the same block partition of the word space. The
// reduce is the same ar_vsum, so the stored word is the plain all-reduce's to the bit; then
// h[r][c*n + i] += inj[r][c] * y[r][i] in hc_write's arithmetic over that bf16 word
// (r4d_ar_hc_write_word, as the two-rank r4d_ar_hc_write_kernel).
//
// In place is safe: a thread reads every source of word k before it stores word k, its own block
// pushed that word to the peers before the handshake, and no peer reads this rank's `y`. A word is
// eight elements and n is a multiple of eight, so a word never straddles a row and the thread that
// reduces it owns every stream element behind it.
template <int WS>
__global__ void r4d_ar_hc_write_wide_kernel(
    int4* __restrict__ y4, R4DArWidePeers<int4, WS - 1> peers, const int4* __restrict__ my_base,
    unsigned int* __restrict__ my_flags, unsigned int* __restrict__ seq_ctrs,
    const unsigned short* __restrict__ inj, unsigned short* __restrict__ h, int n16, int n,
    int hc, long irow, long hrow, int slot_stride16, int rank, int drain, int acq, int pub) {
  constexpr int NPEERS = WS - 1;

  const int b = blockIdx.x, nb = gridDim.x, tid = threadIdx.x, nt = blockDim.x;
  __shared__ unsigned int s_seq, s_slot;
  r4d_ar_seq_take(seq_ctrs, R4D_AR_WIDE_MAX_BLOCKS, s_seq, s_slot);  // device-resident, replay-safe
  const unsigned int s = s_seq;
  const int parity = (int)s_slot;

  const int w = (n16 + nb - 1) / nb;
  const int start = b * w;
  int end = start + w; if (end > n16) end = n16;

  int4* ps[NPEERS];
#pragma unroll
  for (int p = 0; p < NPEERS; ++p)
    ps[p] = peers.scratch[p] + ((size_t)parity * WS + rank) * (size_t)slot_stride16;

  {
    int k = start + tid;
    for (; k + 3 * nt < end; k += 4 * nt) {
      const int4 v0 = y4[k], v1 = y4[k + nt], v2 = y4[k + 2 * nt], v3 = y4[k + 3 * nt];
#pragma unroll
      for (int p = 0; p < NPEERS; ++p) {
        ps[p][k] = v0; ps[p][k + nt] = v1; ps[p][k + 2 * nt] = v2; ps[p][k + 3 * nt] = v3;
      }
    }
    for (; k < end; k += nt) {
      const int4 v = y4[k];
#pragma unroll
      for (int p = 0; p < NPEERS; ++p) ps[p][k] = v;
    }
  }

  r4d_ar_handshake<WS>(peers, my_flags, b, rank, 2u * s, drain, acq, pub);

  const int4* src[WS];
#pragma unroll
  for (int r = 0; r < WS; ++r)
    src[r] = (r == rank) ? (const int4*)y4
                         : (my_base + ((size_t)parity * WS + r) * (size_t)slot_stride16);

  const int wpr = n >> 3;                 // 16B words a row
  for (int k = start + tid; k < end; k += nt) {
    const int4 v = ar_vsum<__hip_bfloat16, 8, WS>(src, k);
    y4[k] = v;
    const int row = k / wpr, i0 = (k - row * wpr) * 8;
    r4d_ar_hc_write_word(inj, h, row, i0, n, hc, irow, hrow,
                         reinterpret_cast<const unsigned short*>(&v));
  }
}

// THE SAME WRITE WITH THE MoE GATHER AS ITS PROLOGUE, bf16 only: the four-rank form of the
// two-rank r4d_ar_gather_hc_write_kernel. `y` is not read but produced -- a block's words are
// summed back to token order from the routed experts' rows by r4d_ar_gather_word, kept as this
// rank's message and pushed to every peer, exactly as if moe_gather had written y and the kernel
// above had read it back. From the handshake on it IS the kernel above: the same seq take, the
// same value 2*s, the same (parity, source) slots, the same ar_vsum in ascending rank order over
// the same sources -- this rank's own word from y, which the same thread stored before the
// handshake -- and the same r4d_ar_hc_write_word over the stored bf16 word. So the bytes are those
// of moe_gather, the four-rank exact all-reduce and hc_write.
//
// The slot table is the two-rank kernel's: without INV a block's rows (at most R4D_ARG_ROWS, which
// the launcher sizes the grid for) find their slots once, in LDS, by one pass over the sorted
// list; with INV they read the inverse table r4d_ar_gather_inv built in front.
template <int WS, bool INV>
__global__ void r4d_ar_gather_hc_write_wide_kernel(
    int4* __restrict__ y4, R4DArWidePeers<int4, WS - 1> peers, const int4* __restrict__ my_base,
    unsigned int* __restrict__ my_flags, unsigned int* __restrict__ seq_ctrs,
    const unsigned short* __restrict__ inj, unsigned short* __restrict__ h, int n16, int n,
    int hc, long irow, long hrow, int slot_stride16, int rank, int drain, int acq, int pub,
    R4dArGather G, const int* __restrict__ inv) {
  constexpr int NPEERS = WS - 1;

  const int b = blockIdx.x, nb = gridDim.x, tid = threadIdx.x, nt = blockDim.x;
  __shared__ unsigned int s_seq, s_slot;
  __shared__ int   s_i[R4D_ARG_ROWS * R4D_ARG_SLOTS];
  __shared__ float s_w[R4D_ARG_ROWS * R4D_ARG_SLOTS];
  __shared__ float s_g[R4D_ARG_ROWS];
  r4d_ar_seq_take(seq_ctrs, R4D_AR_WIDE_MAX_BLOCKS, s_seq, s_slot);  // device-resident, replay-safe
  const unsigned int s = s_seq;
  const int parity = (int)s_slot;

  const int w = (n16 + nb - 1) / nb;
  const int start = b * w;
  int end = start + w; if (end > n16) end = n16;
  const int wpr = n >> 3;                 // 16B words a row
  const int row_lo = start / wpr;
  const int nrow = end > start ? (end - 1) / wpr - row_lo + 1 : 0;
  const int K = G.top_k;

  int4* ps[NPEERS];
#pragma unroll
  for (int p = 0; p < NPEERS; ++p)
    ps[p] = peers.scratch[p] + ((size_t)parity * WS + rank) * (size_t)slot_stride16;

  /* The slot table. A slot no sorted row claims reads -1, so the clear and the scan are two
   * phases with a barrier between them, as in moe_gather. */
  if (!INV) {
    for (int x = tid; x < nrow * K; x += nt) s_i[x] = -1;
    __syncthreads();
    {
      const int lo = row_lo * K, hi = (row_lo + nrow) * K;
      for (int i = tid; i < G.T; i += nt) {
        const int f = G.sorted[i];
        if (f >= lo && f < hi) s_i[f - lo] = i;
      }
    }
    for (int x = tid; x < nrow * K; x += nt)
      s_w[x] = bf2f(G.ew[(long)(row_lo + x / K) * G.ew_ld + x % K]);
    if (G.sh)
      for (int r = tid; r < nrow; r += nt) {
        const float v0 = bf2f(G.sg[(long)(row_lo + r) * G.sg_ld]);
        s_g[r] = G.sig ? (1.0f / (1.0f + __expf(-v0))) : v0;
      }
    __syncthreads();
  }

  /* The gather, one word a thread a pass, kept and pushed to every peer. */
  for (int k = start + tid; k < end; k += nt) {
    const int row = k / wpr, i0 = (k - row * wpr) * 8;
    const int4 out = r4d_ar_gather_word<INV>(G, inv, s_i, s_w, s_g, row, row - row_lo, i0);
    y4[k] = out;
#pragma unroll
    for (int p = 0; p < NPEERS; ++p) ps[p][k] = out;
  }

  r4d_ar_handshake<WS>(peers, my_flags, b, rank, 2u * s, drain, acq, pub);

  const int4* src[WS];
#pragma unroll
  for (int r = 0; r < WS; ++r)
    src[r] = (r == rank) ? (const int4*)y4
                         : (my_base + ((size_t)parity * WS + r) * (size_t)slot_stride16);

  for (int k = start + tid; k < end; k += nt) {
    const int4 v = ar_vsum<__hip_bfloat16, 8, WS>(src, k);
    y4[k] = v;
    const int row = k / wpr, i0 = (k - row * wpr) * 8;
    r4d_ar_hc_write_word(inj, h, row, i0, n, hc, irow, hrow,
                         reinterpret_cast<const unsigned short*>(&v));
  }
}

// --- host side ---
template <int WS>
static void ar_wide_launch(const long* peer_scratch, const long* peer_flags, long my_scratch,
                           long my_flags, long seq_ctrs, long slot_stride16, long inp, long out,
                           long n_elem, long dtype, long rank, long stream_i, long nblocks,
                           long nthreads, long drain, long acq, long pub) {
  constexpr int NPEERS = WS - 1;
  static_assert(NPEERS <= R4D_AR_WIDE_MAX_PEERS, "npeers exceeds the peer bundle bound");

  if (rank < 0 || rank >= WS)
    throw std::runtime_error("all_reduce: rank " + std::to_string(rank) +
                             " out of range for the " + std::to_string(WS) +
                             " ranks this entry point is compiled for");

  const int esize = (dtype == 2) ? 4 : 2;
  const long nbytes = (long)n_elem * esize;
  if (nbytes % 16 != 0) throw std::runtime_error("all_reduce: n_elem*esize not 16B-aligned");
  const long n16l = nbytes / 16;
  if (n16l > (long)slot_stride16)
    throw std::runtime_error("all_reduce: message (" + std::to_string(n16l) +
                             " 16B words) exceeds the per-source slot stride (" +
                             std::to_string(slot_stride16) + ")");
  const int n16 = (int)n16l;

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
#define R4D_AR_WIDE_LAUNCH(T, LN)                                                     \
  r4d_ar_oneshot_wide_exact_kernel<T, LN, WS><<<nb, nt, 0, st>>>(                     \
      (const int4*)inp, peers, (const int4*)my_scratch, (unsigned int*)my_flags,      \
      (int4*)out, (unsigned int*)seq_ctrs, n16, (int)slot_stride16, (int)rank,        \
      (int)drain, (int)acq, (int)pub)
  if (dtype == 0) R4D_AR_WIDE_LAUNCH(__hip_bfloat16, 8);
  else if (dtype == 1) R4D_AR_WIDE_LAUNCH(__half, 8);
  else if (dtype == 2) R4D_AR_WIDE_LAUNCH(float, 4);
  else throw std::runtime_error("all_reduce: bad dtype");
#undef R4D_AR_WIDE_LAUNCH
}

void r4d_ar_oneshot_4rank_exact(long peer_scratch0, long peer_scratch1, long peer_scratch2,
                                long peer_flags0, long peer_flags1, long peer_flags2,
                                long my_scratch, long my_flags, long seq_ctrs, long slot_stride16,
                                long inp, long out, long n_elem, long dtype, long rank,
                                long stream, long nblocks, long nthreads, long drain, long acq,
                                long pub) {
  const long ps[3] = {peer_scratch0, peer_scratch1, peer_scratch2};
  const long pf[3] = {peer_flags0, peer_flags1, peer_flags2};
  ar_wide_launch<4>(ps, pf, my_scratch, my_flags, seq_ctrs, slot_stride16, inp, out, n_elem,
                    dtype, rank, stream, nblocks, nthreads, drain, acq, pub);
}

void r4d_ar_oneshot_8rank_exact(long peer_scratch0, long peer_scratch1, long peer_scratch2,
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
  ar_wide_launch<8>(ps, pf, my_scratch, my_flags, seq_ctrs, slot_stride16, inp, out, n_elem,
                    dtype, rank, stream, nblocks, nthreads, drain, acq, pub);
}

// ar_wide_launch's checks, for the fused write: the same rank range, slot stride, grid clamps and
// allocation extents, plus the write's own two shape rules.
template <int WS>
static void ar_hc_write_wide_launch(const long* peer_scratch, const long* peer_flags,
                                    long my_scratch, long my_flags, long seq_ctrs,
                                    long slot_stride16, long y, long inj, long h, long M, long n,
                                    long hc, long irow, long hrow, long rank, long stream_i,
                                    long nblocks, long nthreads, long drain, long acq, long pub) {
  constexpr int NPEERS = WS - 1;
  static_assert(NPEERS <= R4D_AR_WIDE_MAX_PEERS, "npeers exceeds the peer bundle bound");

  if (rank < 0 || rank >= WS)
    throw std::runtime_error("ar_hc_write: rank " + std::to_string(rank) +
                             " out of range for the " + std::to_string(WS) +
                             " ranks this entry point is compiled for");
  if (M <= 0 || n <= 0 || n % 8)
    throw std::runtime_error("ar_hc_write: n must be a positive multiple of 8");
  if (hc < 1 || hrow % 8)
    throw std::runtime_error("ar_hc_write: the streams must be whole 16B words");

  const long n16l = M * n / 8;            // bf16: eight elements a 16B word
  if (n16l > (long)slot_stride16)
    throw std::runtime_error("ar_hc_write: message (" + std::to_string(n16l) +
                             " 16B words) exceeds the per-source slot stride (" +
                             std::to_string(slot_stride16) + ")");
  const int n16 = (int)n16l;

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
      throw std::runtime_error("ar_hc_write: null peer scratch/flag pointer at index " +
                               std::to_string(i));
    peers.scratch[i] = (int4*)peer_scratch[i];
    peers.flags[i] = (unsigned int*)peer_flags[i];
    peers.rank[i] = (i < (int)rank) ? i : i + 1;
  }

  const long slot_bytes = 2L * WS * (long)slot_stride16 * 16;
  const long flag_bytes = (long)nb * WS * (long)sizeof(unsigned int);
  r4d_ar_check_extent(my_scratch, slot_bytes, "ar_hc_write", "my_scratch");
  r4d_ar_check_extent(my_flags, flag_bytes, "ar_hc_write", "my_flags");
  r4d_ar_check_extent(seq_ctrs, 2L * R4D_AR_WIDE_MAX_BLOCKS * (long)sizeof(unsigned int),
                      "ar_hc_write", "seq_ctrs");
  for (int i = 0; i < NPEERS; ++i) {
    r4d_ar_check_extent(peer_scratch[i], slot_bytes, "ar_hc_write", "peer scratch");
    r4d_ar_check_extent(peer_flags[i], flag_bytes, "ar_hc_write", "peer flags");
  }

  hipStream_t st = (hipStream_t)stream_i;
  r4d_ar_hc_write_wide_kernel<WS><<<nb, nt, 0, st>>>(
      (int4*)y, peers, (const int4*)my_scratch, (unsigned int*)my_flags, (unsigned int*)seq_ctrs,
      (const unsigned short*)inj, (unsigned short*)h, n16, (int)n, (int)hc, irow, hrow,
      (int)slot_stride16, (int)rank, (int)drain, (int)acq, (int)pub);
}

void r4d_ar_hc_write_4rank(long peer_scratch0, long peer_scratch1, long peer_scratch2,
                           long peer_flags0, long peer_flags1, long peer_flags2,
                           long my_scratch, long my_flags, long seq_ctrs, long slot_stride16,
                           long y, long inj, long h, long M, long n, long hc, long irow,
                           long hrow, long rank, long stream, long nblocks, long nthreads,
                           long drain, long acq, long pub) {
  const long ps[3] = {peer_scratch0, peer_scratch1, peer_scratch2};
  const long pf[3] = {peer_flags0, peer_flags1, peer_flags2};
  ar_hc_write_wide_launch<4>(ps, pf, my_scratch, my_flags, seq_ctrs, slot_stride16, y, inj, h,
                             M, n, hc, irow, hrow, rank, stream, nblocks, nthreads, drain, acq,
                             pub);
}

// ar_hc_write_wide_launch's checks plus the gather's, and the two-rank gather's grid rule: at least
// wide enough that no block spans more than R4D_ARG_ROWS rows, or -- where that would take more
// blocks than the reduction asked for and `inv` is given -- the reduction's count over the inverse
// table. Every rank computes the same count and the same choice from the same M, n and scratch.
template <int WS>
static void ar_gather_hc_write_wide_launch(
    const long* peer_scratch, const long* peer_flags, long my_scratch, long my_flags,
    long seq_ctrs, long slot_stride16, long y, long inj, long h, long M, long n, long hc,
    long irow, long hrow, long ye, long ew, long sorted, long sh, long sg, long ye_ld,
    long ew_ld, long sh_ld, long sg_ld, long top_k, long sig, long rank, long stream_i,
    long nblocks, long nthreads, long drain, long acq, long pub, long inv) {
  constexpr int NPEERS = WS - 1;
  static_assert(NPEERS <= R4D_AR_WIDE_MAX_PEERS, "npeers exceeds the peer bundle bound");

  if (rank < 0 || rank >= WS)
    throw std::runtime_error("ar_gather_hc_write: rank " + std::to_string(rank) +
                             " out of range for the " + std::to_string(WS) +
                             " ranks this entry point is compiled for");
  if (M <= 0 || n <= 0 || n % 8)
    throw std::runtime_error("ar_gather_hc_write: n must be a positive multiple of 8");
  if (hc < 1 || hrow % 8)
    throw std::runtime_error("ar_gather_hc_write: the streams must be whole 16B words");
  if (top_k < 1 || top_k > R4D_ARG_SLOTS)
    throw std::runtime_error("ar_gather_hc_write: top_k " + std::to_string(top_k) +
                             " outside 1.." + std::to_string(R4D_ARG_SLOTS));
  if (!ye || !ew || !sorted || ye_ld % 8 || (sh && sh_ld < n) || (sh != 0) != (sg != 0))
    throw std::runtime_error("ar_gather_hc_write: the gather's operands are not 16B rows");

  const long n16l = M * n / 8;            // bf16: eight elements a 16B word
  if (n16l > (long)slot_stride16)
    throw std::runtime_error("ar_gather_hc_write: message (" + std::to_string(n16l) +
                             " 16B words) exceeds the per-source slot stride (" +
                             std::to_string(slot_stride16) + ")");
  const int n16 = (int)n16l;
  const int wpr = (int)(n / 8);

  int nb = (int)nblocks;
  if (nb < 1) nb = R4D_AR_NB_DESIGN;
  if (nb > R4D_AR_WIDE_MAX_BLOCKS) nb = R4D_AR_WIDE_MAX_BLOCKS;
  /* A block of w words starting anywhere touches at most (w - 1) / wpr + 2 rows. */
  const int wmax = (R4D_ARG_ROWS - 2) * wpr + 1;
  const int nb_min = (n16 + wmax - 1) / wmax;
  const bool by_inv = inv != 0 && nb_min > nb;
  if (!by_inv && nb < nb_min) nb = nb_min;
  if (nb > n16) nb = n16;
  if (nb > R4D_AR_WIDE_MAX_BLOCKS)
    throw std::runtime_error("ar_gather_hc_write: the slot table needs " + std::to_string(nb) +
                             " blocks, above the " + std::to_string(R4D_AR_WIDE_MAX_BLOCKS) +
                             " the flags hold, and no inverse table was given");
  int nt = (int)nthreads;
  if (nt < 64) nt = 64;
  if (nt > 1024) nt = 1024;

  R4DArWidePeers<int4, NPEERS> peers{};
  for (int i = 0; i < NPEERS; ++i) {
    if (!peer_scratch[i] || !peer_flags[i])
      throw std::runtime_error("ar_gather_hc_write: null peer scratch/flag pointer at index " +
                               std::to_string(i));
    peers.scratch[i] = (int4*)peer_scratch[i];
    peers.flags[i] = (unsigned int*)peer_flags[i];
    peers.rank[i] = (i < (int)rank) ? i : i + 1;
  }

  const long slot_bytes = 2L * WS * (long)slot_stride16 * 16;
  const long flag_bytes = (long)nb * WS * (long)sizeof(unsigned int);
  r4d_ar_check_extent(my_scratch, slot_bytes, "ar_gather_hc_write", "my_scratch");
  r4d_ar_check_extent(my_flags, flag_bytes, "ar_gather_hc_write", "my_flags");
  r4d_ar_check_extent(seq_ctrs, 2L * R4D_AR_WIDE_MAX_BLOCKS * (long)sizeof(unsigned int),
                      "ar_gather_hc_write", "seq_ctrs");
  for (int i = 0; i < NPEERS; ++i) {
    r4d_ar_check_extent(peer_scratch[i], slot_bytes, "ar_gather_hc_write", "peer scratch");
    r4d_ar_check_extent(peer_flags[i], flag_bytes, "ar_gather_hc_write", "peer flags");
  }

  R4dArGather G;
  G.ye = (const unsigned short*)ye;
  G.ew = (const unsigned short*)ew;
  G.sorted = (const int*)sorted;
  G.sh = (const unsigned short*)sh;
  G.sg = (const unsigned short*)sg;
  G.ye_ld = ye_ld; G.ew_ld = ew_ld; G.sh_ld = sh_ld; G.sg_ld = sg_ld;
  G.top_k = (int)top_k; G.T = (int)(M * top_k); G.sig = (int)sig;

  hipStream_t st = (hipStream_t)stream_i;
  if (by_inv) {
    const int irc = r4d_ar_gather_inv(inv, sorted, (long)G.T, stream_i);
    if (irc != 0)
      throw std::runtime_error("ar_gather_hc_write: the inverse table's launch failed (" +
                               std::to_string(irc) + ")");
    r4d_ar_gather_hc_write_wide_kernel<WS, true><<<nb, nt, 0, st>>>(
        (int4*)y, peers, (const int4*)my_scratch, (unsigned int*)my_flags,
        (unsigned int*)seq_ctrs, (const unsigned short*)inj, (unsigned short*)h, n16, (int)n,
        (int)hc, irow, hrow, (int)slot_stride16, (int)rank, (int)drain, (int)acq, (int)pub, G,
        (const int*)inv);
  } else {
    r4d_ar_gather_hc_write_wide_kernel<WS, false><<<nb, nt, 0, st>>>(
        (int4*)y, peers, (const int4*)my_scratch, (unsigned int*)my_flags,
        (unsigned int*)seq_ctrs, (const unsigned short*)inj, (unsigned short*)h, n16, (int)n,
        (int)hc, irow, hrow, (int)slot_stride16, (int)rank, (int)drain, (int)acq, (int)pub, G,
        nullptr);
  }
}

void r4d_ar_gather_hc_write_4rank(long peer_scratch0, long peer_scratch1, long peer_scratch2,
                                  long peer_flags0, long peer_flags1, long peer_flags2,
                                  long my_scratch, long my_flags, long seq_ctrs,
                                  long slot_stride16, long y, long inj, long h, long M, long n,
                                  long hc, long irow, long hrow, long ye, long ew, long sorted,
                                  long sh, long sg, long ye_ld, long ew_ld, long sh_ld,
                                  long sg_ld, long top_k, long sig, long rank, long stream,
                                  long nblocks, long nthreads, long drain, long acq, long pub,
                                  long inv) {
  const long ps[3] = {peer_scratch0, peer_scratch1, peer_scratch2};
  const long pf[3] = {peer_flags0, peer_flags1, peer_flags2};
  ar_gather_hc_write_wide_launch<4>(ps, pf, my_scratch, my_flags, seq_ctrs, slot_stride16, y,
                                    inj, h, M, n, hc, irow, hrow, ye, ew, sorted, sh, sg, ye_ld,
                                    ew_ld, sh_ld, sg_ld, top_k, sig, rank, stream, nblocks,
                                    nthreads, drain, acq, pub, inv);
}

void r4d_ar_wide_dims(int* max_blocks, int* nb_design, int* max_peers) {
  if (max_blocks) *max_blocks = R4D_AR_WIDE_MAX_BLOCKS;
  if (nb_design) *nb_design = R4D_AR_NB_DESIGN;
  if (max_peers) *max_peers = R4D_AR_WIDE_MAX_PEERS;
}
