/* avx_check_rows.cpp -- embed_lookup_q over a FILE-BACKED table, the form a model serves it in.
 *
 * The geometry table checks the op against libref over a heap buffer, which the gather reads as
 * memory. A table in a container is not read that way: its rows come through a row cache every
 * thread shares, with direct reads of the file for the cache's misses and a reader thread a call
 * ahead (avx_ngram.cpp). What can go wrong there is WHICH bytes arrive -- a slot claimed twice, a
 * row decoded from a slot that was taken over, a wait on a claim that was given up -- and libref
 * has no such path to compare against, so the answer here is the file's own bytes.
 *
 * Two threads play the ranks of one call, as the engine's rank threads do: the same ids, rank 0
 * and 1 of 2, released together, so a thread claims rows, waits on rows the other has claimed, and
 * meets its own claim again when an id repeats within a call. Half the ids come from a small hot
 * set the cache keeps; the rest from a table four times the cache's 2^19 rows, so sets fill and
 * rows are evicted while the other thread may be reading them. Every call also hands over the NEXT
 * call's ids as `ahead`, as the engine's prefill does, so the reader thread claims and reads those
 * rows while this call's threads are still decoding theirs: a third thread in every race above.
 *
 * Every output row is compared bit for bit with that id's bytes in the file through the call's
 * scale, and a padding id must come back as a zero row. The row is 40 codes, so the AVX-512 decode
 * takes one full 32-code step and a masked tail. The table is a file in `dir`, and the filesystem
 * there decides how the gather aligns its direct reads: by the alignment the filesystem reports
 * (ext4, XFS), or by its block size where it reports none (btrfs, and a tmpfs since Linux 6.6). The
 * case says which it ran.
 *
 * With `resident` the op is asked to copy the table into locked RAM (--ngram-table ram), over a
 * table small enough for a default memlock limit. The calls run while the copy is being made, then
 * the case waits for the process's locked memory to grow by the table and runs them again, from
 * the copy. Where the process may not lock that much the gather says so and keeps reading the file,
 * and the case says which of the two it checked; either way every row must be the file's. */
#include "avx_harness.h"

#include <barrier>
#include <chrono>
#include <cerrno>
#include <cstring>
#include <thread>

#include <cstddef>

#include <fcntl.h>
#include <linux/filter.h>
#include <linux/io_uring.h>
#include <linux/seccomp.h>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <unistd.h>

/* EMBED_LOOKUP_Q'S INIT REFUSES A PROCESS THAT MAY NOT MAKE AN io_uring RING, and says why: Docker's
 * default seccomp profile answers EPERM to io_uring_setup, and a server started that way used to
 * load the whole model and then fail every request as "unsupported". A child process installs the
 * same refusal as a seccomp filter and asks the kernel's init, which must refuse a gather into
 * bf16 -- the one that reads through the ring -- name the Docker option on stderr, and accept a
 * gather into f32, which reads the table as memory. Then this process asks: it must accept where a
 * ring can be made and refuse where it cannot, as in a build sandbox. */
bool ngram_init_case(const Lib& avx) {
    const char* what = "embed_lookup_q's init";
    const RadKernelInfo* r = avx.row("embed_lookup_q");
    if (!r || !r->init) {
        std::printf("  %s: the row has no init\n", what);
        return false;
    }
    RadParam bf16[2]{}, f32[2]{};
    bf16[0].key = f32[0].key = "n_embd";
    bf16[0].kind = f32[0].kind = RAD_P_INT;
    bf16[0].ival = f32[0].ival = 160;
    bf16[1].key = f32[1].key = "dtype";
    bf16[1].kind = f32[1].kind = RAD_P_STR;
    bf16[1].sval = "bf16";
    f32[1].sval = "f32";

    int out[2];
    if (::pipe(out) != 0) {
        std::printf("  %s: pipe: %s\n", what, std::strerror(errno));
        return false;
    }
    std::fflush(nullptr);
    const pid_t pid = ::fork();
    if (pid == 0) {
        ::dup2(out[1], 2);
        ::close(out[0]);
        sock_filter f[] = {
            BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(seccomp_data, nr)),
            BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_io_uring_setup, 0, 1),
            BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ERRNO | (EPERM & SECCOMP_RET_DATA)),
            BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW),
        };
        sock_fprog prog{ (unsigned short)(sizeof f / sizeof f[0]), f };
        if (::prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0 ||
            ::prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, &prog) != 0)
            ::_exit(3);
        void* inst = nullptr;
        const int a = r->init(bf16, 2, 0, 1, &inst);
        const int b = r->init(f32, 2, 0, 1, &inst);
        std::fflush(nullptr);
        ::_exit(a == RAD_E_UNSUPPORTED && b == RAD_OK ? 0 : 1);
    }
    ::close(out[1]);
    std::string said;
    char buf[512];
    for (ssize_t k; (k = ::read(out[0], buf, sizeof buf)) > 0;) said.append(buf, (size_t)k);
    ::close(out[0]);
    int ws = 0;
    if (pid < 0 || ::waitpid(pid, &ws, 0) != pid || !WIFEXITED(ws)) {
        std::printf("  %s: the child did not run to the end\n", what);
        return false;
    }
    bool ok = true;
    if (WEXITSTATUS(ws) == 3) {
        std::printf("  %s, where io_uring is refused: SKIPPED -- no seccomp filter can be installed "
                    "here\n", what);
    } else if (WEXITSTATUS(ws) != 0 ||
               said.find("--security-opt seccomp=unconfined") == std::string::npos) {
        std::printf("  %s, where io_uring is refused: FAIL -- %s; it said: %s\n", what,
                    WEXITSTATUS(ws) ? "the statuses were wrong" : "the reason was not named",
                    said.c_str());
        ok = false;
    } else {
        std::printf("  %s, where io_uring is refused: refused, naming the Docker option\n", what);
    }

    io_uring_params p{};
    const long ring = syscall(__NR_io_uring_setup, 1, &p);
    const bool here = ring >= 0;
    if (here) ::close((int)ring);
    void* inst = nullptr;
    const int st = r->init(bf16, 2, 0, 1, &inst);
    const bool right = here ? st == RAD_OK : st == RAD_E_UNSUPPORTED;
    std::printf("  %s, in this process (io_uring %s): %s\n", what, here ? "available" : "refused",
                right ? (here ? "accepted" : "refused") : "FAIL");
    return ok && right;
}

/* This process's locked memory, in bytes: VmLck. */
static long long locked_bytes() {
    long long kb = 0;
    if (std::FILE* f = std::fopen("/proc/self/status", "re")) {
        char line[256];
        while (std::fgets(line, sizeof line, f)) (void)std::sscanf(line, "VmLck: %lld", &kb);
        std::fclose(f);
    }
    return kb << 10;
}

bool row_gather_case(const Lib& avx, const std::string& dir, int64_t calls, bool verbose,
                     bool bf16, bool resident) {
    const RadKernelInfo* r = avx.row("embed_lookup_q");
    const char* what = resident ? "embed_lookup_q from a file, resident"
                     : bf16     ? "embed_lookup_q from a file, bf16 table"
                                : "embed_lookup_q from a file";
    if (!r || !r->launch) {
        std::printf("  %s: no row\n", what);
        return false;
    }
    /* A bf16 table is twice the bytes a row and is read with no scale: its rows are their output,
     * so what this checks there is the cache and the reads at a row size the E4M3 table never
     * has. */
    /* The gather's misses are io_uring reads, and a sandbox can forbid io_uring outright -- Docker's
     * default seccomp profile does. There the gather refuses every call by design, so the case
     * says why it cannot run rather than counting a refusal per call. */
    {
        io_uring_params p{};
        const long ring = syscall(__NR_io_uring_setup, 1, &p);
        if (ring < 0) {
            std::printf("  %s: SKIPPED -- io_uring is unavailable here (%s)\n", what,
                        std::strerror(errno));
            return true;
        }
        ::close((int)ring);
    }
    /* 2.5 MiB resident, which a default memlock limit of 8 MiB holds. */
    const int64_t n = 40, V = int64_t(1) << (resident ? 16 : 21), M = 512, hot = 4096;
    const int64_t rb = bf16 ? 2 * n : n;
    const float scale = 0.375f;

    const std::string path = dir + "/rad-avx-check-rows.bin";
    uint64_t x = 0x9e3779b97f4a7c15ull;
    auto rnd = [&x]() { x ^= x << 13; x ^= x >> 7; x ^= x << 17; return x; };
    {
        const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
        if (fd < 0) { std::printf("  %s: cannot create %s\n", what, path.c_str());
                      return false; }
        std::vector<uint64_t> chunk(1 << 16);
        bool wrote = true;
        for (int64_t done = 0; done < V * rb && wrote; done += (int64_t)(chunk.size() * 8)) {
            for (auto& w : chunk) w = rnd();
            const size_t want = (size_t)std::min<int64_t>(V * rb - done, (int64_t)chunk.size() * 8);
            wrote = ::write(fd, chunk.data(), want) == (ssize_t)want;
        }
        wrote = wrote && ::fsync(fd) == 0;
        ::close(fd);
        if (!wrote) { ::unlink(path.c_str()); std::printf("  %s: short write\n", what);
                      return false; }
    }
    /* A filesystem without direct I/O -- one that refuses the open, as a tmpfs did before Linux
     * 6.6, or one that reports the file has none -- is one the gather refuses by design, so the
     * case cannot run there; it says so rather than counting a refusal per call. Every other
     * filesystem the gather has to read, with or without a reported alignment. */
    bool reported = false;
    {
        const int dfd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECT);
        struct statx sx {};
        const bool dio = dfd >= 0 &&
                         ::statx(dfd, "", AT_EMPTY_PATH, STATX_DIOALIGN, &sx) == 0 &&
                         !((sx.stx_mask & STATX_DIOALIGN) && sx.stx_dio_offset_align == 0);
        reported = (sx.stx_mask & STATX_DIOALIGN) != 0;
        if (dfd >= 0) ::close(dfd);
        if (!dio) {
            ::unlink(path.c_str());
            std::printf("  %s: SKIPPED -- %s cannot do direct I/O\n", what, dir.c_str());
            return true;
        }
    }
    const char* granule = reported ? "the alignment the filesystem reports"
                                   : "no alignment reported, read by the block size";
    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    void* map = fd < 0 ? MAP_FAILED : mmap(nullptr, (size_t)(V * rb), PROT_READ, MAP_SHARED, fd, 0);
    if (fd >= 0) ::close(fd);
    if (map == MAP_FAILED) { ::unlink(path.c_str()); std::printf("  %s: mmap\n", what);
                             return false; }
    const uint8_t* tab = static_cast<const uint8_t*>(map);

    std::vector<int32_t> hotset((size_t)hot);
    for (auto& h : hotset) h = (int32_t)(rnd() % (uint64_t)V);
    std::vector<int32_t> ids((size_t)(calls * M));
    for (int64_t c = 0; c < calls; ++c)
        for (int64_t m = 0; m < M; ++m) {
            const uint64_t v = rnd();
            int32_t& id = ids[(size_t)(c * M + m)];
            if (m > 0 && v % 29 == 0)  id = ids[(size_t)(c * M + m - 1)];   /* repeated in a call */
            else if (v % 16 == 0)      id = -1;                             /* padding */
            else if (v % 16 < 8)       id = hotset[(v >> 8) % (uint64_t)hot];
            else                       id = (int32_t)((v >> 8) % (uint64_t)V);
        }
    uint16_t lut[256];
    for (int b = 0; b < 256; ++b) lut[b] = rad_f32_to_bf16(rad_fp8e4m3_to_f32((uint8_t)b) * scale);

    auto tensor = [](void* p, uint32_t dt, std::initializer_list<int64_t> shape) {
        RadTensor t{};
        t.data = p;
        t.dtype = dt;
        t.rank = (uint32_t)shape.size();
        int i = 0;
        for (int64_t s : shape) t.shape[i++] = s;
        int64_t st = 1;
        for (int k = (int)t.rank - 1; k >= 0; --k) { t.stride[k] = st; st *= t.shape[k]; }
        return t;
    };

    std::barrier<> go(2);
    int64_t bad[2] = {0, 0}, refused[2] = {0, 0};
    auto rank = [&](int rk) {
        uint16_t sc = rad_f32_to_bf16(scale);
        std::vector<uint16_t> out((size_t)(M * n));
        std::vector<RadTensor> ts = {
            tensor(nullptr, RAD_I32, {M}),
            tensor(const_cast<uint8_t*>(tab), bf16 ? RAD_BF16 : RAD_F8E4M3, {V, n}),
            tensor(bf16 ? nullptr : &sc, RAD_BF16, {1}), tensor(out.data(), RAD_BF16, {M, n}),
            tensor(nullptr, RAD_I32, {M}) };
        RadParam ps[4] = {};
        ps[0].key = "M";        ps[0].kind = RAD_P_INT; ps[0].ival = M;
        ps[1].key = "n_embd";   ps[1].kind = RAD_P_INT; ps[1].ival = n;
        ps[2].key = "n_vocab";  ps[2].kind = RAD_P_INT; ps[2].ival = V;
        ps[3].key = "resident"; ps[3].kind = RAD_P_INT; ps[3].ival = 1;
        RadArgs a{};
        a.t = ts.data(); a.n_t = (int)ts.size();
        a.p = ps; a.n_p = resident ? 4 : 3;
        a.rank = rk; a.world_size = 2;
        for (int64_t c = 0; c < calls; ++c) {
            ts[0].data = &ids[(size_t)(c * M)];
            /* The last call has nothing ahead of it, and says so with an absent operand. */
            ts[4].data = c + 1 < calls ? &ids[(size_t)((c + 1) * M)] : nullptr;
            std::fill(out.begin(), out.end(), (uint16_t)0xdead);
            go.arrive_and_wait();
            if (r->launch(&a, nullptr) != RAD_OK) { ++refused[rk]; continue; }
            for (int64_t m = 0; m < M; ++m) {
                const int32_t id = ids[(size_t)(c * M + m)];
                for (int64_t i = 0; i < n; ++i) {
                    uint16_t want = 0;
                    if (id >= 0 && bf16) std::memcpy(&want, tab + ((int64_t)id * n + i) * 2, 2);
                    else if (id >= 0)    want = lut[tab[(int64_t)id * n + i]];
                    if (out[(size_t)(m * n + i)] != want) { ++bad[rk]; break; }
                }
            }
        }
    };
    const long long locked0 = locked_bytes();
    auto round = [&] {
        std::thread t0(rank, 0), t1(rank, 1);
        t0.join();
        t1.join();
    };
    round();
    /* The copy is made in the background from the first call. Once the locked memory has grown
     * by the table it is in place, or about to be, so a short wait more and every call reads it. */
    bool copied = false;
    if (resident) {
        for (int i = 0; i < 300 && !copied; ++i) {
            copied = locked_bytes() - locked0 >= V * rb;
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        if (copied) round();
    }
    /* THE ADDRESS RANGE STAYS RESERVED. libavx resolves a table to its file once, by the address
     * range of its mapping, for the life of the process (avx_ngram.cpp, table_file) -- the
     * engine's container mapping is one. A case that unmapped here could have the next case's file
     * mapped at the same addresses, and the gather would read this file through the fd it kept:
     * every row of the next case wrong, on whichever run the kernel happens to reuse the range.
     * An inaccessible anonymous mapping over the same range frees the file and keeps the range. */
    if (mmap(map, (size_t)(V * rb), PROT_NONE, MAP_FIXED | MAP_PRIVATE | MAP_ANONYMOUS |
             MAP_NORESERVE, -1, 0) == MAP_FAILED)
        munmap(map, (size_t)(V * rb));
    ::unlink(path.c_str());

    const bool pass = bad[0] + bad[1] + refused[0] + refused[1] == 0;
    const char* from = !resident ? ""
                     : copied    ? ", then as many again from the copy in RAM"
                                 : ", all from the file: the copy was not made here (above)";
    if (verbose || !pass)
        std::printf("  %s in %s (%s): %lld calls x 2 ranks%s, rows wrong %lld/%lld, "
                    "calls refused %lld/%lld %s\n", what, dir.c_str(), granule, (long long)calls,
                    from, (long long)bad[0], (long long)bad[1], (long long)refused[0],
                    (long long)refused[1], pass ? "ok" : "FAIL");
    else
        std::printf("  %s in %s (%s): %lld calls x 2 ranks%s, every row exact\n", what,
                    dir.c_str(), granule, (long long)calls, from);
    return pass;
}
