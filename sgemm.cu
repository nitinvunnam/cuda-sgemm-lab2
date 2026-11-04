// sgemm.cu -- Lab 2 harness.  PROVIDED; do not modify the measurement.
//
//   ./sgemm RUNG [N] [--once]
//   ./sgemm check
//   ./sgemm sweep [N]
//   ./sgemm list      the rungs switched on in parts.h, one per line
//   ./sgemm list-sweep    the Part 4 configurations, one per line
//
//   RUNG    naive | coalesced | tiled | reg1d | reg2d | mine | cublas
//           | all      (every rung switched on in parts.h, then cuBLAS)
//           | coalesced:BXxBY  | tiled:T     (one Part 4 configuration)
//   N       matrix order, a multiple of 128; default 4096
//   --once  launch the rung exactly once and exit: no warm-up, no
//           timing, no check, and no cuBLAS unless RUNG is cublas.
//           ncu.sh and the sanitizers use this mode so that each run
//           contains exactly one launch of the kernel under study.
//           `all --once` also launches every Part 4 configuration
//           once, so the sanitizers cover them.
//
// Every rung computes C = alpha*A*B + beta*C on N x N row-major float
// matrices, alpha = 1.25 and beta = -0.75.  beta is nonzero so that a
// kernel that ignores the old C fails the check.  A, B and the initial
// C are uniform in [-1, 1] from a fixed seed, so every run of every
// rung sees the same inputs.
//
// Timed mode, for each rung:
//   1. check: run once from the initial C and compare every element
//      with cuBLAS on the same inputs.  A rung that fails is reported
//      and not timed.  The tolerance is 1e-5 * N absolute, far above
//      the rounding difference between two correct summation orders
//      and far below what any indexing error produces.
//   2. time: 3 warm-up launches, then launches timed one at a time
//      with CUDA events until 1 s has accumulated (at least 5, at most
//      25).  The figure reported is the MEDIAN, in ms and in GFLOP/s
//      counting 2*N^3 flops per call.
//
// The percentage of cuBLAS compares medians from the same process on
// the same GPU.  cuBLAS is held to plain FP32 (CUBLAS_DEFAULT_MATH; no
// TF32 tensor cores), so it does the same arithmetic you do.
//
// `check` runs every implemented rung, and every Part 4 configuration,
// at N = 128, 384 and 1024 and exits non-zero if any of them fails.
// It is the first thing the TAs run at checkout.
//
// `sweep` checks and times the Part 4 configurations: the coalesced
// kernel at four block shapes and the tiled kernel at three tile
// sizes, with the block shape, threads and warps per block, and grid
// size printed beside each time.  Occupancy is left for you.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include <unistd.h>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "parts.h"
#include "kernels.cuh"

#define CUDA_CHECK(x)                                                     \
    do {                                                                  \
        cudaError_t e_ = (x);                                             \
        if (e_ != cudaSuccess) {                                          \
            std::fprintf(stderr, "%s:%d: %s\n  -> %s\n", __FILE__,        \
                         __LINE__, #x, cudaGetErrorString(e_));           \
            std::exit(2);                                                 \
        }                                                                 \
    } while (0)

#define CUBLAS_CHECK(x)                                                   \
    do {                                                                  \
        cublasStatus_t s_ = (x);                                          \
        if (s_ != CUBLAS_STATUS_SUCCESS) {                                \
            std::fprintf(stderr, "%s:%d: %s -> cuBLAS status %d\n",       \
                         __FILE__, __LINE__, #x, static_cast<int>(s_));   \
            std::exit(2);                                                 \
        }                                                                 \
    } while (0)

namespace {

constexpr float ALPHA = 1.25f;
constexpr float BETA  = -0.75f;

using Launcher = std::function<void(int, float, const float*,
                                    const float*, float, float*)>;
struct Rung {
    std::string name;
    Launcher    fn;
    int         bx = 0, by = 0;        // thread-block shape, Part 4 only
};

cublasHandle_t g_cublas = nullptr;

// Row-major C = A*B is column-major C^T = B^T * A^T, so cuBLAS gets
// the operands in the other order.  No transposes are performed.
void sgemm_cublas(int N, float alpha, const float* A, const float* B,
                  float beta, float* C)
{
    CUBLAS_CHECK(cublasSgemm(g_cublas, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
                             &alpha, B, N, A, N, &beta, C, N));
}

std::vector<Rung> implemented()
{
    std::vector<Rung> r{{"naive", sgemm_naive}};
#if HAVE_COALESCED
    r.push_back({"coalesced", sgemm_coalesced});
#endif
#if HAVE_TILED
    r.push_back({"tiled", sgemm_tiled});
#endif
#if HAVE_REG1D
    r.push_back({"reg1d", sgemm_reg1d});
#endif
#if HAVE_REG2D
    r.push_back({"reg2d", sgemm_reg2d});
#endif
#if HAVE_MINE
    r.push_back({"mine", sgemm_mine});
#endif
    return r;
}

// Part 4: the coalesced kernel at four block shapes, the tiled kernel
// at three tile sizes.  Every shape divides 128, so every N.
std::vector<Rung> sweep_rungs()
{
    std::vector<Rung> r;
#if HAVE_SWEEP
    const int shapes[][2] = {{32, 32}, {32, 1}, {4, 4}, {1, 32}};
    for (const auto& s : shapes) {
        const int bx = s[0], by = s[1];
        r.push_back({"coalesced:" + std::to_string(bx) + "x" +
                         std::to_string(by),
                     [bx, by](int N, float al, const float* A,
                              const float* B, float be, float* C) {
                         sgemm_coalesced_shape(bx, by, N, al, A, B, be, C);
                     },
                     bx, by});
    }
    for (int t : {8, 16, 32})
        r.push_back({"tiled:" + std::to_string(t),
                     [t](int N, float al, const float* A, const float* B,
                         float be, float* C) {
                         sgemm_tiled_size(t, N, al, A, B, be, C);
                     },
                     t, t});
#endif
    return r;
}

// FP32 lanes per SM, for the "% of FP32 peak" column.
int fp32_lanes_per_sm(int major, int minor)
{
    if (major == 7) return 64;                       // Volta, Turing
    if (major == 8 && minor == 0) return 64;         // A100
    if (major >= 8) return 128;                      // GA10x, Ada, H100, ...
    return 0;                                        // unknown: no column
}

struct Device {
    std::string name;
    int major = 0, minor = 0, sms = 0, clock_khz = 0;
    double gib = 0, peak_gflops = 0;
};

Device describe_device()
{
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp p{};
    CUDA_CHECK(cudaGetDeviceProperties(&p, dev));
    Device d;
    d.name  = p.name;
    d.major = p.major;
    d.minor = p.minor;
    d.sms   = p.multiProcessorCount;
    d.gib   = static_cast<double>(p.totalGlobalMem) / (1 << 30);
    CUDA_CHECK(cudaDeviceGetAttribute(&d.clock_khz, cudaDevAttrClockRate,
                                      dev));
    d.peak_gflops = 2.0 * d.sms * fp32_lanes_per_sm(d.major, d.minor) *
                    d.clock_khz * 1e-6;
    return d;
}

void print_header(const Device& d, int N)
{
    char host[256] = "?";
    gethostname(host, sizeof host - 1);
    std::printf("# host %s\n", host);
    std::printf("# device %s  CC %d.%d  %d SMs  %d MHz  %.0f GiB\n",
                d.name.c_str(), d.major, d.minor, d.sms,
                d.clock_khz / 1000, d.gib);
    if (d.peak_gflops > 0)
        std::printf("# FP32 peak %.0f GFLOP/s (SMs x lanes x 2 x max "
                    "clock)\n", d.peak_gflops);
    if (N > 0)
        std::printf("# N=%d  alpha=%.2f beta=%.2f  %.3g flop per call  "
                    "median of timed launches\n",
                    N, ALPHA, BETA, 2.0 * N * N * static_cast<double>(N));
}

// Device buffers for one problem size, filled from a fixed seed.
struct Problem {
    int    N = 0;
    float* A   = nullptr;
    float* B   = nullptr;
    float* C0  = nullptr;      // initial C, never written
    float* C   = nullptr;      // the rung's output
    float* Ref = nullptr;      // cuBLAS's output from C0
    bool   have_ref = false;

    explicit Problem(int n) : N(n)
    {
        const std::size_t count = static_cast<std::size_t>(N) * N;
        const std::size_t bytes = count * sizeof(float);
        std::vector<float> h(count);
        std::mt19937 rng(379);
        std::uniform_real_distribution<float> u(-1.0f, 1.0f);
        float* dst[3];
        for (int m = 0; m < 3; ++m) {
            for (auto& x : h) x = u(rng);
            CUDA_CHECK(cudaMalloc(&dst[m], bytes));
            CUDA_CHECK(cudaMemcpy(dst[m], h.data(), bytes,
                                  cudaMemcpyHostToDevice));
        }
        A = dst[0];
        B = dst[1];
        C0 = dst[2];
        CUDA_CHECK(cudaMalloc(&C, bytes));
        CUDA_CHECK(cudaMalloc(&Ref, bytes));
        reset_C();
    }
    Problem(const Problem&) = delete;
    Problem& operator=(const Problem&) = delete;
    ~Problem()
    {
        cudaFree(A);
        cudaFree(B);
        cudaFree(C0);
        cudaFree(C);
        cudaFree(Ref);
    }

    std::size_t bytes() const
    {
        return static_cast<std::size_t>(N) * N * sizeof(float);
    }
    void reset_C()
    {
        CUDA_CHECK(cudaMemcpy(C, C0, bytes(), cudaMemcpyDeviceToDevice));
    }

    // cuBLAS from C0, then a double-precision spot check of cuBLAS
    // itself, so that a layout mistake in this file cannot pass.
    void make_reference()
    {
        if (have_ref) return;
        CUDA_CHECK(cudaMemcpy(Ref, C0, bytes(), cudaMemcpyDeviceToDevice));
        sgemm_cublas(N, ALPHA, A, B, BETA, Ref);
        CUDA_CHECK(cudaDeviceSynchronize());

        const std::size_t count = static_cast<std::size_t>(N) * N;
        std::vector<float> a(count), b(count), c0(count), ref(count);
        CUDA_CHECK(cudaMemcpy(a.data(), A, bytes(), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(b.data(), B, bytes(), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(c0.data(), C0, bytes(),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(ref.data(), Ref, bytes(),
                              cudaMemcpyDeviceToHost));
        std::mt19937 rng(42);
        std::uniform_int_distribution<int> pick(0, N - 1);
        for (int s = 0; s < 16; ++s) {
            const int i = pick(rng), j = pick(rng);
            const std::size_t row = static_cast<std::size_t>(i) * N;
            double acc = 0;
            for (int k = 0; k < N; ++k)
                acc += static_cast<double>(a[row + k]) *
                       b[static_cast<std::size_t>(k) * N + j];
            const std::size_t ij = static_cast<std::size_t>(i) * N + j;
            const double want = ALPHA * acc + BETA * c0[ij];
            if (std::fabs(want - ref[ij]) > 1e-5 * N) {
                std::fprintf(stderr, "harness: cuBLAS reference disagrees "
                             "with the CPU at (%d,%d): %g vs %g\n",
                             i, j, ref[ij], want);
                std::exit(2);
            }
        }
        have_ref = true;
    }
};

// Launch once from the initial C and compare with the reference.
// Returns the largest absolute error, or +inf for a NaN or a fault.
double check_rung(Problem& p, const Rung& r)
{
    p.make_reference();
    p.reset_C();
    r.fn(p.N, ALPHA, p.A, p.B, BETA, p.C);
    cudaError_t e = cudaGetLastError();
    if (e == cudaSuccess) e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", r.name.c_str(),
                     cudaGetErrorString(e));
        std::exit(2);       // a fault poisons the context; stop here
    }
    const std::size_t count = static_cast<std::size_t>(p.N) * p.N;
    std::vector<float> got(count), want(count);
    CUDA_CHECK(cudaMemcpy(got.data(), p.C, p.bytes(),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(want.data(), p.Ref, p.bytes(),
                          cudaMemcpyDeviceToHost));
    double worst = 0;
    std::size_t at = 0;
    for (std::size_t i = 0; i < count; ++i) {
        const double d = std::fabs(static_cast<double>(got[i]) - want[i]);
        if (!(d <= worst)) {            // also catches NaN
            worst = std::isnan(d) ? INFINITY : d;
            at = i;
            if (std::isinf(worst)) break;
        }
    }
    if (worst > 1e-5 * p.N)
        std::fprintf(stderr, "%s: N=%d first worst element C[%zu][%zu] = "
                     "%g, expected %g\n", r.name.c_str(), p.N, at / p.N,
                     at % p.N,
                     got[at], want[at]);
    return worst;
}

// Median milliseconds per launch.
double time_rung(Problem& p, const Rung& r)
{
    for (int w = 0; w < 3; ++w)
        r.fn(p.N, ALPHA, p.A, p.B, BETA, p.C);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    std::vector<float> ms;
    double total = 0;
    while (ms.size() < 25 && (ms.size() < 5 || total < 1000.0)) {
        CUDA_CHECK(cudaEventRecord(start));
        r.fn(p.N, ALPHA, p.A, p.B, BETA, p.C);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float t = 0;
        CUDA_CHECK(cudaEventElapsedTime(&t, start, stop));
        ms.push_back(t);
        total += t;
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    std::sort(ms.begin(), ms.end());
    return ms[ms.size() / 2];
}

void print_row(const Device& d, int N, const char* name, double ms,
               double cublas_ms, double err, bool ok)
{
    const double gflops = 2.0 * N * N * static_cast<double>(N) / ms * 1e-6;
    std::printf("%-10s %10.3f %10.1f", name, ms, gflops);
    if (d.peak_gflops > 0)
        std::printf(" %6.1f%%", 100.0 * gflops / d.peak_gflops);
    std::printf(" %7.1f%%", cublas_ms > 0 ? 100.0 * cublas_ms / ms : 100.0);
    if (err >= 0)
        std::printf("  %9.2e  %s\n", err, ok ? "OK" : "FAIL");
    else
        std::printf("  %9s  %s\n", "-", "reference");
    std::fflush(stdout);
}

// One line of the Part 4 table.
void print_sweep_row(int N, const Rung& r, double ms, double cublas_ms,
                     bool ok)
{
    const double gflops = 2.0 * N * N * static_cast<double>(N) / ms * 1e-6;
    const int threads = r.bx * r.by;
    const long blocks = static_cast<long>(N / r.bx) * (N / r.by);
    char shape[16];
    std::snprintf(shape, sizeof shape, "%dx%d", r.bx, r.by);
    std::printf("%-15s %6s %5d %5d %8ld %9.3f %9.1f %6.1f%%  %s\n",
                r.name.c_str(), shape, threads, (threads + 31) / 32, blocks,
                ms, gflops, ok ? 100.0 * cublas_ms / ms : 0.0,
                ok ? "OK" : "FAIL");
    std::fflush(stdout);
}

int usage()
{
    std::fprintf(stderr,
                 "usage: sgemm RUNG [N] [--once]\n"
                 "       sgemm check | sweep [N] | list | list-sweep\n"
                 "RUNG: naive coalesced tiled reg1d reg2d mine cublas all\n"
                 "      coalesced:BXxBY tiled:T  (see list-sweep)\n"
                 "N: a multiple of %d (default 4096)\n", N_MULTIPLE);
    return 1;
}

} // namespace

int main(int argc, char** argv)
{
    if (argc < 2) return usage();
    const std::string which = argv[1];
    if (which == "list") {          // for the Makefile; needs no GPU
        for (const Rung& r : implemented())
            std::printf("%s\n", r.name.c_str());
        return 0;
    }
    if (which == "list-sweep") {
        for (const Rung& r : sweep_rungs())
            std::printf("%s\n", r.name.c_str());
        return 0;
    }
    int  N    = 4096;
    bool once = false;
    for (int a = 2; a < argc; ++a) {
        if (std::strcmp(argv[a], "--once") == 0) once = true;
        else N = std::atoi(argv[a]);
    }
    if (N <= 0 || N % N_MULTIPLE != 0) {
        std::fprintf(stderr, "N must be a positive multiple of %d\n",
                     N_MULTIPLE);
        return 1;
    }

    CUBLAS_CHECK(cublasCreate(&g_cublas));
    CUBLAS_CHECK(cublasSetMathMode(g_cublas, CUBLAS_DEFAULT_MATH));
    const std::vector<Rung> all = implemented();
    const std::vector<Rung> sweep = sweep_rungs();
    std::vector<Rung> everything = all;
    everything.insert(everything.end(), sweep.begin(), sweep.end());
    const Device dev = describe_device();
    int status = 0;

    if (which == "check") {
        print_header(dev, 0);
        for (int n : {128, 384, 1024}) {
            Problem p(n);
            for (const Rung& r : everything) {
                const double err = check_rung(p, r);
                const bool ok = err <= 1e-5 * n;
                std::printf("check %-15s N=%-5d max_err %9.2e  %s\n",
                            r.name.c_str(), n, err, ok ? "OK" : "FAIL");
                status |= !ok;
            }
        }
    } else if (which == "sweep") {
        if (sweep.empty()) {
            std::fprintf(stderr, "Part 4 is not switched on in parts.h\n");
            return 1;
        }
        Problem p(N);
        print_header(dev, N);
        std::printf("%-15s %6s %5s %5s %8s %9s %9s %7s\n", "config",
                    "block", "thr", "warps", "blocks", "ms", "GFLOP/s",
                    "%cuBLAS");
        const double cublas_ms = time_rung(p, {"cublas", sgemm_cublas});
        for (const Rung& r : sweep) {
            const bool ok = check_rung(p, r) <= 1e-5 * N;
            status |= !ok;
            print_sweep_row(N, r, ok ? time_rung(p, r) : INFINITY,
                            cublas_ms, ok);
        }
    } else {
        std::vector<Rung> run;
        if (which == "all")
            run = once ? everything : all;
        else if (which == "cublas")
            run = {};
        else {
            for (const Rung& r : everything)
                if (which == r.name) run = {r};
            if (run.empty()) {
                std::fprintf(stderr, "rung '%s' is not implemented "
                             "(see parts.h) or not a rung\n", which.c_str());
                return usage();
            }
        }
        Problem p(N);

        if (once) {
            if (which == "cublas")
                sgemm_cublas(N, ALPHA, p.A, p.B, BETA, p.C);
            for (const Rung& r : run)
                r.fn(N, ALPHA, p.A, p.B, BETA, p.C);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        } else {
            print_header(dev, N);
            std::printf("%-10s %10s %10s%s %8s  %9s\n", "rung", "ms",
                        "GFLOP/s", dev.peak_gflops > 0 ? "   %peak" : "",
                        "%cuBLAS", "max_err");
            const Rung cublas{"cublas", sgemm_cublas};
            const double cublas_ms = time_rung(p, cublas);
            for (const Rung& r : run) {
                const double err = check_rung(p, r);
                const bool ok = err <= 1e-5 * N;
                status |= !ok;
                const double ms = ok ? time_rung(p, r) : INFINITY;
                print_row(dev, N, r.name.c_str(), ms, cublas_ms, err, ok);
            }
            print_row(dev, N, "cublas", cublas_ms, cublas_ms, -1, true);
        }
    }

    CUBLAS_CHECK(cublasDestroy(g_cublas));
    return status;
}
