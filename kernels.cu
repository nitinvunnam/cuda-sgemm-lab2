/*
 * ================================================================
 *
 * Filename:    kernels.cu
 *
 * Description: Lab 2 -- the SGEMM ladder.  One kernel and one
 *              launcher per rung.
 *
 * Created:     TODO
 * EID:         TODO
 * Author:      TODO
 * Provenance:  I certify that all code contained herein is mine
 *              alone except where otherwise noted.
 *
 * ================================================================
 */

// Every launcher computes C = alpha*A*B + beta*C for N x N row-major
// matrices in device memory (element (i,j) of A is A[i*N + j]),
// launches on the default stream, and returns without synchronizing.
// N is a multiple of 128.  The tile shapes are fixed in kernels.cuh;
// use the constants, not literals, so that the report's arithmetic
// and your code agree.
//
// Each stub below compiles and does nothing, so `make check` fails a
// rung until you write it.  Replace the [[maybe_unused]] stubs; the
// attribute is only there so the empty bodies build warning-free.
// Code marked PROVIDED is complete: read it, do not change it.

#include <cuda_runtime.h>

#include "kernels.cuh"

// ---------------------------------------------------------------------
// Part 1 -- naive.
//
// One thread per element of C.  32 x 32 thread blocks, an
// (N/32) x (N/32) grid, and this mapping, exactly:
//
//     row = blockIdx.x * 32 + threadIdx.x;
//     col = blockIdx.y * 32 + threadIdx.y;
//
// Each thread computes the dot product of row `row` of A with column
// `col` of B, then writes alpha*dot + beta*C[row][col].
// ---------------------------------------------------------------------
__global__ void naive_kernel(int N, float alpha,
                             const float* A, const float* B,
                             float beta, float* C)
{
    int row = blockIdx.x * 32 + threadIdx.x;
    int col = blockIdx.y * 32 + threadIdx.y;

    float dot = 0.0f;

    for (int k = 0; k < N; k++) {
        dot += A[row * N + k] * B[k * N + col];
    }

    int idx = row * N + col;
    C[idx] = alpha * dot + beta * C[idx];
}

void sgemm_naive([[maybe_unused]] int N, [[maybe_unused]] float alpha,
                 [[maybe_unused]] const float* A,
                 [[maybe_unused]] const float* B,
                 [[maybe_unused]] float beta, [[maybe_unused]] float* C)
{
    dim3 block(32, 32);
    dim3 grid(N / 32, N / 32);

    naive_kernel<<<grid, block>>>(N, alpha, A, B, beta, C);
}

// ---------------------------------------------------------------------
// Part 2 -- coalesced.
//
// The same kernel with the roles of x and y exchanged:
//
//     row = blockIdx.y * 32 + threadIdx.y;
//     col = blockIdx.x * 32 + threadIdx.x;
//
// Nothing else changes.  (Part 4 generalizes this kernel; see below.)
// ---------------------------------------------------------------------
__global__ void coalesced_kernel(int N, float alpha,
                                 const float* A, const float* B,
                                 float beta, float* C)
{
    int row = blockIdx.y * 32 + threadIdx.y;
    int col = blockIdx.x * 32 + threadIdx.x;

    float dot = 0.0f;

    for (int k = 0; k < N; k++) {
        dot += A[row * N + k] * B[k * N + col];
    }

    int idx = row * N + col;
    C[idx] = alpha * dot + beta * C[idx];
}
void sgemm_coalesced([[maybe_unused]] int N, [[maybe_unused]] float alpha,
                     [[maybe_unused]] const float* A,
                     [[maybe_unused]] const float* B,
                     [[maybe_unused]] float beta, [[maybe_unused]] float* C)
{
    dim3 block(32,32);
    dim3 grid(N/32, N/32);

    coalesced_kernel<<<grid, block>>>(N, alpha, A, B, beta, C);
}

// ---------------------------------------------------------------------
// Part 3 -- shared-memory tiling.
//
// T x T thread blocks, coalesced mapping as in Part 2.  Walk the K
// dimension one tile at a time: each thread loads one element of the
// A tile and one of the B tile into __shared__ arrays, the block
// synchronizes, each thread accumulates T products from shared
// memory, and the block synchronizes again before the next tile
// overwrites the arrays.
//
// The kernel is a template on the tile size T, because T sets the
// size of the __shared__ arrays and must be known at compile time.
// The launchers below are PROVIDED: sgemm_tiled runs T = TILE (32)
// for this part, and Part 4 runs T = 8, 16 and 32 through
// sgemm_tiled_size.  Write only the kernel body, using T, never TILE
// or a literal.
// ---------------------------------------------------------------------

template <int T>
__global__ void sgemm_tiled_kernel([[maybe_unused]] int N,
                                   [[maybe_unused]] float alpha,
                                   [[maybe_unused]] const float* A,
                                   [[maybe_unused]] const float* B,
                                   [[maybe_unused]] float beta,
                                   [[maybe_unused]] float* C)
{
    __shared__ float As[T][T];
    __shared__ float Bs[T][T];

    int row = blockIdx.y * T + threadIdx.y;
    int col = blockIdx.x * T + threadIdx.x;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    float dot = 0.0f;

    for(int t = 0; t < N; t += T) {

        // Each thread loads one element of A and B
        As[ty][tx] = A[row * N + t + tx];
        Bs[ty][tx] = B[(t + ty) * N + col];

        __syncthreads();

        for (int k = 0; k < T; k++) {
            dot += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    int idx = row * N + col;
    C[idx] = alpha * dot + beta * C[idx];
}

// PROVIDED: launch the kernel instantiated for one tile size.
template <int T>
void launch_tiled(int N, float alpha, const float* A, const float* B,
                  float beta, float* C)
{
    const dim3 block(T, T);
    const dim3 grid(N / T, N / T);
    sgemm_tiled_kernel<T><<<grid, block>>>(N, alpha, A, B, beta, C);
}

// PROVIDED: pick the instantiation at run time.
void sgemm_tiled_size(int tile, int N, float alpha, const float* A,
                      const float* B, float beta, float* C)
{
    switch (tile) {
    case 8:  launch_tiled<8>(N, alpha, A, B, beta, C);  break;
    case 16: launch_tiled<16>(N, alpha, A, B, beta, C); break;
    case 32: launch_tiled<32>(N, alpha, A, B, beta, C); break;
    default: break;          // the harness only asks for 8, 16, 32
    }
}

// PROVIDED: Part 3's rung is the TILE x TILE instantiation.
void sgemm_tiled(int N, float alpha, const float* A, const float* B,
                 float beta, float* C)
{
    sgemm_tiled_size(TILE, N, alpha, A, B, beta, C);
}

// ---------------------------------------------------------------------
// Part 4 -- block shape.
//
// Generalize the Part 2 kernel so the harness can sweep its block
// shape.  sgemm_coalesced_shape launches it with a bx x by thread
// block and an (N/bx) x (N/by) grid.  Inside the kernel, index with
// blockDim, not with 32:
//
//     row = blockIdx.y * blockDim.y + threadIdx.y;
//     col = blockIdx.x * blockDim.x + threadIdx.x;
//
// Then make sgemm_coalesced call sgemm_coalesced_shape(32, 32, ...),
// so the kernel exists once.  (The tile-size half of Part 4 needs no
// new code: sgemm_tiled_size above already runs your Part 3 kernel at
// T = 8, 16 and 32.)
// ---------------------------------------------------------------------

void sgemm_coalesced_shape([[maybe_unused]] int bx, [[maybe_unused]] int by,
                           [[maybe_unused]] int N,
                           [[maybe_unused]] float alpha,
                           [[maybe_unused]] const float* A,
                           [[maybe_unused]] const float* B,
                           [[maybe_unused]] float beta,
                           [[maybe_unused]] float* C)
{
    // TODO
}

// ---------------------------------------------------------------------
// Part 5 -- 1-D register blocking.  PROVIDED: read it before you write
// Part 6, which extends the same structure to two dimensions.
//
// One thread block computes a reg1d::BM x reg1d::BN (64 x 64) block of
// C with reg1d::THREADS (512) threads in a 1-D block.  Shared arrays
// hold a BM x BK slice of A and a BK x BN slice of B; each thread
// loads one element of each per slice.  Each thread owns TM (8)
// consecutive rows of ONE column of the C block and keeps those TM
// partial sums in a register array.  In the inner loop, for each k of
// the slice, the B element is loaded once into a register and used
// for all TM products.
// ---------------------------------------------------------------------

namespace {
namespace r1 = reg1d;
static_assert(r1::BM * r1::BK == r1::THREADS, "one A element per thread");
static_assert(r1::BK * r1::BN == r1::THREADS, "one B element per thread");
}

__global__ void __launch_bounds__(reg1d::THREADS)
sgemm_reg1d_kernel(int N, float alpha, const float* __restrict__ A,
                   const float* __restrict__ B, float beta,
                   float* __restrict__ C)
{
    using namespace reg1d;
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    const int tid       = threadIdx.x;
    const int threadCol = tid % BN;
    const int threadRow = tid / BN;            // owns rows threadRow*TM ..
    const int aRow = tid / BK, aCol = tid % BK;
    const int bRow = tid / BN, bCol = tid % BN;

    // Move A, B and C to this block's corner.
    A += blockIdx.y * BM * N;
    B += blockIdx.x * BN;
    C += blockIdx.y * BM * N + blockIdx.x * BN;

    float acc[TM] = {};
    for (int t = 0; t < N; t += BK) {
        As[aRow * BK + aCol] = A[aRow * N + aCol];
        Bs[bRow * BN + bCol] = B[bRow * N + bCol];
        __syncthreads();
        A += BK;                                // next slice of A ...
        B += BK * N;                            // ... and of B
        for (int k = 0; k < BK; ++k) {
            const float b = Bs[k * BN + threadCol];
            for (int i = 0; i < TM; ++i)
                acc[i] += As[(threadRow * TM + i) * BK + k] * b;
        }
        __syncthreads();
    }
    for (int i = 0; i < TM; ++i) {
        float& c = C[(threadRow * TM + i) * N + threadCol];
        c = alpha * acc[i] + beta * c;
    }
}

void sgemm_reg1d(int N, float alpha, const float* A, const float* B,
                 float beta, float* C)
{
    const dim3 grid(N / reg1d::BN, N / reg1d::BM);
    sgemm_reg1d_kernel<<<grid, reg1d::THREADS>>>(N, alpha, A, B, beta, C);
}

// ---------------------------------------------------------------------
// Part 6 -- 2-D register blocking.
//
// One thread block computes a reg2d::BM x reg2d::BN block of C with
// reg2d::THREADS threads.  Each thread owns a TM x TN block of C and
// keeps TM*TN partial sums in registers.  For each k of the BK slice,
// copy TM elements of the A column and TN elements of the B row from
// shared memory into two small register arrays, then do the TM*TN
// products from registers.  Each thread loads more than one element
// of each shared tile per slice; work out how many from the shapes.
// The Part 5 kernel above is the model to extend.
// ---------------------------------------------------------------------

void sgemm_reg2d([[maybe_unused]] int N, [[maybe_unused]] float alpha,
                 [[maybe_unused]] const float* A,
                 [[maybe_unused]] const float* B,
                 [[maybe_unused]] float beta, [[maybe_unused]] float* C)
{
    // TODO
}

// ---------------------------------------------------------------------
// Part 7 -- your kernel.
//
// Start from your reg2d and make ONE of the changes listed in the
// README, the one your Part 6 profile argues for.  Any tile shapes you
// change are yours; define them here, not in kernels.cuh.  It must
// handle every N that is a multiple of 128.
// ---------------------------------------------------------------------

void sgemm_mine([[maybe_unused]] int N, [[maybe_unused]] float alpha,
                [[maybe_unused]] const float* A,
                [[maybe_unused]] const float* B,
                [[maybe_unused]] float beta, [[maybe_unused]] float* C)
{
    // TODO
}
