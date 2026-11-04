// kernels.cuh -- the launchers you implement, and the tile shapes each
// rung must use.  PROVIDED; do not change the declarations or the
// constants (sgemm_mine is the exception: its constants are yours).
//
// Every launcher computes
//
//     C = alpha * A * B + beta * C
//
// for N x N row-major float matrices that already live in device
// memory, launches on the default stream, and returns WITHOUT
// synchronizing.  The harness times the launch with CUDA events and
// checks for launch and execution errors itself.
//
// N is always a multiple of 128 (the harness refuses anything else),
// so no rung needs to handle a partial tile.  Every index into A, B
// or C is still a bug if it can leave [0, N*N).

#ifndef LAB2_KERNELS_CUH
#define LAB2_KERNELS_CUH

// Part 1: one thread per element of C, threadIdx.x selects the ROW.
void sgemm_naive    (int N, float alpha, const float* A, const float* B,
                     float beta, float* C);

// Part 2: one thread per element of C, threadIdx.x selects the COLUMN.
void sgemm_coalesced(int N, float alpha, const float* A, const float* B,
                     float beta, float* C);

// Part 3: TILE x TILE tiles of A and B staged in shared memory
// (launcher provided; you write the kernel body).
void sgemm_tiled    (int N, float alpha, const float* A, const float* B,
                     float beta, float* C);

// Part 4: the Part 2 kernel with a bx x by thread block (blockDim.x
// = bx), and the Part 3 kernel with tile x tile tiles, tile in
// {8, 16, 32}.  sgemm_coalesced should become a call to
// sgemm_coalesced_shape(32, 32, ...), so the kernel exists once;
// sgemm_tiled_size is provided in kernels.cu.  bx, by and tile divide
// every N the harness uses.
void sgemm_coalesced_shape(int bx, int by, int N, float alpha,
                           const float* A, const float* B, float beta,
                           float* C);
void sgemm_tiled_size     (int tile, int N, float alpha, const float* A,
                           const float* B, float beta, float* C);

// Part 5 (provided): each thread computes TM elements of one column
// of C.
void sgemm_reg1d    (int N, float alpha, const float* A, const float* B,
                     float beta, float* C);

// Part 6: each thread computes a TM x TN block of C.
void sgemm_reg2d    (int N, float alpha, const float* A, const float* B,
                     float beta, float* C);

// Part 7: your own kernel.
void sgemm_mine     (int N, float alpha, const float* A, const float* B,
                     float beta, float* C);

// ---- tile shapes -----------------------------------------------------

inline constexpr int N_MULTIPLE = 128;   // every N the harness uses

// Parts 1-3: 32 x 32 thread blocks; Part 3 tiles are TILE x TILE.
inline constexpr int TILE = 32;

// Part 5: a BM x BN block of C per thread block, BK-wide slices of the
// K dimension, TM results per thread -> BM*BN/TM = 512 threads.
namespace reg1d {
inline constexpr int BM = 64, BN = 64, BK = 8, TM = 8;
inline constexpr int THREADS = BM * BN / TM;
}

// Part 6: a BM x BN block of C per thread block, a TM x TN block per
// thread -> BM*BN/(TM*TN) = 256 threads.
namespace reg2d {
inline constexpr int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;
inline constexpr int THREADS = BM * BN / (TM * TN);
}

#endif // LAB2_KERNELS_CUH
