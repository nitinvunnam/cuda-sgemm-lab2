# Lab 2 — The SGEMM ladder: from naive CUDA to register blocking, with a profiler

| | |
|---|---|
| **Assigned** | Monday, October 26 |
| **Checkpoint Due** | Wednesday, November 4 @ 11:59pm (Canvas) |
| **Final Due** |  Sunday, November 15 @ 11:59pm (Canvas) |
| **Weight** | 12% of final course grade |
| **Work** | Individual |
| **Checkout** | Oral, week of November 16 |


## Overview

We have combined Labs 2 and 3 into this lab to give you more time to work on your final project as the semester closes out.  Note the change from the syllabus in that this lab is individual and is worth 12% of your final grade.  The lab is due on Sunday, November 15 at 11:59pm.  The checkpoint is due on Wednesday, November 4 at 11:59pm.


In this lab, you will work through six versions of a CUDA matrix-multiplication program, starting with a basic implementation and building toward more advanced ones. We call this progression a “ladder” where each version is a “rung” that builds on ideas introduced in the previous versions. You will take one matrix multiply up a ladder of six versions. Each version keeps the arithmetic identical and changes one thing about where the operands come from: first the order in which a warp's threads touch global memory, then whether a tile is staged in shared memory, then how many results each thread keeps in registers. You write five of the six; the fifth rung is provided as a model for the sixth. The harness checks every version against cuBLAS and times it, so after each rung you see exactly what the change bought and how far you still are from the library. Partway up the ladder you will also run two of your kernels across a small range of block shapes and tile sizes, and explain why the same kernel runs at different speeds when only its launch configuration changes.

Each version keeps the arithmetic identical while changing how threads access and reuse data. You will explore:

- The order in which a warp's threads touch global memory.
- Whether a tile is staged in shared memory.
- How many results each thread keeps in registers.

You write five of the six versions; part 5 is provided as a model for part 6. For each version, the harness:

- Checks your results against cuBLAS.
- Measures execution time so you can see what the change bought and how far you still are from the library.

Partway up the ladder you will also run two of your kernels across a small range of block shapes and tile sizes, and explain why the same kernel runs at different speeds when only its launch configuration changes.

Alongside the timings, you will use Nsight Compute (`ncu`), NVIDIA's kernel profiler, to read what the hardware did:

- How many memory sectors each load touched.
- How many shared-memory wavefronts each shared load cost.
- How many registers each thread used.
- How many warps were resident.

The last rung is yours. You will:

- Profile your best kernel and identify what limits it.
- Make one change that the profile argues for.
- Measure and explain the effect of that change.
- Explain which further change you chose not to make and why.

This lab asks for far less measurement than the last one. One batch job at the end collects the final performance results:

- Two timing tables: the ladder and a seven-configuration sweep.
- Two matching profiler tables.
- One full profiler report.

Each table is one command. There are no thread-count curves or repeated runs to aggregate; the harness takes the median for you. Keep your predictions and barrier-experiment outputs as you work, since those are separate from the final batch job. The work is in the kernels and in the reasoning, and the questions say exactly which numbers to use.

This lab is done individually and checked out orally. All graded numbers come from a Lonestar6 A100. After logging into Lonestar6, get a GPU node with one of these commands:

- `idev -p gpu-a100` — production queue.
- `idev -p gpu-a100-dev` — development/debugging queue.
- `idev -p gpu-a100-small` — only 1 GPU (which is all you need) and may be easier to get time on. 

Each command requests a GPU node for 30 minutes by default. Set the session duration with `-t HH:MM:SS`, for example `idev -p gpu-a100-dev -t 01:00:00` for one hour. This is runtime after the job starts, not queue wait time. The maximum duration is **48 hours** for `gpu-a100` and **2 hours** for `gpu-a100-dev`.

 See TACC's [idev user guide](https://docs.tacc.utexas.edu/software/idev/).

**START EARLY!!** It may take a while to get a GPU-enabled node since they are in high demand.

## Time estimates

The estimates below are generous, so plan your weeks around the upper end. Each includes writing a draft of that part's report answers, but none includes time spent waiting in the queue for a GPU node, which can be hours when demand is high.

| Part | What you do | Estimated time | Due |
|---|---|---|---|
| 0 — Setup | Log in, build, get a GPU node, check that `ncu` can read counters | 1–2 hours | Checkpoint (setup); final (answers) |
| 1 — Naive | First CUDA kernel and launcher; sector prediction; roofline question | 2–3 hours | Checkpoint (code); final (answers) |
| 2 — Coalesced | Swap the mapping; compare predictions with `ncu` | 1–2 hours | Checkpoint (code); final (answers) |
| 3 — Tiling | Tiled kernel body; barrier experiment; two predictions | 4–5 hours | Checkpoint (code); final (answers) |
| 4 — Block shape and tile size | One launcher; prediction table by hand; sweep and three questions | 3–4 hours | Checkpoint (code); final (answers) |
| 5 — 1-D blocking | Read the provided kernel; one question | 1–2 hours | Final |
| 6 — 2-D blocking | Hardest kernel in the lab; occupancy by hand; read the full profile | 6–8 hours | Final |
| 7 — Your change | One listed change; prediction and comparison; the change not made | 4–6 hours | Final |
| Final run and report | `sbatch job.slurm`; assemble tables; edit the report to six pages | 3–5 hours | Final |
| **Total** | | **25–37 hours** | |

The checkpoint work (Parts 0 to 3 and the Part 4 launcher) comes to about 9–13 hours, so start as soon as the lab is posted.

## Checkpoint

**Due Wednesday, November 4 @ 11:59pm on Canvas.** The checkpoint is **code only**: Parts 1 to 4 must build, pass `make check`, and pass `make sanitize` on a Lonestar6 A100. No report answers are due at the checkpoint. The questions for Parts 0 to 4 are answered in the final report.

**What you write by the checkpoint:**

| Part | What must work |
|---|---|
| 1 | `sgemm_naive` (always on; it has no switch) |
| 2 | `sgemm_coalesced`, with `HAVE_COALESCED` set to 1 |
| 3 | `sgemm_tiled_kernel<T>`, with `HAVE_TILED` set to 1 |
| 4 | `sgemm_coalesced_shape`, with `sgemm_coalesced` calling it, and `HAVE_SWEEP` set to 1 so the checks cover all seven configurations |

**What you submit on Canvas:**

1. `kernels.cu` and `parts.h`.
2. `check.txt` and `sanitize.txt`, made on a Lonestar6 GPU node with

   ```
   $ make check    2>&1 | tee check.txt
   $ make sanitize 2>&1 | tee sanitize.txt
   ```

**How it is graded:** 10 points, 2.5 for each of Parts 1 to 4. A part earns its 2.5 points if it passes both `make check` and `make sanitize` in the submitted outputs. For Part 4, that means every configuration passes. Nothing else is graded at the checkpoint.

**Not due at the checkpoint:** the Part 3 barrier experiment, the `ncu` tables, the Part 4 prediction table and sweep, and every report question. You do not need to run `make sweep` or `make ncu-sweep` for the checkpoint. When you do run them, fill in the Part 4 prediction table first; the report must show the table as you wrote it before the run.

## Objectives

By the end of this lab you will be able to:

- Write, launch, and check a CUDA kernel, and map a thread index onto a matrix element deliberately.
- Predict, from the mapping alone, how many 32-byte sectors one warp's load touches, and confirm the prediction with `ncu`.
- Stage tiles in shared memory, place the two barriers the tiling needs, and show with `compute-sanitizer` what each one prevents.
- Predict, for a given block shape or tile size, how coalescing, idle lanes, occupancy, and data reuse change, and explain a measured ranking of launch configurations from those four effects.
- Count shared-memory loads per multiply-add for a given blocking scheme, and turn that count into a bound on throughput.
- Compute occupancy by hand from registers per thread, and explain why the fastest kernel in the ladder is the one with the lowest occupancy.
- Read an Nsight Compute report, name the limiter of a kernel, make and evaluate one change that addresses it, and justify one change you chose not to make.

## Background

**The operation.** Every version computes the same matrix operation:

```text
C_new = alpha × (A × B) + beta × C_old
```

First, multiply matrices A and B. Scale the result by `alpha`, then add `beta` times the original values of C. Store the result back in C.

A, B, and C are N × N matrices containing `float` values, already stored in GPU memory. They use row-major storage, meaning one complete row is stored after another. With indices starting at zero, element `(i, j)` of A is accessed as `A[i*N + j]`.

The harness uses `alpha = 1.25` and `beta = −0.75`. Because beta is nonzero, your kernel must read the original C values before replacing them. Ignoring those values produces an incorrect answer.

N is always a multiple of 128, so the required tiles fit evenly into the matrices without partially filled tiles at the edges.

Each of the N² output entries requires N multiply-adds, giving N³ multiply-adds overall. Counting each multiplication and addition separately gives approximately 2N³ floating-point operations, which the lab uses to calculate performance.

**The harness.** `sgemm.cu` allocates A, B, and C, fills them from a fixed seed, and runs the rungs you have switched on in `parts.h`. For each rung it first checks every element against cuBLAS on the same inputs, then times the kernel: three warm-up launches, then launches timed one at a time until a second has accumulated, reporting the median. cuBLAS is held to plain FP32 arithmetic (no TF32 tensor cores), so it does exactly the work you do. Do not modify the measurement.

```
./sgemm RUNG [N]          check and time one rung (plus cuBLAS)
./sgemm all [N]           every implemented rung, then cuBLAS
./sgemm check             every implemented rung, and every Part 4
                          configuration, at N = 128, 384, 1024
./sgemm sweep [N]         the Part 4 configurations, checked and timed
./sgemm RUNG N --once     one launch, nothing else (for ncu and the
                          sanitizers)
```

A ladder line looks like this:

```
rung               ms    GFLOP/s   %peak  %cuBLAS    max_err
reg2d           x.xxx     xxxx.x   xx.x%    xx.x%   1.23e-05  OK
```

`%peak` is against the FP32 peak computed from the device's SM count, its FP32 lanes per SM, and its maximum clock; `%cuBLAS` compares medians taken in the same process.

**The machine.** Lonestar6's GPU nodes each hold three NVIDIA A100 PCIe 40 GB cards; this lab uses one. The figures the questions use:

```
NVIDIA A100-PCIE-40GB, compute capability 8.0
  108 SMs; 4 partitions per SM, 16 FP32 lanes per partition
  64 FP32 lanes per SM: up to 64 FMAs, two warp-instructions, per cycle
  max clock 1410 MHz -> FP32 peak 108 x 64 x 2 x 1.41 GHz = 19.5 TFLOP/s
  HBM2: 40 GB at 1,555 GB/s;  L2: 40 MB
per SM:
  65,536 32-bit registers
  up to 64 resident warps (2,048 threads) and 32 resident blocks
  up to 164 KB of shared memory, 32 banks, one 128-byte wavefront per
  cycle
```

The GPU used as the lecture's example (compute capability 8.9) has 128 FP32 lanes and 48 resident warps per SM. The A100 has 64 and 64. Use the A100's numbers here. The difference is the reason to state the compute capability whenever you quote a per-SM limit.

**The sanitizers.** `compute-sanitizer` is to a CUDA program what valgrind, AddressSanitizer, and ThreadSanitizer are to a CPU program. It ships with the CUDA toolkit and needs no special build. This lab uses three of its tools:

```
memcheck    out-of-bounds and misaligned device accesses; with
            --leak-check full, device memory never freed
racecheck   shared-memory hazards between threads of a block:
            a read and a write, or two writes, with no barrier between
synccheck   __syncthreads() reached by some threads of a block and
            not others
```

`make sanitize` runs all three on every implemented rung, and every Part 4 configuration, at N = 256. This is the lab's equivalent of the TSan requirement: a rung that `make sanitize` reports is treated as broken.

**The profiler.** `ncu` runs a program, intercepts its kernel launches, and replays each profiled kernel as many times as it needs to read the hardware counters you asked for. Two of its behaviors matter here. First, it locks the GPU clock to the card's base frequency while profiling, so kernel times under `ncu` are longer than the ladder's; compare rungs with each other inside the `ncu` table, never `ncu` times with ladder times. Second, it profiles every launch unless told otherwise, which is why `ncu.sh` runs each rung with `--once`.

`make ncu` runs `ncu.sh`, which profiles one launch of each rung, plus cuBLAS's GEMM kernel, and prints one table; `make ncu-sweep` does the same for the Part 4 configurations:

```
rung       ms  regs  occ%  SM%  DRAM%  DRAM_MB  gld_Msec  sec/req  wf/shld
```

The header comment of `ncu.sh` defines each column and gives the metric behind it, so you can ask `ncu` for any of them by hand. The two columns that carry the most weight in the questions are `sec/req`, the 32-byte sectors touched per global-load instruction, and `wf/shld`, the shared-memory wavefronts per shared-load instruction. For 4-byte loads, a `wf/shld` of 1 means no bank conflicts and n means an n-way conflict.

`make ncu-full RUNG=reg2d` collects Nsight Compute's full set of sections for one rung and writes them as text to `reg2d_full.txt`. It also saves `reg2d_full.ncu-rep`, which you can open in the Nsight Compute GUI on your laptop if you prefer reading it there; the GUI is a free download and needs no GPU.

## Setup on Lonestar6

Compile on a login node; run on a GPU node. The login nodes have no GPUs.

```
$ module load gcc cuda
$ cd lab2/starter_files
$ make                        # needs nvcc from CUDA 12 or later
```

If `make` reports an older CUDA, run `module spider cuda` and load a 12.x version by name. Then get a GPU for interactive work:

```
$ idev -p gpu-a100            # or: idev -p gpu-a100-dev
$ module load gcc cuda        # again, on the GPU node
$ nvidia-smi
$ ./sgemm cublas 4096
```

Before the first profile, check that `ncu` may read the counters:

```
$ ncu --metrics sm__cycles_elapsed.avg ./sgemm cublas 1024 --once
```

It must print a number for `sm__cycles_elapsed.avg`. If it prints `ERR_NVGPUCTRPERM`, stop and tell the TAs before you spend more time on the node.

For the graded run, fill in the account in `job.slurm` and `sbatch` it. It requests the `gpu-a100` queue, records the modules, `nvidia-smi`, and the compiler and profiler versions, then runs everything the report uses. A batch job waits in the same queue as `idev`, so submit it well before the deadline. 

Do not use the `gpu-h100` queue: the H100 is a different chip, and the questions are written for the A100.

## Files

```
lab2/
  README.md               this file
  starter_files/
    kernels.cuh           PROVIDED: launcher declarations, tile shapes
    parts.h               PROVIDED: HAVE_* switches, all 0 except the
                          provided reg1d
    kernels.cu            stubs for Parts 1-4, 6, 7; the Part 3
                          launchers and all of Part 5 are provided
    sgemm.cu              provided harness (do-not-modify)
    ncu.sh                one profiled launch per rung -> one table
    Makefile              check / sanitize / ladder / sweep / ncu /
                          ncu-sweep / ncu-full / ptxas targets
    job.slurm             the graded run, on gpu-a100
    AI_LOG.md             record of assistant use
```

## Specifications

**As you complete each part of the lab, cross-reference the [Deliverables](#deliverables) and [Grading](#grading) sections.**

The tile shapes (the dimensions of the matrix pieces processed together) for Parts 3, 5 and 6 are fixed in `kernels.cuh`. Use the named constants rather than literals, so that your code and the report's arithmetic agree. Each part's stub in `kernels.cu` states the mapping or the structure that part requires. Turn on a part's switch in `parts.h` when its kernel is written; `make check` and `make sanitize` pick it up on the next build.

The code for Parts 1 to 4 is due at the **checkpoint** (see [Checkpoint](#checkpoint) for exactly what to submit). Parts 5 to 7 and the report are due at the final deadline. The report questions for Parts 0 to 4 are answered in the final report.

### Part 0 — Set up and check the counters

Do the setup on ls6. Record in the report the two `# device` and `# host` lines the harness prints, the output of `nvcc --version`, and the result of the `ncu` counter check.

### Part 1 — Naive

Write `sgemm_naive`: one thread per element of C, 32 × 32 blocks, and the mapping in the stub, in which `threadIdx.x` selects the row. Each thread computes one dot product of length N and writes `alpha*dot + beta*C`. Run `make check` and `make sanitize`.

Before you profile it, answer in the report: for one warp executing one iteration of the k loop, which addresses do the 32 lanes read from A, and which from B? How many 32-byte sectors does each of those two loads touch? From that, predict the kernel's `sec/req` (the one read of C per thread is negligible).

Then the roofline question. If every operand of every multiply-add came from DRAM, how many flops would the kernel perform per byte moved, and what GFLOP/s would 1,555 GB/s allow? Compare that bound with naive's measured GFLOP/s and with its `DRAM_MB`. Did the operands come from DRAM? If not, where did they come from?

### Part 2 — Coalesced

Write `sgemm_coalesced`: the same kernel with `threadIdx.x` selecting the column. Nothing else changes.

Redo the address analysis from Part 1 for the new mapping and predict `sec/req` again. Compare both predictions with the measured `sec/req` column. Then give the speedup over naive from the ladder, and compare it with the ratio of the two rungs' `gld_Msec`. Are the two ratios the same? Explain why or why not.

### Part 3 — Shared-memory tiling

Write the body of `sgemm_tiled_kernel<T>` as the stub describes: T × T tiles, coalesced mapping, each thread loading one element of each tile, two `__syncthreads()` per tile. The kernel is a template on the tile size T because T sets the size of the `__shared__` arrays. The launchers are provided: `sgemm_tiled` runs it with T = 32 for this part, and Part 4 runs it at other sizes. Write the body in terms of T, never 32. It must pass `make check` and `make sanitize`.

Then run the barrier experiment. Delete the first `__syncthreads()` in the k-tile loop, rebuild, and run

```
$ ./sgemm check
$ compute-sanitizer --tool racecheck ./sgemm tiled 256 --once
```

Record what each command reports. Restore it, delete the second barrier, and do the same. Restore both. In the report, say which hazard each barrier prevents (read-after-write or write-after-read, and between which two accesses), and explain any case in which `make check` and `racecheck` disagree.

Two predictions to check against the `ncu` table:

- Global loads. From the tile size and the sectors each tiled load touches, predict the ratio of coalesced's `gld_Msec` to tiled's. Compare with the measurement.
- The new limiter. In the inner loop as written in the source, how many shared-memory loads does a warp issue per FMA? (The compiler may combine some of them; the source view of the full report shows what it did, but the source count is the one asked for here and in Parts 5 and 6.) Each shared load costs at least one wavefront, an SM serves one wavefront per cycle, and an SM can issue two FMA warp-instructions per cycle. From those numbers, what fraction of the FP32 peak can this kernel reach at most? Compare with tiled's `%peak`.

### Part 4 — Block shape and tile size

Parts 1 to 3 all used 32 × 32 thread blocks. That was a choice, and this part measures what the choice is worth. The arithmetic stays the same in every configuration; only the launch changes.

Write `sgemm_coalesced_shape` as the stub describes: your Part 2 kernel launched with a bx × by block, indexing with `blockDim` instead of 32. Then make `sgemm_coalesced` call it with 32 × 32, so the kernel exists once. The tile-size half needs no new code: the provided `sgemm_tiled_size` already runs your Part 3 kernel at T = 8, 16 and 32. Switch on `HAVE_SWEEP`; `make check` and `make sanitize` now cover every configuration below.

`make sweep` checks and times seven configurations at N = 4096, and `make ncu-sweep` profiles the same seven into one table with the columns of `make ncu`:

```
coalesced kernel, bx x by block:  32x32  32x1  4x4  1x32
tiled kernel, T x T tile:         8      16    32
```

For each configuration the sweep prints the block shape, the threads and warps per block, the number of blocks in the grid, the time, and the percentage of cuBLAS.

**Before you run either**, fill in this table for the four coalesced shapes by hand, and put it in the report as you first wrote it:

```
shape  threads/  warps/  active lanes  blocks/  warps/  occupancy  sec/req
       block     block   per warp      SM       SM
```

- **Blocks per SM** is the smallest of three limits from the A100 figures under Background: 32 block slots, 64 warp slots divided by warps per block, and 65,536 registers divided by the registers one block needs. Take registers per thread from `make ptxas`.
- **Sectors per request:** warps are formed from consecutive thread numbers with x fastest. Work out which rows and columns the 32 lanes of one warp hold for each shape, then count the 32-byte sectors that warp's A load and B load each touch, as in Parts 1 and 2.

Then run `make sweep` and `make ncu-sweep` and answer three questions.

- **The predictions.** Compare your occupancy column with `occ%`, and your `sec/req` column with the measured one. Explain any that disagree.
- **The shapes.** Rank the four coalesced shapes by GFLOP/s and, for each that loses to 32 × 32, name the effect responsible: coalescing, idle lanes, or occupancy, citing the column that shows it. One shape performs almost exactly like a rung you have already written; name the rung and say why. 32 × 1 and 4 × 4 have the same occupancy and the same `sec/req` in your table; compare their times and explain the difference.
- **Tile size.** As T goes from 8 to 16 to 32, how do these change: global-load sectors per FMA, barriers per FMA, the number of warps that wait at each barrier, and shared memory per block? From those, predict the ordering of the three tile sizes. Then explain the measured ordering. Is the largest tile the fastest? If not, why not?

### Part 5 — 1-D register blocking (provided)

`sgemm_reg1d` is provided and switched on: a 64 × 64 block of C per thread block, 512 threads, BK = 8, and each thread computing TM = 8 results in one column, with the B element held in a register across all eight products. Read it closely; Part 6 extends the same structure to two dimensions.

Count the shared-memory loads per FMA in its inner loop, and by what factor that count fell from tiled. Compare that factor with the measured speedup over tiled. Name at least one other change between the two kernels that contributes to the difference, and point to the column of the `ncu` table that shows it.

### Part 6 — 2-D register blocking

Write `sgemm_reg2d` as the stub describes: a 128 × 128 block of C per thread block, 256 threads, BK = 8, and an 8 × 8 block of results per thread held in registers. It must pass `make check` and `make sanitize`.

Three questions.

- **Shared loads.** How many shared-memory loads per FMA does the inner loop issue now? Redo the Part 3 bound with this count. Is shared-memory bandwidth still the limit?
- **Occupancy, by hand.** Take reg2d's registers per thread from the `regs` column (or `make ptxas`). From the A100's 65,536 registers per SM, how many blocks of 256 threads fit on one SM, how many warps is that, and what occupancy, out of 64 warps? Compare with the `occ%` column. reg2d has the lowest occupancy in the ladder and the highest throughput. Explain why that is not a contradiction.
- **The limiter.** Run `make ncu-full RUNG=reg2d` and read four sections of `reg2d_full.txt`: GPU Speed Of Light Throughput, Occupancy, Memory Workload Analysis, and Warp State Statistics. In one paragraph, name what limits reg2d: which unit is busiest, which stall reason dominates, and what reg2d's `wf/shld` says about its shared-memory accesses. Quote the numbers you use.

### Part 7 — One change the profile asks for, and one it does not

Write `sgemm_mine`: start from your reg2d and make exactly one of these changes, the one that best addresses the limiter you named in Part 6.

- Store the A slice transposed in shared memory, so that a thread's TM values of A are contiguous and can be read with `float4` loads.
- Use `float4` loads from global memory when filling the shared tiles.
- Prefetch the next slice into registers while computing on the current one (double buffering).
- Change the tile shapes (BM, BN, BK, TM, TN), if Part 6 says registers or occupancy are the problem.

Any tile shapes you change are yours; define them in `kernels.cu`. It must handle every N that is a multiple of 128, and pass `make check` and `make sanitize`.

Before you write it, record in the report the metric that motivates the change, its value for reg2d, and your prediction: which columns of the `ncu` table should move, in which direction, and roughly how far. After it works, compare the `reg2d` and `mine` rows of the final ladder and `ncu` tables, and explain the result. A change that does not pay off earns full credit if the explanation accounts for why, with numbers from the profile.

Then one paragraph on the change you did not make. Name a second optimization that your reg2d or `mine` profile points to (one from the list above, or another), the metric that points to it, and why you chose not to make it: what it would cost in code or in resources, and why you expect the payoff to be small. Back the expectation with a number from your tables.

## Deliverables

At the checkpoint (details under [Checkpoint](#checkpoint)):

- `kernels.cu`, and `parts.h` with `HAVE_COALESCED`, `HAVE_TILED`, and `HAVE_SWEEP` set to 1. The naive rung is always on and has no switch.
- `check.txt` and `sanitize.txt`: the output of `make check` and `make sanitize` from a Lonestar6 GPU node.

At the final deadline:

- `kernels.cu`, and `parts.h` with every switch set to 1.
- `lab2.o<jobid>` from one run of `job.slurm` with every switch at 1, together with the `ladder.txt`, `sweep.txt`, `ncu.txt`, `ncu_sweep.txt`, and `reg2d_full.txt` it wrote.
- The output of the four Part 3 barrier-experiment commands.
- A report (PDF, six pages or fewer) answering each part's questions, with the ladder, sweep, and both `ncu` tables included as the tools printed them, and the Part 4 prediction table as you wrote it before the run.
- `AI_LOG.md`.

## Checklist

- [ ] `make` is warning-free with `-std=c++20 -Wall -Wextra`.
- [ ] Checkpoint: `kernels.cu`, `parts.h`, `check.txt`, and `sanitize.txt` submitted by November 4, with Parts 1 to 4 passing both.
- [ ] Every rung passes `make check`, and `make sanitize` reports no errors with every switch at 1.
- [ ] Part 0: device and host lines, `nvcc --version`, and the counter check recorded.
- [ ] Part 1: per-load sector prediction made before profiling; roofline bound computed and compared with GFLOP/s and `DRAM_MB`.
- [ ] Part 2: both `sec/req` predictions compared with measurement; speedup compared with the `gld_Msec` ratio.
- [ ] Part 3: barrier experiment recorded and explained; `gld_Msec` ratio and shared-memory bound predicted and checked.
- [ ] Part 4: prediction table filled in before the run; predictions compared with `occ%` and `sec/req`; the four shapes ranked and explained, including the rung one shape reproduces and the 32 × 1 against 4 × 4 comparison; the tile-size ordering explained.
- [ ] Part 5: shared loads per FMA counted for the provided kernel; speedup compared and the other change named.
- [ ] Part 6: shared-load bound redone; occupancy computed by hand and compared; limiter named from `reg2d_full.txt` with numbers.
- [ ] Part 7: one listed change made; motivating metric and prediction recorded before the change; before/after rows compared and explained; the change not made justified with a number.
- [ ] Report is six pages or fewer; tables are as the tools printed them, from an LS6 A100.

## Checkout

The checkout is oral, on a Lonestar6 A100. A TA builds your submission, runs `make check`, `make sanitize`, and `./sgemm all 4096`, and then asks about specific lines of specific kernels: why a barrier is where it is, what a given index expression evaluates to for a given thread, how many registers a thread holds, what the provided reg1d kernel's indexing does.

## Grading

100 points. The checkpoint is graded on the code alone. Final code is checked by the TAs at checkout with every switch in `parts.h` set to 1; the report is graded against the questions each part asks.

**Rules that apply before points are counted.** A rung that fails `make check`, or for which `make sanitize` reports any error, earns zero code points; its report points still count. A build with warnings loses 5 points. A report over six pages loses 5 points. A rung you cannot explain at the checkout loses its code points. A table whose header does not show an A100 on a Lonestar6 host earns no points for the questions that use it.

| Part | Item | Points |
|---|---|---|
| Checkpoint (10) | Parts 1 to 4 each pass `make check` and `make sanitize` on Lonestar6 by the checkpoint date (2.5 each) | 10 |
| 0 — Setup (4) | Device and host lines, `nvcc --version` recorded | 2 |
| | `ncu` counter check recorded | 2 |
| 1 — Naive (9) | `sgemm_naive` correct and sanitizer-clean, with the specified mapping | 4 |
| | Per-lane addresses and sectors per load for A and B; `sec/req` predicted before profiling | 2 |
| | Flops per byte and the DRAM bound; compared with measured GFLOP/s and `DRAM_MB`, with where the operands came from | 3 |
| 2 — Coalesced (7) | `sgemm_coalesced` correct and sanitizer-clean | 2 |
| | `sec/req` predicted for the new mapping; both predictions compared with measurement | 2 |
| | Speedup compared with the `gld_Msec` ratio and the difference explained | 3 |
| 3 — Tiling (14) | `sgemm_tiled_kernel<T>` correct at every T, racecheck-clean, two barriers | 4 |
| | Barrier experiment: four outputs recorded; hazard each barrier prevents; any disagreement between the two tools explained | 4 |
| | `gld_Msec` ratio predicted from the tile size and checked | 2 |
| | Shared loads per FMA and the resulting bound; compared with `%peak` | 4 |
| 4 — Block shape and tile size (11) | `sgemm_coalesced_shape` correct and sanitizer-clean at every shape; `sgemm_coalesced` calls it | 1 |
| | Prediction table complete and made before the run: lanes, blocks and warps per SM, occupancy, `sec/req` | 4 |
| | Predictions compared with `occ%` and `sec/req`; disagreements explained | 2 |
| | Shapes ranked with the effect behind each loss; the rung one shape reproduces; 32 × 1 against 4 × 4 | 2 |
| | Tile-size trends stated and the measured ordering explained | 2 |
| 5 — 1-D blocking (4) | Shared loads per FMA and the factor; compared with the speedup; another change named and shown in the `ncu` table | 4 |
| 6 — 2-D blocking (17) | `sgemm_reg2d` correct and sanitizer-clean | 6 |
| | Shared-load count and bound redone | 2 |
| | Occupancy computed by hand and compared with `occ%`; why the lowest occupancy is the fastest | 4 |
| | Limiter named from the full report: busiest unit, dominant stall, `wf/shld`, with numbers | 5 |
| 7 — Your change (18) | `sgemm_mine` makes one listed change; correct and sanitizer-clean for every N that is a multiple of 128 | 5 |
| | Motivating metric, its value, and a prediction, recorded before the change | 4 |
| | `reg2d` and `mine` compared in both final tables | 2 |
| | Result explained from the profile, whether or not it paid off | 4 |
| | The change not made: metric, cost, and expected payoff, with a number | 3 |
| Report quality (6) | Numbers wherever the questions ask for them; no restating of the assignment | 2 |
| | Tables included as the tools printed them, from the graded run | 3 |
| | `AI_LOG.md` | 1 |
| **Total** | | **100** |
