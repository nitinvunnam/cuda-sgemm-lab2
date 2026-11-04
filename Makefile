# Lab 2 -- the SGEMM ladder, on a Lonestar6 A100 node.
#
#   make                 build ./sgemm (CUDA 12+, -std=c++20, sm_80)
#   make check           every implemented rung at N = 128, 384, 1024
#   make sanitize        compute-sanitizer memcheck (with leak check),
#                        racecheck and synccheck on every implemented
#                        rung                                <-- required
#   make ladder          the timed ladder at N (default 4096) -> ladder.txt
#   make sweep           Part 4: block shapes and tile sizes, timed,
#                        at N                               -> sweep.txt
#   make ncu             the metric table, every rung + cuBLAS -> ncu.txt
#   make ncu-sweep       the metric table for the Part 4 sweep
#                                                       -> ncu_sweep.txt
#   make ncu-full RUNG=reg2d
#                        full Nsight Compute report for one rung
#                        -> reg2d_full.ncu-rep and reg2d_full.txt
#   make ptxas           registers and shared memory per kernel, as
#                        the compiler allocated them
#   make clean
#
# Compiling works on a login node.  Everything that runs a kernel
# needs a GPU node: idev -p gpu-a100-dev, or sbatch job.slurm.

NVCC    ?= nvcc
ARCH    ?= sm_80
NVFLAGS ?= -std=c++20 -O3 -arch=$(ARCH) -lineinfo -Xcompiler -Wall,-Wextra
LDLIBS  := -lcublas
SRCS    := sgemm.cu kernels.cu
HDRS    := kernels.cuh parts.h

# N: ladder and ncu size.  SAN_N: sanitizer size (racecheck is slow).
# RUNG: the target of ncu-full.
N      ?= 4096
SAN_N  ?= 256
RUNG   ?= reg2d

NVCC_MAJOR := $(shell $(NVCC) --version 2>/dev/null | \
                sed -n 's/.*release \([0-9]*\)\..*/\1/p')

all: sgemm

sgemm: $(SRCS) $(HDRS)
	@test "$(NVCC_MAJOR)" -ge 12 2>/dev/null || { \
	    echo "need nvcc from CUDA 12 or later for -std=c++20;" \
	         "found '$(NVCC_MAJOR)'.  See the Setup section."; exit 1; }
	$(NVCC) $(NVFLAGS) -o $@ $(SRCS) $(LDLIBS)

check: sgemm
	./sgemm check

sanitize: sgemm
	compute-sanitizer --tool memcheck --leak-check full \
	    --error-exitcode 1 ./sgemm all $(SAN_N) --once
	compute-sanitizer --tool racecheck --error-exitcode 1 \
	    ./sgemm all $(SAN_N) --once
	compute-sanitizer --tool synccheck --error-exitcode 1 \
	    ./sgemm all $(SAN_N) --once

ladder: sgemm
	./sgemm all $(N) | tee ladder.txt

sweep: sgemm
	./sgemm sweep $(N) | tee sweep.txt

ncu: sgemm
	bash ncu.sh ./sgemm $(N) $$(./sgemm list) cublas | tee ncu.txt

ncu-sweep: sgemm
	bash ncu.sh ./sgemm $(N) $$(./sgemm list-sweep) | tee ncu_sweep.txt

ncu-full: sgemm
	ncu --set full --import-source on -f -o $(RUNG)_full \
	    ./sgemm $(RUNG) $(N) --once
	ncu --import $(RUNG)_full.ncu-rep --page details > $(RUNG)_full.txt
	@echo "wrote $(RUNG)_full.ncu-rep and $(RUNG)_full.txt"

ptxas: $(SRCS) $(HDRS)
	$(NVCC) $(NVFLAGS) -Xptxas -v -c kernels.cu -o /dev/null 2>&1 | \
	    grep -E "Compiling entry|Used" | cu++filt

clean:
	rm -f sgemm *.ncu-rep *_full.txt

.PHONY: all check sanitize ladder sweep ncu ncu-sweep ncu-full ptxas clean
