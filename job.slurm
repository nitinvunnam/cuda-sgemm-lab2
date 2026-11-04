#!/bin/bash
#----------------------------------------------------------------------
# ECE 379K Lab 2 -- SLURM batch script for one Lonestar6 A100.
#
#     sbatch job.slurm
#     squeue -u $USER
#     cat lab2.o<jobid>
#
# Everything the report needs comes out of one run of this script:
# the device, the check, the sanitizers, the ladder, the sweep, both
# ncu tables, and the full ncu report for reg2d.  Submit the
# lab2.o<jobid> file with the rest of your deliverables.
#
# gpu-a100 gives the job a whole node with three A100s; the lab uses
# one.  If that queue is backed up, gpu-a100-small (one A100 of a
# shared node) works the same way.  Fill in the account from the TACC
# session.
#----------------------------------------------------------------------
#SBATCH -J lab2
#SBATCH -o lab2.o%j
#SBATCH -e lab2.e%j
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:45:00
#SBATCH -A TODO_ACCOUNT

set -x
module load gcc cuda
module list 2>&1
nvidia-smi
nvcc --version
ncu --version
hostname

# Plain FP32 everywhere: no TF32 tensor cores behind cuBLAS's back.
export NVIDIA_TF32_OVERRIDE=0

make clean
make
make check
make sanitize
make ladder
make sweep
make ncu
make ncu-sweep
make ncu-full RUNG=reg2d
make ptxas
