#!/bin/bash
#SBATCH -J lab2
#SBATCH -o lab2.o%j
#SBATCH -e lab2.e%j
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:45:00
#SBATCH -A TRA25006

set -x
set -e
set -o pipefail

module load gcc
module swap gcc nvidia/24.7
module load cuda/12.5

export CUBLAS_PATH=/home1/apps/nvidia/Linux_aarch64/24.7/math_libs/12.5/targets/sbsa-linux
export CPATH=$CUBLAS_PATH/include:$CPATH
export LIBRARY_PATH=$CUBLAS_PATH/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$CUBLAS_PATH/lib:$LD_LIBRARY_PATH

module list 2>&1
nvidia-smi
nvcc --version
ncu --version
hostname

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