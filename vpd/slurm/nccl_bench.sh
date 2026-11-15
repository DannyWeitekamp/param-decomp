#!/bin/bash
# NCCL all-reduce bandwidth with default settings vs NCCL_P2P_LEVEL=SYS.
#   vpd/slurm/nccl_bench.sh <n_gpus> [extra sbatch args...]
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
NGPU=$1; shift
sbatch --export=ALL,VPD_REPO="$VPD_REPO" $VPD_SBATCH_ARGS --nodes=1 --ntasks=$NGPU \
  --gres=gpu:${VPD_GPU_TYPE:+$VPD_GPU_TYPE:}$NGPU -c 4 --mem=64G -t 00:20:00 -J vpd-nccl \
  -o "$VPD_ROOT/logs/nccl-%j.out" "$@" "$VPD_REPO/vpd/slurm/nccl_bench.sbatch"
