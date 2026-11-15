#!/bin/bash
# Submitted by nccl_bench.sh.
source "$VPD_REPO/vpd/env.sh"; unset NCCL_P2P_LEVEL
nvidia-smi topo -m
run() {
  TAG=$1 MASTER_ADDR=$(hostname) MASTER_PORT=$((29500 + RANDOM % 1000)) srun --ntasks=$SLURM_NTASKS --cpu-bind=none \
    bash -c 'RANK=$SLURM_PROCID LOCAL_RANK=$SLURM_LOCALID WORLD_SIZE=$SLURM_NTASKS exec "$VPD_REPO/.venv/bin/python" "$VPD_REPO/vpd/tools/nccl_bench.py"'
}
run default
NCCL_P2P_LEVEL=SYS run p2p_level_sys
