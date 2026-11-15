#!/bin/bash
# One SPD worker per Slurm task (launched by srun from train.sbatch). Using srun instead of
# torchrun means every worker receives SIGTERM directly and can checkpoint before exiting;
# torchrun's agent SIGKILLs workers ~30 s after it gets SIGTERM, too short for a checkpoint.
set -euo pipefail
source "$VPD_REPO/vpd/env.sh"
# Single-task runs stay non-distributed (the toy scripts never call init_distributed).
if (( SLURM_NTASKS > 1 )); then
  export RANK=$SLURM_PROCID LOCAL_RANK=$SLURM_LOCALID WORLD_SIZE=$SLURM_NTASKS
  export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
  export MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
fi
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
cd "$VPD_REPO"
exec .venv/bin/python -u "$@"
